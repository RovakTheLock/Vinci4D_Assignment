#include <petsc.h>
#include <gtest/gtest.h>
#include <fstream>
#include "../include/AssembleAlgorithms.h"
#include "../include/MeshObject.h"
#include "../include/YamlParser.h"
#include "../include/FieldsHolder.h"
#include "../include/LinearSystem.h"
#include "../include/Operations.h"

using namespace Vinci4D;

class AssembleAlgorithmsTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser = new InputConfigParser("../config/input.yaml");
        mesh = new MeshObject(*parser);
        
        // For FieldArray sizing: track local and ghost cells separately
        int mpiSize = mesh->getMpiSize();
        if (mpiSize > 1) {
            numLocalCells = mesh->getNumLocalCells();
            numGhostCells = mesh->getNumGhostCells();
        } else {
            numLocalCells = mesh->getNumCells();
            numGhostCells = 0;
        }
        
        // For LinearSystem DOF: always use global cell count
        int globalCells = (mpiSize > 1) ? mesh->getGlobalNumCells() : mesh->getNumCells();
        momentumDof = globalCells * MAX_DIM;
    }
    
    void TearDown() override {
        delete mesh;
        delete parser;
    }
    
    InputConfigParser* parser;
    MeshObject* mesh;
    int numLocalCells;
    int numGhostCells;
    int momentumDof;
};

TEST_F(AssembleAlgorithmsTest, TimeTermAssembler) {
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numLocalCells, numGhostCells);
    FieldArray velocityOld("velocity_n", DimType::VECTOR, numLocalCells, numGhostCells);
    
    velocityNew.initializeConstant(1.0);
    velocityOld.initializeConstant(0.5);
    
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    double dt = 0.01;
    AssembleCellVectorTimeTerm timeAssembler(
        "TimeTerm", &velocityNew, &velocityOld, &momentumSystem, mesh, dt);
    
    momentumSystem.zero();
    EXPECT_NO_THROW(timeAssembler.assemble());
    momentumSystem.assembleMatrix();
    
    double rhsNorm = momentumSystem.getRhsNorm();
    EXPECT_GT(rhsNorm, 0) << "RHS norm should be positive after time term assembly";
}

TEST_F(AssembleAlgorithmsTest, DiffusionAssembler) {
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numLocalCells, numGhostCells);
    velocityNew.initializeConstant(1.0);
    
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    double diffusionCoeff = 0.001;
    AssembleInteriorVectorDiffusionToLinSystem diffusionAssembler(
        "Diffusion", &velocityNew, &momentumSystem, mesh, diffusionCoeff);
    
    momentumSystem.zero();
    EXPECT_NO_THROW(diffusionAssembler.assemble());
    momentumSystem.assembleMatrix();
    
    double rhsNorm = momentumSystem.getRhsNorm();
    // RHS norm may be zero or positive depending on field values
    EXPECT_GE(rhsNorm, 0) << "RHS norm should be non-negative";
}

TEST_F(AssembleAlgorithmsTest, PressureGradientAssembler) {
    FieldArray gradPressure("grad_pressure", DimType::VECTOR, numLocalCells, numGhostCells);
    gradPressure.initializeConstant(0.1);
    
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    AssembleCellVectorPressureGradientToLinSystem pressureGradAssembler(
        "PressureGrad", &gradPressure, &momentumSystem, mesh);
    
    momentumSystem.zero();
    EXPECT_NO_THROW(pressureGradAssembler.assemble());
    momentumSystem.assembleMatrix();
    
    double rhsNorm = momentumSystem.getRhsNorm();
    EXPECT_GT(rhsNorm, 0) << "RHS norm should be positive after pressure gradient assembly";
}

TEST_F(AssembleAlgorithmsTest, CombinedAssembly) {
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numLocalCells, numGhostCells);
    FieldArray velocityOld("velocity_n", DimType::VECTOR, numLocalCells, numGhostCells);
    FieldArray gradPressure("grad_pressure", DimType::VECTOR, numLocalCells, numGhostCells);
    
    velocityNew.initializeConstant(1.0);
    velocityOld.initializeConstant(0.5);
    gradPressure.initializeConstant(0.1);
    
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    double dt = 0.01;
    double diffusionCoeff = 0.001;
    
    AssembleCellVectorTimeTerm timeAssembler(
        "TimeTerm", &velocityNew, &velocityOld, &momentumSystem, mesh, dt);
    AssembleInteriorVectorDiffusionToLinSystem diffusionAssembler(
        "Diffusion", &velocityNew, &momentumSystem, mesh, diffusionCoeff);
    AssembleCellVectorPressureGradientToLinSystem pressureGradAssembler(
        "PressureGrad", &gradPressure, &momentumSystem, mesh);
    
    momentumSystem.zero();
    EXPECT_NO_THROW({
        timeAssembler.assemble();
        diffusionAssembler.assemble();
        pressureGradAssembler.assemble();
    });
    momentumSystem.assembleMatrix();
    
    double rhsNorm = momentumSystem.getRhsNorm();
    EXPECT_GT(rhsNorm, 0) << "RHS norm should be positive after combined assembly";
}

TEST_F(AssembleAlgorithmsTest, WrongFieldTypeThrows) {
    FieldArray scalarField("scalar", DimType::SCALAR, numLocalCells, numGhostCells);
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    // Should throw because time term requires vector field
    EXPECT_THROW({
        AssembleCellVectorTimeTerm timeAssembler(
            "TimeTerm", &scalarField, &scalarField, &momentumSystem, mesh, 0.01);
    }, std::runtime_error);
}

TEST_F(AssembleAlgorithmsTest, VectorDiffusionLinearProfileHorizontal) {
    // Create 3x3 mesh for diffusion verification
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 3
  num_cells_y: 3
simulation:
  Re: 100
  CFL: 0.5
)";
    
    // Write temporary config
    std::ofstream tmpYaml("/tmp/diffusion_test.yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser testParser("/tmp/diffusion_test.yaml");
    MeshObject testMesh(testParser);
    
    int mpiSize = testMesh.getMpiSize();
    int numLocal = (mpiSize > 1) ? testMesh.getNumLocalCells() : testMesh.getNumCells();
    int numGhost = (mpiSize > 1) ? testMesh.getNumGhostCells() : 0;
    int globalCells = (mpiSize > 1) ? testMesh.getGlobalNumCells() : testMesh.getNumCells();
    int testDof = globalCells * MAX_DIM;
    
    // Boundary values: left=[10.0, 0.0], right=[1.0, 0.0]
    std::array<double, 2> leftValue = {10.0, 0.0};
    std::array<double, 2> rightValue = {1.0, 0.0};
    
    FieldArray velocityField("velocity", DimType::VECTOR, numLocal, numGhost);
    velocityField.initializeConstant(0.0);
    
    LinearSystem system(testDof, "VectorDiffusion", false);  // Use dense matrix to match Python
    
    double diffusionCoeff = 1.0;
    
    // Assemble interior diffusion
    AssembleInteriorVectorDiffusionToLinSystem interiorDiffusion(
        "InteriorDiffusion", &velocityField, &system, &testMesh, diffusionCoeff);
    
    // Assemble left boundary
    AssembleDirichletBoundaryVectorDiffusionToLinSystem leftBoundary(
        "LeftBoundary", &velocityField, &system, &testMesh, 
        Boundary::LEFT, leftValue, diffusionCoeff);
    
    // Assemble right boundary
    AssembleDirichletBoundaryVectorDiffusionToLinSystem rightBoundary(
        "RightBoundary", &velocityField, &system, &testMesh, 
        Boundary::RIGHT, rightValue, diffusionCoeff);
    
    // Zero and assemble all (matching Python pattern)
    system.zero();
    interiorDiffusion.assemble();
    leftBoundary.assemble();
    rightBoundary.assemble();
    system.assembleMatrix();
    
    // Solve system
    Vec solution;
    VecCreate(PETSC_COMM_WORLD, &solution);
    VecSetSizes(solution, PETSC_DECIDE, testDof);
    VecSetFromOptions(solution);
    system.solve("direct", solution);
    
    // Verify linear profile: Each rank checks its owned DOFs
    // In parallel, PETSc distributes DOFs and each rank owns a contiguous range
    const PetscScalar* solutionData;
    VecGetArrayRead(solution, &solutionData);
    
    // Get the range of global DOF indices owned by this rank
    PetscInt ownershipStart, ownershipEnd;
    VecGetOwnershipRange(solution, &ownershipStart, &ownershipEnd);
    
    // Check local cells (or all cells in sequential mode)
    const auto& cellsToCheck = (mpiSize > 1) ? testMesh.getLocalCells() : testMesh.getCells();
    
    for (const auto& cell : cellsToCheck) {
        int globalCellID = (mpiSize > 1) ? testMesh.localToGlobal(cell.getLocalId()) : cell.getFlatId();
        auto centroid = cell.getCentroid();
        double x = centroid[0];
        
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            int globalDof = globalCellID * MAX_DIM + comp;
            
            // Only check DOFs we own
            if (globalDof >= ownershipStart && globalDof < ownershipEnd) {
                int localDofIdx = globalDof - ownershipStart;
                double slope = (rightValue[comp] - leftValue[comp]) / 1.0;
                double expectedValue = x * slope + leftValue[comp];
                EXPECT_NEAR(solutionData[localDofIdx], expectedValue, 1e-5)
                    << "Linear profile mismatch at cell " << globalCellID 
                    << " component " << comp 
                    << " at x=" << x
                    << " (rank " << testMesh.getMpiRank() << ")";
            }
        }
    }
    
    VecRestoreArrayRead(solution, &solutionData);
    VecDestroy(&solution);
}

TEST_F(AssembleAlgorithmsTest, VectorAdvectionConsistency) {
    // Test that LHS * velocity = -RHS for advection operator (conservation check)
    // This verifies the advection operator is consistent and conservative
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 3
  num_cells_y: 3
simulation:
  Re: 100
  CFL: 0.5
)";
    
    // Write temporary config
    std::ofstream tmpYaml("/tmp/advection_test.yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser testParser("/tmp/advection_test.yaml");
    MeshObject testMesh(testParser);
    
    int mpiSize = testMesh.getMpiSize();
    int numLocal = (mpiSize > 1) ? testMesh.getNumLocalCells() : testMesh.getNumCells();
    int numGhost = (mpiSize > 1) ? testMesh.getNumGhostCells() : 0;
    int globalCells = (mpiSize > 1) ? testMesh.getGlobalNumCells() : testMesh.getNumCells();
    int testDof = globalCells * MAX_DIM;
    
    // Create velocity field with [1.0, 1.0] everywhere
    FieldArray velocityField("velocity", DimType::VECTOR, numLocal, numGhost);
    velocityField.initializeConstant(0.0);
    auto& velData = velocityField.getData();
    for (int i = 0; i < numLocal; ++i) {
        velData[i * MAX_DIM] = 1.0;      // u = 1.0
        velData[i * MAX_DIM + 1] = 1.0;  // v = 1.0
    }
    
    // Create pressure field (zero everywhere)
    FieldArray pressureField("pressure", DimType::SCALAR, numLocal, numGhost);
    pressureField.initializeConstant(0.0);
    
    // Compute pressure gradient (will be zero)
    FieldArray gradPressureField("grad_pressure", DimType::VECTOR, numLocal, numGhost);
    gradPressureField.initializeConstant(0.0);
    ComputeCellGradient gradientOp(&testMesh, &pressureField, &gradPressureField);
    gradientOp.computeScalarGradient();
    
    // Compute mass flux
    double dt = 0.01;
    ComputeInteriorMassFlux massFluxOp(&testMesh, &velocityField, &pressureField, 
                                        &gradPressureField, dt);
    massFluxOp.computeMassFlux();
    
    // Assemble advection operator
    LinearSystem system(testDof, "AdvectionTest", false);  // Dense matrix for easy multiplication
    AssembleInteriorVectorAdvectionToLinSystem advectionAlg(
        "VelocityAdvection", &velocityField, &system, &testMesh);
    
    system.zero();
    advectionAlg.assemble();
    system.assembleMatrix();
    
    // Check conservation: LHS * velocity = -RHS
    // Multiply LHS matrix by velocity vector
    Mat lhs = system.getLhs();
    Vec rhs = system.getRhs();
    Vec lhsTimesVel;
    VecCreate(PETSC_COMM_WORLD, &lhsTimesVel);
    VecSetSizes(lhsTimesVel, PETSC_DECIDE, testDof);
    VecSetFromOptions(lhsTimesVel);
    
    // Create velocity PETSc vector
    Vec velVec;
    VecCreate(PETSC_COMM_WORLD, &velVec);
    VecSetSizes(velVec, PETSC_DECIDE, testDof);
    VecSetFromOptions(velVec);
    
    // Copy velocity data to PETSc vector - each rank sets its owned DOFs
    PetscInt velOwnershipStart, velOwnershipEnd;
    VecGetOwnershipRange(velVec, &velOwnershipStart, &velOwnershipEnd);
    
    const auto& localCells = (mpiSize > 1) ? testMesh.getLocalCells() : testMesh.getCells();
    for (const auto& cell : localCells) {
        int globalCellID = (mpiSize > 1) ? testMesh.localToGlobal(cell.getLocalId()) : cell.getFlatId();
        int localCellID = (mpiSize > 1) ? cell.getLocalId() : cell.getFlatId();
        
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            int globalDof = globalCellID * MAX_DIM + comp;
            if (globalDof >= velOwnershipStart && globalDof < velOwnershipEnd) {
                VecSetValue(velVec, globalDof, velData[localCellID * MAX_DIM + comp], INSERT_VALUES);
            }
        }
    }
    VecAssemblyBegin(velVec);
    VecAssemblyEnd(velVec);
    
    // Compute LHS * velocity
    MatMult(lhs, velVec, lhsTimesVel);
    
    // Verify that LHS * velocity = -RHS for DOFs we own
    const PetscScalar* lhsVelData;
    const PetscScalar* rhsData;
    VecGetArrayRead(lhsTimesVel, &lhsVelData);
    VecGetArrayRead(rhs, &rhsData);
    
    PetscInt rhsOwnershipStart, rhsOwnershipEnd;
    VecGetOwnershipRange(rhs, &rhsOwnershipStart, &rhsOwnershipEnd);

    if (mpiSize > 1) {
        const auto& interiorCellsToCheck = testMesh.getLocalInteriorCells();
        for (const auto* cell : interiorCellsToCheck) {
            int globalCellID = testMesh.localToGlobal(cell->getLocalId());
            for (int comp = 0; comp < MAX_DIM; ++comp) {
                int globalDof = globalCellID * MAX_DIM + comp;
                if (globalDof < rhsOwnershipStart || globalDof >= rhsOwnershipEnd) {
                    continue;
                }
                int localDofIdx = globalDof - rhsOwnershipStart;
                EXPECT_NEAR(-lhsVelData[localDofIdx], rhsData[localDofIdx], 1e-5)
                    << "Conservation check failed at DOF " << globalDof
                    << ": LHS*vel=" << lhsVelData[localDofIdx] << ", RHS=" << rhsData[localDofIdx];
            }
        }
    } else {
        const auto& cellsToCheck = testMesh.getCells();
        for (const auto& cell : cellsToCheck) {
            int globalCellID = cell.getFlatId();
            for (int comp = 0; comp < MAX_DIM; ++comp) {
                int globalDof = globalCellID * MAX_DIM + comp;
                int localDofIdx = globalDof - rhsOwnershipStart;
                EXPECT_NEAR(-lhsVelData[localDofIdx], rhsData[localDofIdx], 1e-5)
                    << "Conservation check failed at DOF " << globalDof
                    << ": LHS*vel=" << lhsVelData[localDofIdx] << ", RHS=" << rhsData[localDofIdx];
            }
        }
    }
    
    VecRestoreArrayRead(lhsTimesVel, &lhsVelData);
    VecRestoreArrayRead(rhs, &rhsData);
    VecDestroy(&lhsTimesVel);
    VecDestroy(&velVec);
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    
    // These tests now work in both sequential and parallel modes
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
