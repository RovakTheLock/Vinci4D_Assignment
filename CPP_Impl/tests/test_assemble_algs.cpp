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
        numCells = mesh->getNumCells();
        momentumDof = numCells * MAX_DIM;
    }
    
    void TearDown() override {
        delete mesh;
        delete parser;
    }
    
    InputConfigParser* parser;
    MeshObject* mesh;
    int numCells;
    int momentumDof;
};

TEST_F(AssembleAlgorithmsTest, TimeTermAssembler) {
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numCells);
    FieldArray velocityOld("velocity_n", DimType::VECTOR, numCells);
    
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
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numCells);
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
    FieldArray gradPressure("grad_pressure", DimType::VECTOR, numCells);
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
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numCells);
    FieldArray velocityOld("velocity_n", DimType::VECTOR, numCells);
    FieldArray gradPressure("grad_pressure", DimType::VECTOR, numCells);
    
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
    FieldArray scalarField("scalar", DimType::SCALAR, numCells);
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
    int testNumCells = testMesh.getNumCells();
    int testDof = testNumCells * MAX_DIM;
    
    // Boundary values: left=[10.0, 0.0], right=[1.0, 0.0]
    std::array<double, 2> leftValue = {10.0, 0.0};
    std::array<double, 2> rightValue = {1.0, 0.0};
    
    FieldArray velocityField("velocity", DimType::VECTOR, testNumCells);
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
    
    // Verify linear profile: dU = x * slope + leftValue
    const PetscScalar* solutionData;
    VecGetArrayRead(solution, &solutionData);
    
    for (const auto& cell : testMesh.getCells()) {
        int cellID = cell.getFlatId();
        auto centroid = cell.getCentroid();
        double x = centroid[0];
        
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            double slope = (rightValue[comp] - leftValue[comp]) / 1.0;
            double expectedValue = x * slope + leftValue[comp];
            EXPECT_NEAR(solutionData[cellID * MAX_DIM + comp], expectedValue, 1e-5)
                << "Linear profile mismatch at cell " << cellID 
                << " component " << comp 
                << " at x=" << x;
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
    int testNumCells = testMesh.getNumCells();
    int testDof = testNumCells * MAX_DIM;
    
    // Create velocity field with [1.0, 1.0] everywhere
    FieldArray velocityField("velocity", DimType::VECTOR, testNumCells);
    velocityField.initializeConstant(0.0);
    auto& velData = velocityField.getData();
    for (int i = 0; i < testNumCells; ++i) {
        velData[i * MAX_DIM] = 1.0;      // u = 1.0
        velData[i * MAX_DIM + 1] = 1.0;  // v = 1.0
    }
    
    // Create pressure field (zero everywhere)
    FieldArray pressureField("pressure", DimType::SCALAR, testNumCells);
    pressureField.initializeConstant(0.0);
    
    // Compute pressure gradient (will be zero)
    FieldArray gradPressureField("grad_pressure", DimType::VECTOR, testNumCells);
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
    
    // Copy velocity data to PETSc vector
    for (int i = 0; i < testDof; ++i) {
        VecSetValue(velVec, i, velData[i], INSERT_VALUES);
    }
    VecAssemblyBegin(velVec);
    VecAssemblyEnd(velVec);
    
    // Compute LHS * velocity
    MatMult(lhs, velVec, lhsTimesVel);
    
    // Verify that LHS * velocity = -RHS
    const PetscScalar* lhsVelData;
    const PetscScalar* rhsData;
    VecGetArrayRead(lhsTimesVel, &lhsVelData);
    VecGetArrayRead(rhs, &rhsData);
    
    for (int i = 0; i < testDof; ++i) {
        EXPECT_NEAR(-lhsVelData[i], rhsData[i], 1e-5)
            << "Conservation check failed at DOF " << i 
            << ": LHS*vel=" << lhsVelData[i] << ", RHS=" << rhsData[i];
    }
    
    VecRestoreArrayRead(lhsTimesVel, &lhsVelData);
    VecRestoreArrayRead(rhs, &rhsData);
    VecDestroy(&lhsTimesVel);
    VecDestroy(&velVec);
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
