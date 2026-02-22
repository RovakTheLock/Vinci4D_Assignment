#include <petsc.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include "../include/YamlParser.h"
#include "../include/MeshObject.h"
#include "../include/FieldsHolder.h"
#include "../include/LinearSystem.h"
#include "../include/AssembleAlgorithms.h"
#include "../include/Operations.h"
#include "../include/VtkOutput.h"

using namespace Vinci4D;

class PseudoSimulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load mesh configuration from file
        parser = new InputConfigParser("../config/input.yaml");
        mesh = new MeshObject(*parser);
        
        // Get simulation parameters
        double Re = 10000.0;  // Reynolds number
        diffusionCoeff = 1.0 / Re;
        
        // Field initialization (local + ghost in parallel, global for system DOFs)
        mpiSize = mesh->getMpiSize();
        if (mpiSize > 1) {
            numLocalCells = mesh->getNumLocalCells();
            numGhostCells = mesh->getNumGhostCells();
            globalCells = mesh->getGlobalNumCells();
        } else {
            numLocalCells = mesh->getNumCells();
            numGhostCells = 0;
            globalCells = numLocalCells;
        }
        momentumDof = globalCells * MAX_DIM;
        pressureDof = globalCells;
        
        velocityNp1 = new FieldArray("velocity_np1", DimType::VECTOR, numLocalCells, numGhostCells);
        velocityN = new FieldArray("velocity_n", DimType::VECTOR, numLocalCells, numGhostCells);
        pressureField = new FieldArray("pressure", DimType::SCALAR, numLocalCells, numGhostCells);
        gradPressureField = new FieldArray("grad_pressure", DimType::VECTOR, numLocalCells, numGhostCells);
        dPressure = new FieldArray("dPressure", DimType::SCALAR, numLocalCells, numGhostCells);
        grad_dPressure = new FieldArray("grad_dPressure", DimType::VECTOR, numLocalCells, numGhostCells);
        
        velocityNp1->initializeConstant(0.0);
        velocityN->initializeConstant(0.0);
        pressureField->initializeConstant(0.0);
        gradPressureField->initializeConstant(0.0);
        dPressure->initializeConstant(0.0);
        grad_dPressure->initializeConstant(0.0);
        
        // Create linear systems
        int localMomentumDof = numLocalCells * MAX_DIM;
        int localPressureDof = numLocalCells;
        momentumSystem = new LinearSystem(momentumDof, "momentum", true, localMomentumDof);
        pressureSystem = new LinearSystem(pressureDof, "pressure", true, localPressureDof);
        
        // Compute time step
        CFLTimeStepCompute cflCompute(mesh, 0.5);
        dt = cflCompute.computeTimeStep();
    }
    
    void TearDown() override {
        delete velocityNp1;
        delete velocityN;
        delete pressureField;
        delete gradPressureField;
        delete dPressure;
        delete grad_dPressure;
        delete momentumSystem;
        delete pressureSystem;
        delete mesh;
        delete parser;
    }
    
    InputConfigParser* parser;
    MeshObject* mesh;
    int mpiSize;
    int numLocalCells;
    int numGhostCells;
    int globalCells;
    int momentumDof;
    int pressureDof;
    FieldArray* velocityNp1;
    FieldArray* velocityN;
    FieldArray* pressureField;
    FieldArray* gradPressureField;
    FieldArray* dPressure;
    FieldArray* grad_dPressure;
    LinearSystem* momentumSystem;
    LinearSystem* pressureSystem;
    double dt;
    double diffusionCoeff;
};

/**
 * @test Test boundary diffusion assemblies
 * 
 * Verifies that boundary conditions can be applied for lid-driven cavity
 */
TEST_F(PseudoSimulationTest, BoundaryDiffusionAssemblies) {
    std::array<double, 2> topBoundaryValue = {1.0, 0.0};
    std::array<double, 2> otherBoundaryValue = {0.0, 0.0};
    
    EXPECT_NO_THROW({
        AssembleDirichletBoundaryVectorDiffusionToLinSystem top_diffusion(
            "TopBoundary", velocityNp1, momentumSystem, mesh,
            Boundary::TOP, topBoundaryValue, diffusionCoeff);
        
        AssembleDirichletBoundaryVectorDiffusionToLinSystem left_diffusion(
            "LeftBoundary", velocityNp1, momentumSystem, mesh,
            Boundary::LEFT, otherBoundaryValue, diffusionCoeff);
        
        AssembleDirichletBoundaryVectorDiffusionToLinSystem right_diffusion(
            "RightBoundary", velocityNp1, momentumSystem, mesh,
            Boundary::RIGHT, otherBoundaryValue, diffusionCoeff);
        
        AssembleDirichletBoundaryVectorDiffusionToLinSystem bottom_diffusion(
            "BottomBoundary", velocityNp1, momentumSystem, mesh,
            Boundary::BOTTOM, otherBoundaryValue, diffusionCoeff);
    });
}

/**
 * @test Test advection and pressure Poisson
 */
TEST_F(PseudoSimulationTest, AdvectionAndPressurePoissonAssemblies) {
    EXPECT_NO_THROW({
        AssembleInteriorVectorAdvectionToLinSystem advection(
            "Advection", velocityNp1, momentumSystem, mesh);
        
        AssembleInteriorPressurePoissonSystem poisson(
            "PressurePoisson", pressureField, pressureSystem, mesh, dt);
    });
}

/**
 * @test Test gradient computation
 */
TEST_F(PseudoSimulationTest, GradientComputation) {
    pressureField->initializeConstant(1.0);
    
    ComputeCellGradient pressureGradient(mesh, pressureField, gradPressureField);
    EXPECT_NO_THROW(pressureGradient.computeScalarGradient());
    
    auto& gradData = gradPressureField->getData();
    double sumGrad = 0.0;
    const auto& cellsToCheck = (mpiSize > 1) ? mesh->getLocalInteriorCells() : mesh->getInteriorCells();
    for (const auto* cell : cellsToCheck) {
        int localId = (mpiSize > 1) ? cell->getLocalId() : cell->getFlatId();
        sumGrad += std::abs(gradData[localId * MAX_DIM + 0]);
        sumGrad += std::abs(gradData[localId * MAX_DIM + 1]);
    }
    EXPECT_LT(sumGrad, 1e-8) << "Gradient of constant field should be near zero";
}

/**
 * @test Single coupled iteration with full assembly
 */
TEST_F(PseudoSimulationTest, SingleCoupledIteration) {
    std::array<double, 2> topBoundary = {1.0, 0.0};
    std::array<double, 2> noBoundary = {0.0, 0.0};
    
    AssembleCellVectorTimeTerm timeTermAlg("TimeTerm", velocityNp1, velocityN, 
                                           momentumSystem, mesh, dt);
    AssembleInteriorVectorDiffusionToLinSystem diffusionAlg("Diffusion", velocityNp1, 
                                                             momentumSystem, mesh, diffusionCoeff);
    
    AssembleDirichletBoundaryVectorDiffusionToLinSystem boundaryTopAlg(
        "BoundaryTop", velocityNp1, momentumSystem, mesh, Boundary::TOP, topBoundary, diffusionCoeff);
    AssembleDirichletBoundaryVectorDiffusionToLinSystem boundaryLeftAlg(
        "BoundaryLeft", velocityNp1, momentumSystem, mesh, Boundary::LEFT, noBoundary, diffusionCoeff);
    AssembleDirichletBoundaryVectorDiffusionToLinSystem boundaryRightAlg(
        "BoundaryRight", velocityNp1, momentumSystem, mesh, Boundary::RIGHT, noBoundary, diffusionCoeff);
    AssembleDirichletBoundaryVectorDiffusionToLinSystem boundaryBottomAlg(
        "BoundaryBottom", velocityNp1, momentumSystem, mesh, Boundary::BOTTOM, noBoundary, diffusionCoeff);
    
    AssembleInteriorVectorAdvectionToLinSystem advectionAlg("Advection", velocityNp1, 
                                                             momentumSystem, mesh);
    AssembleCellVectorPressureGradientToLinSystem pressureGradAlg("PressureGrad", 
                                                                   gradPressureField, 
                                                                   momentumSystem, mesh);
    
    std::vector<AssembleSystemBase*> momentumAlgorithms = {
        &timeTermAlg, &diffusionAlg, &boundaryTopAlg, &boundaryLeftAlg,
        &boundaryRightAlg, &boundaryBottomAlg, &advectionAlg, &pressureGradAlg
    };
    
    momentumSystem->zero();
    for (auto* alg : momentumAlgorithms) {
        EXPECT_NO_THROW(alg->assemble());
    }
    momentumSystem->assembleMatrix();
    EXPECT_GE(momentumSystem->getRhsNorm(), 0.0);
    
    AssembleInteriorPressurePoissonSystem pressurePoissonAlg("PressurePoisson", 
                                                              pressureField, 
                                                              pressureSystem, mesh, dt);
    pressureSystem->zero();
    EXPECT_NO_THROW(pressurePoissonAlg.assemble());
    pressureSystem->assembleMatrix();
    EXPECT_GE(pressureSystem->getRhsNorm(), 0.0);
    
    ComputeInteriorMassFlux massFluxOp(mesh, velocityNp1, pressureField, 
                                        gradPressureField, dt);
    EXPECT_NO_THROW(massFluxOp.computeMassFlux());
    
    ComputeCellGradient pressureGradOp(mesh, pressureField, gradPressureField);
    EXPECT_NO_THROW(pressureGradOp.computeScalarGradient());
}

/**
 * @test Multi-step coupled momentum-pressure solver
 */
TEST_F(PseudoSimulationTest, CoupledSolverLoop) {
    double terminationTime = parser->getTerminationTime();
    int numTimeSteps = static_cast<int>(std::ceil(terminationTime / dt));
    if (numTimeSteps < 1) {
        numTimeSteps = 1;
    }
    int numNonlinearIters = parser->getNumNonlinearIterations();
    double continuityTol = parser->getContinuityTolerance();
    double momentumTol = parser->getMomentumTolerance();
    double alphaV = 1.0;
    double alphaP = 1.0;
    
    std::array<double, 2> topBoundary = {1.0, 0.0};
    std::array<double, 2> noBoundary = {0.0, 0.0};
    
    AssembleCellVectorTimeTerm timeTermAlg("TimeTerm", velocityNp1, velocityN, 
                                           momentumSystem, mesh, dt);
    AssembleInteriorVectorDiffusionToLinSystem diffusionAlg("Diffusion", velocityNp1, 
                                                             momentumSystem, mesh, diffusionCoeff);
    AssembleDirichletBoundaryVectorDiffusionToLinSystem boundaryTopAlg(
        "BoundaryTop", velocityNp1, momentumSystem, mesh, Boundary::TOP, topBoundary, diffusionCoeff);
    AssembleDirichletBoundaryVectorDiffusionToLinSystem boundaryLeftAlg(
        "BoundaryLeft", velocityNp1, momentumSystem, mesh, Boundary::LEFT, noBoundary, diffusionCoeff);
    AssembleDirichletBoundaryVectorDiffusionToLinSystem boundaryRightAlg(
        "BoundaryRight", velocityNp1, momentumSystem, mesh, Boundary::RIGHT, noBoundary, diffusionCoeff);
    AssembleDirichletBoundaryVectorDiffusionToLinSystem boundaryBottomAlg(
        "BoundaryBottom", velocityNp1, momentumSystem, mesh, Boundary::BOTTOM, noBoundary, diffusionCoeff);
    AssembleInteriorVectorAdvectionToLinSystem advectionAlg("Advection", velocityNp1, 
                                                             momentumSystem, mesh);
    AssembleCellVectorPressureGradientToLinSystem pressureGradAlg("PressureGrad", 
                                                                   gradPressureField, 
                                                                   momentumSystem, mesh);
    
    std::vector<AssembleSystemBase*> momentumAlgorithms = {
        &timeTermAlg, &diffusionAlg, &boundaryTopAlg, &boundaryLeftAlg,
        &boundaryRightAlg, &boundaryBottomAlg, &advectionAlg, &pressureGradAlg
    };
    
    AssembleInteriorPressurePoissonSystem pressurePoissonAlg("PressurePoisson", 
                                                              pressureField, 
                                                              pressureSystem, mesh, dt);
    
    ComputeInteriorMassFlux massFluxOp(mesh, velocityNp1, pressureField, 
                                        gradPressureField, dt);
    ComputeCellGradient pressureGradOp(mesh, pressureField, gradPressureField);
    ComputeCellGradient dPressureGradOp(mesh, dPressure, grad_dPressure);
    
    std::vector<LinearSystem*> systems = {momentumSystem, pressureSystem};
    LogObject logger(systems);
    PerformanceTimer timer;
    VtkOutput vtkWriter(parser->getOutputDirectory(), parser->getOutputFrequency());
    bool isRoot = (mesh->getMpiRank() == 0);
    if (isRoot) {
        std::cout << "Time step used is: " << dt << " seconds" << std::endl;
        std::cout << "Number of time steps: " << numTimeSteps << std::endl;
        std::cout << "Nonlinear iterations per step: " << numNonlinearIters << std::endl;
        std::cout << "Momentum tolerance: " << momentumTol << std::endl;
        std::cout << "Continuity tolerance: " << continuityTol << std::endl;
    }
    
    for (int t = 0; t < numTimeSteps; ++t) {
        if (isRoot) {
            std::cout << "\nTime step " << t + 1 << "/" << numTimeSteps << " Current time = " << t * dt << std::endl;
            std::cout << "===============================================================================================" << std::endl;
        }
        for (int iter = 0; iter < numNonlinearIters; ++iter) {
            // Momentum assembly
            timer.startTimer("MomentumAssembly");
            pressureField->exchangeGhostCells(*mesh);
            velocityNp1->exchangeGhostCells(*mesh);
            velocityN->exchangeGhostCells(*mesh);
            momentumSystem->zero();
            for (auto* alg : momentumAlgorithms) {
                alg->assemble();
            }
            timer.endTimer();
            
            momentumSystem->assembleMatrix();
            double momentumRhsNorm = momentumSystem->getRhsNorm();
            
            // Solve momentum system for velocity corrections
            Vec momentumSolution;
            VecCreate(PETSC_COMM_WORLD, &momentumSolution);
            VecSetSizes(momentumSolution, PETSC_DECIDE, momentumSystem->getNumDof());
            VecSetFromOptions(momentumSolution);
            timer.startTimer("MomentumSolve");
            momentumSystem->solve("gmres", momentumSolution, 1e-6, 1e-10, 1000);
            timer.endTimer();
            
            // Increment velocity with correction (not replace!)
            const PetscScalar* momentumData = nullptr;
            VecGetArrayRead(momentumSolution, &momentumData);
            PetscInt momStart, momEnd;
            VecGetOwnershipRange(momentumSolution, &momStart, &momEnd);
            auto& velData = velocityNp1->getData();
            const auto& velCells = (mpiSize > 1) ? mesh->getLocalCells() : mesh->getCells();
            for (const auto& cell : velCells) {
                int localId = (mpiSize > 1) ? cell.getLocalId() : cell.getFlatId();
                int globalId = (mpiSize > 1) ? mesh->localToGlobal(localId) : cell.getFlatId();
                for (int comp = 0; comp < MAX_DIM; ++comp) {
                    int globalDof = globalId * MAX_DIM + comp;
                    if (globalDof >= momStart && globalDof < momEnd) {
                        int localDofIdx = globalDof - momStart;
                        velData[localId * MAX_DIM + comp] += alphaV * momentumData[localDofIdx];
                    }
                }
            }
            VecRestoreArrayRead(momentumSolution, &momentumData);
            VecDestroy(&momentumSolution);

            velocityNp1->exchangeGhostCells(*mesh);
            pressureField->exchangeGhostCells(*mesh);
            massFluxOp.computeMassFlux();
            
            // Pressure assembly (full matrix on first iteration, RHS-only on subsequent)
            timer.startTimer("PressureAssembly");
            if (iter == 0) {
                // First iteration: full matrix assembly
                pressureSystem->zero();  // Recreate matrix structure with diagonal entries
                pressurePoissonAlg.assemble();
                pressureSystem->assembleMatrix();
                // Pin first row to remove nullspace (pressure is defined up to a constant)
                pressureSystem->pinFirstRow();
            } else {
                // Subsequent iterations: only clear and reassemble RHS
                pressureSystem->zeroRhs();
                pressurePoissonAlg.assembleRhs();
                pressureSystem->assembleMatrix();
            }
            timer.endTimer();
            
            
            // Solve pressure system for pressure corrections
            Vec pressureSolution;
            VecCreate(PETSC_COMM_WORLD, &pressureSolution);
            VecSetSizes(pressureSolution, PETSC_DECIDE, pressureSystem->getNumDof());
            VecSetFromOptions(pressureSolution);
            timer.startTimer("ContinuitySolve");
            pressureSystem->solvePressure(pressureSolution, 1e-7);  // Loose tolerance for intermediate iterations
            double pressureRhsNorm = pressureSystem->getRhsNorm();
            timer.endTimer();
            
            // Store solution as pressure correction in dPressure field
            const PetscScalar* pressureData = nullptr;
            VecGetArrayRead(pressureSolution, &pressureData);
            PetscInt presStart, presEnd;
            VecGetOwnershipRange(pressureSolution, &presStart, &presEnd);
            auto& dPresData = dPressure->getData();
            const auto& presCells = (mpiSize > 1) ? mesh->getLocalCells() : mesh->getCells();
            for (const auto& cell : presCells) {
                int localId = (mpiSize > 1) ? cell.getLocalId() : cell.getFlatId();
                int globalId = (mpiSize > 1) ? mesh->localToGlobal(localId) : cell.getFlatId();
                if (globalId >= presStart && globalId < presEnd) {
                    int localDofIdx = globalId - presStart;
                    dPresData[localId] = pressureData[localDofIdx];
                }
            }
            VecRestoreArrayRead(pressureSolution, &pressureData);
            VecDestroy(&pressureSolution);
            
            // Increment accumulated pressure with correction
            auto& pData = pressureField->getData();
            const auto& pCells = (mpiSize > 1) ? mesh->getLocalCells() : mesh->getCells();
            for (const auto& cell : pCells) {
                int localId = (mpiSize > 1) ? cell.getLocalId() : cell.getFlatId();
                pData[localId] += alphaP * dPresData[localId];
            }
            
            // Compute gradients of pressure and pressure correction
            dPressureGradOp.computeScalarGradient();
            pressureGradOp.computeScalarGradient();
            gradPressureField->exchangeGhostCells(*mesh);
            grad_dPressure->exchangeGhostCells(*mesh);
            // Correct velocity with pressure correction gradient
            auto& velData2 = velocityNp1->getData();
            auto& gradDPresData = grad_dPressure->getData();
            const auto& velCells2 = (mpiSize > 1) ? mesh->getLocalCells() : mesh->getCells();
            for (const auto& cell : velCells2) {
                int localId = (mpiSize > 1) ? cell.getLocalId() : cell.getFlatId();
                for (int comp = 0; comp < MAX_DIM; ++comp) {
                    int localIndex = localId * MAX_DIM + comp;
                    velData2[localIndex] -= alphaV * dt * gradDPresData[localIndex];
                }
            }
            velocityNp1->exchangeGhostCells(*mesh);
            massFluxOp.computeMassFlux();
            
            // Check for residual growth (avoiding NaN propagation)
            if (momentumRhsNorm > 1e3 || pressureRhsNorm > 1e3) {
                if (isRoot) {
                    std::cerr << "Residual growth detected: momentum=" << momentumRhsNorm 
                              << ", pressure=" << pressureRhsNorm << std::endl;
                }
                break;  // Exit nonlinear iteration loop
            }
            logger.reportLog(iter);
            if (momentumRhsNorm < momentumTol && pressureRhsNorm < continuityTol) {
                break;
            }
        }
        if (isRoot) {
            std::cout << "===============================================================================================" << std::endl;
        }
        if (vtkWriter.shouldWrite(t)) {
            std::vector<FieldArray*> fields = {pressureField, velocityNp1};
            vtkWriter.write(t, t * dt, *mesh, fields);
        }

        velocityNp1->copyTo(*velocityN);
        
    }
    timer.report_timing_statistics();
    
    EXPECT_TRUE(true) << "Coupled solver loop completed successfully";
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    
    // Configure PETSc solver options for better performance
    // Use AMG (GAMG) preconditioner for elliptic problems
    PetscOptionsSetValue(nullptr, "-pc_type", "gamg");
    
    // Alternative: Use hypre BoomerAMG (if available)
    // PetscOptionsSetValue(nullptr, "-pc_type", "hypre");
    // PetscOptionsSetValue(nullptr, "-pc_hypre_type", "boomeramg");
    
    // Improve ILU fill level if not using AMG
    // PetscOptionsSetValue(nullptr, "-pc_factor_levels", "3");
    
    ::testing::InitGoogleTest(&argc, argv);
    int mpiRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    if (mpiRank != 0) {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
        delete listeners.Release(listeners.default_xml_generator());
    }
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
