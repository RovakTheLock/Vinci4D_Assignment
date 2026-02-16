#include <petsc.h>
#include "YamlParser.h"
#include "MeshObject.h"
#include "FieldsHolder.h"
#include "LinearSystem.h"
#include "AssembleAlgorithms.h"
#include "Operations.h"
#include "VtkOutput.h"
#include <iostream>
#include <string>

using namespace Vinci4D;

int main(int argc, char** argv) {
    // Initialize PETSc
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    
    std::cout << "=== Vinci4D C++ Implementation ===" << std::endl;
    std::cout << "PETSc-based Finite Volume CFD Solver\n" << std::endl;
    
    try {
        // Parse command line arguments
        std::string inputFile = "input.yaml";
        if (argc > 1) {
            inputFile = argv[1];
        }
        
        std::cout << "Loading configuration from: " << inputFile << std::endl;
        
        // Parse configuration
        InputConfigParser configParser(inputFile);
        
        // Create mesh
        std::cout << "\nGenerating mesh..." << std::endl;
        MeshObject mesh(configParser);
        
        // Create field arrays
        std::cout << "\nCreating field arrays..." << std::endl;
        FieldArray pressureField("pressure", DimType::SCALAR, mesh.getNumCells());
        FieldArray velocityNew("velocity_np1", DimType::VECTOR, mesh.getNumCells());
        FieldArray velocityOld("velocity_n", DimType::VECTOR, mesh.getNumCells());
        FieldArray gradPressure("grad_pressure", DimType::VECTOR, mesh.getNumCells());
        FieldArray massFluxFace("mass_flux_face", DimType::SCALAR, mesh.getNumFaces());
        
        // Initialize fields
        pressureField.initializeConstant(0.0);
        velocityNew.initializeConstant(0.0);
        velocityOld.initializeConstant(0.0);
        gradPressure.initializeConstant(0.0);
        
        std::cout << "  Created " << mesh.getNumCells() << " cell-centered fields" << std::endl;
        std::cout << "  Created " << mesh.getNumFaces() << " face-centered fields" << std::endl;
        
        // Create linear systems
        std::cout << "\nCreating linear systems..." << std::endl;
        int momentumDof = mesh.getNumCells() * MAX_DIM;
        int pressureDof = mesh.getNumCells();
        
        LinearSystem momentumSystem(momentumDof, "Momentum", true);
        LinearSystem pressureSystem(pressureDof, "Pressure", true);
        
        // Compute time step
        CFLTimeStepCompute cflCompute(&mesh, configParser.getCFL());
        double dt = cflCompute.computeTimeStep();
        std::cout << "  Computed time step: " << dt << " seconds" << std::endl;
        
        // Create assembly algorithms
        std::cout << "\nSetting up assembly algorithms..." << std::endl;
        double diffusionCoeff = 1.0 / configParser.getRe();
        
        AssembleCellVectorTimeTerm timeTermAssembler(
            "TimeTerm", &velocityNew, &velocityOld, &momentumSystem, &mesh, dt);
        
        AssembleInteriorVectorDiffusionToLinSystem diffusionAssembler(
            "Diffusion", &velocityNew, &momentumSystem, &mesh, diffusionCoeff);
        
        // Example: Assemble and solve a simple diffusion problem
        std::cout << "\nAssembling and solving a test system..." << std::endl;
        
        // Initialize some non-zero velocity to make the problem non-trivial
        velocityNew.initializeConstant(1.0);
        velocityOld.initializeConstant(0.5);
        
        // Zero the system
        momentumSystem.zero();
        
        // Assemble time term (this will create a non-zero RHS)
        timeTermAssembler.assemble();
        
        // Assemble diffusion term
        diffusionAssembler.assemble();
        
        // Finalize assembly
        momentumSystem.assembleMatrix();
        
        // Create solution vector and initialize to zero
        Vec solution;
        VecCreate(PETSC_COMM_WORLD, &solution);
        VecSetSizes(solution, PETSC_DECIDE, momentumDof);
        VecSetFromOptions(solution);
        VecSet(solution, 0.0);
        VecAssemblyBegin(solution);
        VecAssemblyEnd(solution);
        
        // Solve using GMRES
        std::cout << "\nSolving linear system with GMRES..." << std::endl;
        momentumSystem.solve("gmres", solution, 1e-8, 1e-10, 1000);

        // Write VTK output if requested by configuration
        VtkOutput vtkWriter(configParser.getOutputDirectory(),
                            configParser.getOutputFrequency());
        if (vtkWriter.shouldWrite(0)) {
            std::vector<FieldArray*> fields = {&pressureField, &velocityNew};
            vtkWriter.write(0, 0.0, mesh, fields);
            std::cout << "  Wrote VTK output to " << configParser.getOutputDirectory() << std::endl;
        }
        
        std::cout << "\nSimulation setup completed successfully!" << std::endl;
        std::cout << "\nMesh statistics:" << std::endl;
        std::cout << "  Total cells: " << mesh.getNumCells() << std::endl;
        std::cout << "  Interior cells: " << mesh.getInteriorCells().size() << std::endl;
        std::cout << "  Boundary cells: " << mesh.getBoundaryCells().size() << std::endl;
        std::cout << "  Internal faces: " << mesh.getInternalFaces().size() << std::endl;
        std::cout << "  Boundary faces: " << mesh.getBoundaryFaces().size() << std::endl;
        
        // Clean up
        VecDestroy(&solution);
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        PetscFinalize();
        return 1;
    }
    
    // Finalize PETSc
    PetscFinalize();
    
    std::cout << "\n=== Program completed successfully ===" << std::endl;
    return 0;
}
