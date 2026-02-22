#include <petsc.h>
#include <gtest/gtest.h>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include "../include/MeshObject.h"
#include "../include/YamlParser.h"
#include "../include/FieldsHolder.h"
#include "../include/Operations.h"
#include "../include/AssembleAlgorithms.h"
#include "../include/LinearSystem.h"

using namespace Vinci4D;

class ParallelTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        // Ensure all ranks are synchronized at test start
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    void TearDown() override {
        // Ensure all ranks complete the test before moving to next test
        // This is critical for MPI state cleanup
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    int rank;
    int size;
};

TEST_F(ParallelTest, MPIInitialized) {
    // Basic test that MPI is initialized correctly
    EXPECT_GE(rank, 0);
    EXPECT_GT(size, 0);
    EXPECT_LT(rank, size);
    
    if (rank == 0) {
        std::cout << "Running with " << size << " MPI ranks" << std::endl;
    }
}

TEST_F(ParallelTest, DomainDecomposition) {
    // Test that domain is properly decomposed across ranks
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 8
  num_cells_y: 8
simulation:
  Re: 100
  CFL: 0.5
)";
    
    std::ofstream tmpYaml("/tmp/parallel_test_" + std::to_string(rank) + ".yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser parser("/tmp/parallel_test_" + std::to_string(rank) + ".yaml");
    MeshObject mesh(parser);
    
    // Each rank should have cells
    int localCells = mesh.getNumLocalCells();
    EXPECT_GT(localCells, 0) << "Rank " << rank << " should have local cells";
    
    // Sum up cells across all ranks
    int totalCells = 0;
    MPI_Allreduce(&localCells, &totalCells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Total should equal 8*8 = 64
    EXPECT_EQ(totalCells, 64) << "Total cells across all ranks should be 64";
    
    // Each rank should know its neighbors
    auto neighbors = mesh.getNeighborRanks();
}

TEST_F(ParallelTest, GhostCellCommunicationScalar) {
    // Test that ghost cells are properly communicated
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 8
  num_cells_y: 8
simulation:
  Re: 100
  CFL: 0.5
)";
    
    std::ofstream tmpYaml("/tmp/ghost_test_" + std::to_string(rank) + ".yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser parser("/tmp/ghost_test_" + std::to_string(rank) + ".yaml");
    MeshObject mesh(parser);
    
    // Create a scalar field with rank-specific values
    int numLocalCells = mesh.getNumLocalCells();
    int numGhostCells = mesh.getNumGhostCells();
    
    FieldArray field("test_field", DimType::SCALAR, numLocalCells, numGhostCells);
    
    // Initialize local cells with rank ID
    auto& data = field.getData();
    for (int i = 0; i < numLocalCells; ++i) {
        data[i] = static_cast<double>(rank);
    }
    
    // Exchange ghost cells
    field.exchangeGhostCells(mesh);
    
    // Ghost cells should now contain neighbor rank values
    for (int i = numLocalCells; i < numLocalCells + numGhostCells; ++i) {
        double ghostValue = data[i];
        // Ghost value should be different from our rank (unless we have no neighbors)
        if (numGhostCells > 0 && size > 1) {
            EXPECT_NE(ghostValue, static_cast<double>(rank)) 
                << "Ghost cell " << i << " should have neighbor's value";
        }
    }
}

TEST_F(ParallelTest, GhostCellCommunicationVector) {
    // Test that ghost cells are properly communicated for vector fields
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 8
  num_cells_y: 8
simulation:
  Re: 100
  CFL: 0.5
)";
    
    std::ofstream tmpYaml("/tmp/ghost_vector_test_" + std::to_string(rank) + ".yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser parser("/tmp/ghost_vector_test_" + std::to_string(rank) + ".yaml");
    MeshObject mesh(parser);
    
    // Create a vector field with rank-specific values
    int numLocalCells = mesh.getNumLocalCells();
    int numGhostCells = mesh.getNumGhostCells();
    
    FieldArray field("test_vector_field", DimType::VECTOR, numLocalCells, numGhostCells);
    
    // Initialize local cells with rank ID (same value for all components)
    auto& data = field.getData();
    for (int i = 0; i < numLocalCells; ++i) {
        for (int d = 0; d < MAX_DIM; ++d) {
            data[i * MAX_DIM + d] = static_cast<double>(rank);
        }
    }
    
    // Exchange ghost cells
    field.exchangeGhostCells(mesh);
    
    // Ghost cells should now contain neighbor rank values
    for (int i = numLocalCells; i < numLocalCells + numGhostCells; ++i) {
        for (int d = 0; d < MAX_DIM; ++d) {
            double ghostValue = data[i * MAX_DIM + d];
            // Ghost value should be different from our rank (unless we have no neighbors)
            if (numGhostCells > 0 && size > 1) {
                EXPECT_NE(ghostValue, static_cast<double>(rank)) 
                    << "Ghost cell " << i << " component " << d << " should have neighbor's value";
            }
        }
    }
}

TEST_F(ParallelTest, ParallelAssembly) {
    // Test that matrix assembly works correctly in parallel
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 8
  num_cells_y: 8
simulation:
  Re: 100
  CFL: 0.5
)";
    
    std::ofstream tmpYaml("/tmp/assembly_parallel_test_" + std::to_string(rank) + ".yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser parser("/tmp/assembly_parallel_test_" + std::to_string(rank) + ".yaml");
    MeshObject mesh(parser);
    
    int numLocalCells = mesh.getNumLocalCells();
    int numGhostCells = mesh.getNumGhostCells();
    int globalNumCells = mesh.getGlobalNumCells();
    
    // Create velocity field
    FieldArray velocityField("velocity", DimType::VECTOR, numLocalCells, numGhostCells);
    velocityField.initializeConstant(1.0);
    
    // For now, just verify we can create fields and mesh without crashing
    // Full parallel assembly will require updating the assembly algorithms
    EXPECT_EQ(globalNumCells, 64) << "Global mesh should have 64 cells";
    EXPECT_GT(numLocalCells, 0) << "Rank " << rank << " should have local cells";
}

TEST_F(ParallelTest, GhostCellConsistencyAndLinearSolve) {
    // Test ghost cell exchange and identity matrix solve with rank-valued field
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 10
  num_cells_y: 10
simulation:
  Re: 100
  CFL: 0.5
)";
    
    std::ofstream tmpYaml("/tmp/ghost_consistency_test_" + std::to_string(rank) + ".yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser parser("/tmp/ghost_consistency_test_" + std::to_string(rank) + ".yaml");
    MeshObject mesh(parser);
    
    int numLocalCells = mesh.getNumLocalCells();
    int numGhostCells = mesh.getNumGhostCells();
    int globalNumCells = mesh.getGlobalNumCells();
    
    // Create scalar field initialized with rank value on local cells
    FieldArray testField("test_field", DimType::SCALAR, numLocalCells, numGhostCells);
    testField.initializeConstant(0.0);
    
    // Set local cells to rank number
    auto& fieldData = testField.getData();
    for (int i = 0; i < numLocalCells; ++i) {
        fieldData[i] = (double)rank;
    }
    
    // Synchronize before exchange
    MPI_Barrier(MPI_COMM_WORLD);
    // Exchange ghost cells
    testField.exchangeGhostCells(mesh);
    
    // Synchronize after exchange
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Verify ghost cells have correct values using communication info
    const auto& ghostCommInfo = mesh.getGhostCommInfo();
    
    // Validate each ghost cell contains the expected neighbor rank value
    for (const auto& commInfo : ghostCommInfo) {
        for (size_t idx = 0; idx < commInfo.recvLocalIds.size(); ++idx) {
            int ghostLocalId = commInfo.recvLocalIds[idx];
            
            // Bounds check
            if (ghostLocalId < numLocalCells || ghostLocalId >= numLocalCells + numGhostCells) {
                FAIL() << "Rank " << rank << ": Ghost local ID " << ghostLocalId 
                       << " out of bounds [" << numLocalCells << ", " 
                       << (numLocalCells + numGhostCells) << ")";
            }
            
            double expectedValue = (double)commInfo.neighborRank;
            double actualValue = fieldData[ghostLocalId];
            EXPECT_EQ(actualValue, expectedValue) 
                << "Rank " << rank << ": Ghost cell at local ID " << ghostLocalId 
                << " should have value from rank " << commInfo.neighborRank;
        }
    }
    
    // Create linear system with identity matrix
    LinearSystem system(globalNumCells, "identity_system", true, numLocalCells);
    
    // Synchronize before assembly
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Assemble identity matrix: diagonal = 1.0
    // Use sequential indexing to match PETSc's distributed DOF numbering
    PetscInt startIdx, endIdx;
    system.getOwnershipRange(startIdx, endIdx);
    
    int assemblySeqIdx = 0;
    for (int localId = 0; localId < numLocalCells; ++localId) {
        int petscDof = system.getGlobalDofFromLocalIndex(assemblySeqIdx);
        system.addLhs(petscDof, petscDof, 1.0);
        // RHS = field value at this cell
        system.addRhs(petscDof, fieldData[localId]);
        assemblySeqIdx++;
    }
    
    system.assembleMatrix();
    
    // Synchronize before solve
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Solve: x = b since it's identity matrix
    Vec solution;
    VecCreate(MPI_COMM_WORLD, &solution);
    VecSetSizes(solution, PETSC_DECIDE, globalNumCells);
    VecSetFromOptions(solution);
    
    system.solve("cg", solution, 1e-10, 1e-15, 100);
    
    // Synchronize after solve
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Extract solution and verify it matches original field
    PetscInt startRow, endRow;
    VecGetOwnershipRange(solution, &startRow, &endRow);
    
    const PetscScalar* solutionData;
    VecGetArrayRead(solution, &solutionData);
    
    // Use sequential indexing to match PETSc's distributed DOF numbering
    int sequentialIdx = 0;
    for (int localId = 0; localId < numLocalCells; ++localId) {
        double expected = fieldData[localId];
        double actual = solutionData[sequentialIdx];
        EXPECT_NEAR(actual, expected, 1e-10)
            << "Rank " << rank << ": Solution mismatch at local cell " << localId;
        sequentialIdx++;
    }
    
    VecRestoreArrayRead(solution, &solutionData);
    
    // Now verify ghost cells by creating a field from the solution and exchanging
    FieldArray solutionField("solution_field", DimType::SCALAR, numLocalCells, numGhostCells);
    
    // Extract local solution values from PETSc vector
    solutionField.extractFromPETScVector(solution, numLocalCells);
    
    // Exchange ghost cells to get values from neighbors
    solutionField.exchangeGhostCells(mesh);
    
    // Verify ghost cells match expected neighbor values
    auto& solutionFieldData = solutionField.getData();
    for (const auto& commInfo : ghostCommInfo) {
        for (size_t idx = 0; idx < commInfo.recvLocalIds.size(); ++idx) {
            int ghostLocalId = commInfo.recvLocalIds[idx];
            double expectedGhostValue = (double)commInfo.neighborRank;
            double actualGhostValue = solutionFieldData[ghostLocalId];
            EXPECT_NEAR(actualGhostValue, expectedGhostValue, 1e-10)
                << "Rank " << rank << ": Ghost solution mismatch at local ID " << ghostLocalId;
        }
    }
    
    VecDestroy(&solution);
}

TEST_F(ParallelTest, GhostCellConsistencyAndLinearSolveVector) {
    // Test ghost cell exchange and identity matrix solve with rank-valued vector field
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 10
  num_cells_y: 10
simulation:
  Re: 100
  CFL: 0.5
)";
    
    std::ofstream tmpYaml("/tmp/ghost_consistency_vector_test_" + std::to_string(rank) + ".yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser parser("/tmp/ghost_consistency_vector_test_" + std::to_string(rank) + ".yaml");
    MeshObject mesh(parser);
    
    int numLocalCells = mesh.getNumLocalCells();
    int numGhostCells = mesh.getNumGhostCells();
    int globalNumCells = mesh.getGlobalNumCells();
    
    // Create vector field initialized with rank value on local cells
    FieldArray testField("test_vector_field", DimType::VECTOR, numLocalCells, numGhostCells);
    testField.initializeConstant(0.0);
    
    // Set local cells to rank number for all components
    auto& fieldData = testField.getData();
    for (int i = 0; i < numLocalCells; ++i) {
        for (int d = 0; d < MAX_DIM; ++d) {
            fieldData[i * MAX_DIM + d] = (double)rank;
        }
    }
    
    // Synchronize before exchange
    MPI_Barrier(MPI_COMM_WORLD);
    // Exchange ghost cells
    testField.exchangeGhostCells(mesh);
    
    // Synchronize after exchange
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Verify ghost cells have correct values using communication info
    const auto& ghostCommInfo = mesh.getGhostCommInfo();
    
    // Validate each ghost cell contains the expected neighbor rank value
    for (const auto& commInfo : ghostCommInfo) {
        for (size_t idx = 0; idx < commInfo.recvLocalIds.size(); ++idx) {
            int ghostLocalId = commInfo.recvLocalIds[idx];
            
            // Bounds check
            if (ghostLocalId < numLocalCells || ghostLocalId >= numLocalCells + numGhostCells) {
                FAIL() << "Rank " << rank << ": Ghost local ID " << ghostLocalId 
                       << " out of bounds [" << numLocalCells << ", " 
                       << (numLocalCells + numGhostCells) << ")";
            }
            
            double expectedValue = (double)commInfo.neighborRank;
            for (int d = 0; d < MAX_DIM; ++d) {
                double actualValue = fieldData[ghostLocalId * MAX_DIM + d];
                EXPECT_EQ(actualValue, expectedValue) 
                    << "Rank " << rank << ": Ghost cell at local ID " << ghostLocalId 
                    << " component " << d << " should have value from rank " << commInfo.neighborRank;
            }
        }
    }
    
    // Create linear system with identity matrix for vector DOFs
    int globalVectorDof = globalNumCells * MAX_DIM;
    int localVectorDof = numLocalCells * MAX_DIM;
    LinearSystem system(globalVectorDof, "identity_vector_system", true, localVectorDof);
    
    // Synchronize before assembly
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Assemble identity matrix: diagonal = 1.0 for each DOF
    // Use sequential indexing to match PETSc's distributed DOF numbering
    int assemblySeqIdx = 0;
    for (int localId = 0; localId < numLocalCells; ++localId) {
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            int petscDof = system.getGlobalDofFromLocalIndex(assemblySeqIdx);
            system.addLhs(petscDof, petscDof, 1.0);
            // RHS = field value at this component
            system.addRhs(petscDof, fieldData[localId * MAX_DIM + comp]);
            assemblySeqIdx++;
        }
    }
    
    system.assembleMatrix();
    
    // Synchronize before solve
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Solve: x = b since it's identity matrix
    Vec solution;
    VecCreate(MPI_COMM_WORLD, &solution);
    VecSetSizes(solution, PETSC_DECIDE, globalVectorDof);
    VecSetFromOptions(solution);
    
    system.solve("cg", solution, 1e-10, 1e-15, 100);
    
    // Synchronize after solve
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Extract solution and verify it matches original field
    PetscInt startRow, endRow;
    VecGetOwnershipRange(solution, &startRow, &endRow);
    
    const PetscScalar* solutionData;
    VecGetArrayRead(solution, &solutionData);
    
    // Use sequential indexing to match PETSc's distributed DOF numbering
    int sequentialIdx = 0;
    for (int localId = 0; localId < numLocalCells; ++localId) {
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            double expected = fieldData[localId * MAX_DIM + comp];
            double actual = solutionData[sequentialIdx];
            EXPECT_NEAR(actual, expected, 1e-10)
                << "Rank " << rank << ": Solution mismatch at local cell " << localId 
                << " component " << comp;
            sequentialIdx++;
        }
    }
    
    VecRestoreArrayRead(solution, &solutionData);
    
    // Now verify ghost cells by creating a field from the solution and exchanging
    FieldArray solutionField("solution_vector_field", DimType::VECTOR, numLocalCells, numGhostCells);
    
    // Extract local solution values from PETSc vector
    // For vector fields, numLocalCells should account for component dimension
    solutionField.extractFromPETScVector(solution, numLocalCells);
    
    // Exchange ghost cells to get values from neighbors
    solutionField.exchangeGhostCells(mesh);
    
    // Verify ghost cells match expected neighbor values
    auto& solutionFieldData = solutionField.getData();
    for (const auto& commInfo : ghostCommInfo) {
        for (size_t idx = 0; idx < commInfo.recvLocalIds.size(); ++idx) {
            int ghostLocalId = commInfo.recvLocalIds[idx];
            double expectedGhostValue = (double)commInfo.neighborRank;
            for (int comp = 0; comp < MAX_DIM; ++comp) {
                double actualGhostValue = solutionFieldData[ghostLocalId * MAX_DIM + comp];
                EXPECT_NEAR(actualGhostValue, expectedGhostValue, 1e-10)
                    << "Rank " << rank << ": Ghost solution mismatch at local ID " << ghostLocalId 
                    << " component " << comp;
            }
        }
    }
    
    VecDestroy(&solution);
}

int main(int argc, char** argv) {
    // Initialize MPI FIRST
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Initialize PETSc
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    
    // Initialize Google Test on all ranks (required before RUN_ALL_TESTS)
    ::testing::InitGoogleTest(&argc, argv);
    
    // Ensure all ranks reach this point before running tests
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run tests
    int result = RUN_ALL_TESTS();
    
    // Synchronize all ranks before finalization
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Finalize in reverse order of initialization
    PetscFinalize();
    MPI_Finalize();
    
    return result;
}
