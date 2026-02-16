#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include "../include/MeshObject.h"
#include "../include/FieldsHolder.h"
#include "../include/LinearSystem.h"
#include "../include/YamlParser.h"

using namespace Vinci4D;

/**
 * @test Debug version of GhostCellConsistencyAndLinearSolve with detailed tracing
 * Helps identify deadlocks in parallel execution with 4 ranks
 */
TEST(ParallelDebugTest, GhostCellExchangeDebug) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        GTEST_SKIP() << "Test requires exactly 4 MPI ranks";
    }

    // Create test YAML
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

    std::string yaml_path = "/tmp/ghost_debug_" + std::to_string(rank) + ".yaml";
    std::ofstream tmpYaml(yaml_path);
    tmpYaml << yaml_content;
    tmpYaml.close();

    // Use common path - all ranks write to same file (file operations are atomic enough for this test)
    InputConfigParser parser("/tmp/ghost_debug_0.yaml");
    MeshObject mesh(parser);

    int numLocalCells = mesh.getNumLocalCells();
    int numGhostCells = mesh.getNumGhostCells();
    int globalNumCells = mesh.getGlobalNumCells();

    std::cout << "Rank " << rank << ": local=" << numLocalCells 
              << " ghost=" << numGhostCells << " from (" << mesh.getLocalStartX() 
              << ":" << mesh.getLocalEndX() << ", " << mesh.getLocalStartY() 
              << ":" << mesh.getLocalEndY() << ")" << std::endl;

    // Create field and initialize
    FieldArray testField("test_field", DimType::SCALAR, numLocalCells, numGhostCells);
    testField.initializeConstant(0.0);

    auto& fieldData = testField.getData();
    for (int i = 0; i < numLocalCells; ++i) {
        fieldData[i] = (double)rank;
    }

    std::cout << "Rank " << rank << ": initialized local cells with value " << rank << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Debug: print neighbor info
    std::cout << "Rank " << rank << ": neighbors = ";
    for (const auto& n : mesh.getNeighborRanks()) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Exchange ghost cells WITH TIMING
    std::cout << "Rank " << rank << ": Starting ghost cell exchange..." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    double tStart = MPI_Wtime();

    testField.exchangeGhostCells(mesh);

    double tEnd = MPI_Wtime();
    std::cout << "Rank " << rank << ": Ghost cell exchange completed in " << (tEnd - tStart) << " sec" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify
    std::cout << "Rank " << rank << ": Verifying ghost cells..." << std::endl;
    bool ghostCellsValid = true;
    for (int i = numLocalCells; i < numLocalCells + numGhostCells; ++i) {
        if (fieldData[i] < 0 || fieldData[i] >= (double)size) {
            ghostCellsValid = false;
            std::cerr << "Rank " << rank << ": Invalid ghost cell at " << i << ": " << fieldData[i] << std::endl;
        }
    }

    if (ghostCellsValid && numGhostCells > 0) {
        std::cout << "Rank " << rank << ": Ghost cells valid" << std::endl;
    }

    EXPECT_TRUE(ghostCellsValid || numGhostCells == 0) << "Ghost cells invalid on rank " << rank;

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << rank << ": Test completed" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    PetscFinalize();
    MPI_Finalize();
    return result;
}
