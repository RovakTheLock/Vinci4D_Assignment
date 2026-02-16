#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include "../include/MeshObject.h"
#include "../include/FieldsHolder.h"
#include "../include/LinearSystem.h"
#include "../include/YamlParser.h"

using namespace Vinci4D;

/**
 * @test Demonstrate the ghost cell synchronization bug
 *
 * The bug: After updating fields, ghost cells are not synchronized.
 * This causes incorrect boundary values in the next iteration.
 *
 * Test case: Two-step process
 * 1. Update field locally → ghost cells become stale
 * 2. Without exchange, gradient computation uses stale ghost values
 * 3. This causes wrong residuals in the next iteration
 */
TEST(GhostSyncBugTest, MissingExchangeAfterUpdate) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        GTEST_SKIP() << "Test requires 2 MPI ranks";
    }

    InputConfigParser parser("../config/input.yaml");
    MeshObject mesh(parser);

    int numLocal = mesh.getNumLocalCells();
    int numGhost = mesh.getNumGhostCells();

    // Create a scalar field (like pressure)
    FieldArray field("test_field", DimType::SCALAR, numLocal, numGhost);
    auto& data = field.getData();

    // ITERATION 1: Initialize with rank value
    for (int i = 0; i < numLocal; ++i) {
        data[i] = static_cast<double>(rank) * 10.0;  // Rank 0: 0, Rank 1: 10
    }

    // Exchange ghost cells (normal operation)
    field.exchangeGhostCells(mesh);

    // Store initial ghost values for verification
    std::vector<double> initialGhostValues(numGhost);
    for (int i = 0; i < numGhost; ++i) {
        initialGhostValues[i] = data[numLocal + i];
    }

    if (rank == 0) {
        std::cout << "Initial ghost values (Rank 0):\n";
        for (int i = 0; i < std::min(3, numGhost); ++i) {
            std::cout << "  ghost[" << i << "] = " << initialGhostValues[i] << " (from Rank 1: should be 10)\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ITERATION 2: Update local values (simulating coupled solution step)
    // This is what happens when momentum or pressure is updated
    for (int i = 0; i < numLocal; ++i) {
        data[i] += 5.0;  // Add 5 to each local cell
    }

    // BUG: Don't exchange ghost cells here!
    // (This is what's missing in the actual code)
    // field.exchangeGhostCells(mesh);  // <- MISSING!

    // Now check what the ghost cells contain
    if (rank == 0) {
        std::cout << "\nAfter update WITHOUT exchange (Rank 0):\n";
        std::cout << "Local values updated: " << data[0] << " (should be " << 5.0 << ")\n";
        for (int i = 0; i < std::min(3, numGhost); ++i) {
            std::cout << "  ghost[" << i << "] = " << data[numLocal + i]
                      << " (still old value, should be 15 after exchange!)\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Now exchange and see the difference
    field.exchangeGhostCells(mesh);

    if (rank == 0) {
        std::cout << "\nAfter exchange (Rank 0):\n";
        std::cout << "Local values: " << data[0] << " (now 5 from Rank 0 update)\n";
        for (int i = 0; i < std::min(3, numGhost); ++i) {
            std::cout << "  ghost[" << i << "] = " << data[numLocal + i]
                      << " (now updated to 15 from Rank 1)\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify that exchange synced values correctly
    // Rank 0 should have Rank 1's values in ghost cells
    for (int i = 0; i < numGhost; ++i) {
        if (rank == 0) {
            EXPECT_EQ(data[numLocal + i], 15.0)
                << "Ghost cell " << i << " should have Rank 1's updated value (15)";
        }
    }

    SUCCEED();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
