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
#include "../include/AssembleAlgorithms.h"

using namespace Vinci4D;

class BoundaryDiagnosticsTest : public ::testing::Test {
protected:
    InputConfigParser* parser;
    MeshObject* mesh;
    int rank = 0;
    int numRanks = 1;
    int numLocalCells = 0;
    int numGhostCells = 0;
    int globalCells = 0;
    FieldArray* velocityField;

    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

        parser = new InputConfigParser("../config/input.yaml");
        mesh = new MeshObject(*parser);

        if (numRanks > 1) {
            numLocalCells = mesh->getNumLocalCells();
            numGhostCells = mesh->getNumGhostCells();
            globalCells = mesh->getGlobalNumCells();
        } else {
            numLocalCells = mesh->getNumCells();
            numGhostCells = 0;
            globalCells = numLocalCells;
        }

        velocityField = new FieldArray("velocity", DimType::VECTOR, numLocalCells, numGhostCells);
        velocityField->initializeConstant(0.0);

        if (rank == 0) {
            std::cout << "\n=== Boundary Diagnostics Test (np=" << numRanks << ") ===\n";
            std::cout << "Global cells: " << globalCells << "\n";
            std::cout << "Local cells per rank: " << numLocalCells << "\n";
            std::cout << "Ghost cells per rank: " << numGhostCells << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void TearDown() override {
        delete velocityField;
        delete mesh;
        delete parser;
    }

    void dumpBoundaryVelocities(const std::string& label) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (numRanks == 1) {
            // Serial: just print last row
            if (rank == 0) {
                int nx = mesh->getCellCount()[0];
                int ny = mesh->getCellCount()[1];
                std::cout << "\n" << label << ": Last row cells (y=" << ny-1 << "):\n";
                
                auto& data = velocityField->getData();
                for (int i = 0; i < nx; ++i) {
                    int cellIdx = (ny - 1) * nx + i;
                    if (cellIdx < numLocalCells) {
                        std::cout << "  u[" << i << "] = " << std::scientific << std::setprecision(6)
                                  << data[cellIdx * 2 + 0] << "\n";
                    }
                }
            }
        } else if (numRanks == 2) {
            // Parallel: print first and last rows of each rank
            if (rank == 0) {
                std::cout << "\n" << label << "\n";
                std::cout << "Rank 0 first row:\n";
                auto& data = velocityField->getData();
                for (int i = 0; i < std::min(3, numLocalCells); ++i) {
                    std::cout << "  u[" << i << "] = " << std::scientific << std::setprecision(6)
                              << data[i * 2 + 0] << "\n";
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (rank == 1) {
                std::cout << "Rank 1 first row:\n";
                auto& data = velocityField->getData();
                for (int i = 0; i < std::min(3, numLocalCells); ++i) {
                    std::cout << "  u[" << i << "] = " << std::scientific << std::setprecision(6)
                              << data[i * 2 + 0] << "\n";
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
};

TEST_F(BoundaryDiagnosticsTest, FieldExchange) {
    // Initialize with rank identifier on local cells
    auto& data = velocityField->getData();
    for (int i = 0; i < numLocalCells; ++i) {
        data[i * 2 + 0] = static_cast<double>(rank);
        data[i * 2 + 1] = 0.0;
    }

    dumpBoundaryVelocities("After init with rank ID");

    // Exchange ghost cells
    velocityField->exchangeGhostCells(*mesh);

    dumpBoundaryVelocities("After ghost cell exchange");

    if (numRanks == 2) {
        // For parallel: ghost cells should contain neighbor values
        for (int i = numLocalCells; i < numLocalCells + numGhostCells; ++i) {
            if (i < data.size()) {
                EXPECT_NE(data[i * 2 + 0], static_cast<double>(rank))
                    << "Ghost cell should have neighbor value";
            }
        }
    }

    SUCCEED();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
