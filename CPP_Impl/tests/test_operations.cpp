#include <petsc.h>
#include <gtest/gtest.h>
#include <fstream>
#include "../include/Operations.h"
#include "../include/MeshObject.h"
#include "../include/YamlParser.h"
#include "../include/FieldsHolder.h"
#include <vector>

using namespace Vinci4D;

class OperationsTest : public ::testing::Test {
protected:
    // Configurable parameters for test YAML
    int num_cells_x;
    int num_cells_y;
    double cfl;
    
    void SetUp() override {
        // Set default values (can be changed in individual tests before SetUp is called)
        num_cells_x = 20;
        num_cells_y = 20;
        cfl = 0.5;
        
        // Create test-specific YAML configuration with embedded parameters
        std::string yaml_content = "mesh_parameters:\n"
            "  x_range: [0.0, 1.0]\n"
            "  y_range: [0.0, 1.0]\n"
            "  num_cells_x: " + std::to_string(num_cells_x) + "\n"
            "  num_cells_y: " + std::to_string(num_cells_y) + "\n"
            "simulation:\n"
            "  Re: 100\n"
            "  CFL: " + std::to_string(cfl) + "\n";
        
        // Write to rank-specific temp file to avoid race conditions
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::string configFile = "/tmp/operations_test_" + std::to_string(rank) + ".yaml";
        std::ofstream tmpYaml(configFile);
        tmpYaml << yaml_content;
        tmpYaml.close();
        
        parser = new InputConfigParser(configFile);
        mesh = new MeshObject(*parser);
        
        // For FieldArray sizing: track local and ghost cells separately
        int mpiSize = mesh->getMpiSize();
        if (mpiSize > 1) {
            numLocalCells = mesh->getNumLocalCells();
            numGhostCells = mesh->getNumGhostCells();
            // Synchronize all ranks at test start to prevent deadlocks in MPI operations
            MPI_Barrier(MPI_COMM_WORLD);
        } else {
            numLocalCells = mesh->getNumCells();
            numGhostCells = 0;
        }
        std::cout << " Finished setup...\n";
    }
    
    void TearDown() override {
        int mpiSize = mesh->getMpiSize();
        // Synchronize all ranks at test end to prevent deadlocks
        if (mpiSize > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
        delete mesh;
        delete parser;
    }
    
    InputConfigParser* parser;
    MeshObject* mesh;
    int numLocalCells;
    int numGhostCells;
};

TEST_F(OperationsTest, PerformanceTimer) {
    PerformanceTimer timer;
    
    EXPECT_NO_THROW(timer.startTimer("TestEvent"));
    
    // Do some work
    double sum = 0.0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    
    EXPECT_NO_THROW(timer.endTimer());
    EXPECT_GT(sum, 0);
}

TEST_F(OperationsTest, CFLTimeStepCompute) {
    double CFL = parser->getCFL();
    CFLTimeStepCompute cflCompute(mesh, CFL);
    
    double dt = cflCompute.computeTimeStep();
    double dx = 1.0/num_cells_x;
    double expectedDt = cfl*dx/1.0;
    EXPECT_EQ(dt, expectedDt) << "Time step mismatch";
}

TEST_F(OperationsTest, ThreeByThreeCellGradOpScalar) {
    // Test that we compute the right gradient given a linear field with slope 1
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 20
  num_cells_y: 20
simulation:
  Re: 100
  CFL: 0.5
)";
    
    // Write temporary config (rank-specific to avoid race conditions)
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string configFile = "/tmp/gradient_test_20x20_" + std::to_string(rank) + ".yaml";
    std::ofstream tmpYaml(configFile);
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser testParser(configFile);
    MeshObject testMesh(testParser);
    
    int mpiSize = testMesh.getMpiSize();
    int numLocal = (mpiSize > 1) ? testMesh.getNumLocalCells() : testMesh.getNumCells();
    int numGhost = (mpiSize > 1) ? testMesh.getNumGhostCells() : 0;
    
    // Initialize pressure field to be linear in x for testing gradient
    FieldArray pressureField("pressure", DimType::SCALAR, numLocal, numGhost);
    auto& pressureData = pressureField.getData();
    
    // Set field values in local+ghost cells
    const auto& cellsToInit = (mpiSize > 1) ? testMesh.getLocalCells() : testMesh.getCells();
    for (const auto& cell : cellsToInit) {
        int cellID = (mpiSize > 1) ? cell.getLocalId() : cell.getFlatId();
        auto centroid = cell.getCentroid();
        pressureData[cellID] = centroid[0];  // Set pressure to x-coordinate of cell center
    }
    
    // Compute gradient
    FieldArray gradPressureField("grad_pressure", DimType::VECTOR, numLocal, numGhost);
    
    // Exchange ghost cells before gradient computation in parallel mode
    if (mpiSize > 1) {
        pressureField.exchangeGhostCells(testMesh);
    }
    
    ComputeCellGradient gradientOp(&testMesh, &pressureField, &gradPressureField);
    gradientOp.computeScalarGradient();
    
    auto& gradData = gradPressureField.getData();
    
    // For uniform grid, expect gradient to be approximately 1 in x direction and 0 in y direction
    // in interior cells. Each rank checks only its local interior cells
    const auto& interiorCellsToCheck = (mpiSize > 1) ? 
        testMesh.getLocalInteriorCells() : testMesh.getInteriorCells();
    
    for (const auto* interiorCell : interiorCellsToCheck) {
        int cellID = (mpiSize > 1) ? interiorCell->getLocalId() : interiorCell->getFlatId();
        double grad_x = gradData[cellID * MAX_DIM];          // x component of gradient
        double grad_y = gradData[cellID * MAX_DIM + 1];      // y component of gradient
        
        EXPECT_NEAR(grad_x, 1.0, 1e-5) 
            << "X-gradient mismatch at interior cell (local ID: " << cellID << ")";
        EXPECT_NEAR(grad_y, 0.0, 1e-5) 
            << "Y-gradient mismatch at interior cell (local ID: " << cellID << ")";
    }
}

TEST_F(OperationsTest, TwentyByTwentyCellGradOpScalarWithSlope) {
    // Test that we compute the right gradient given a linear field with non-1 slope
    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 20
  num_cells_y: 20
simulation:
  Re: 100
  CFL: 0.5
)";
    
    // Write temporary config (rank-specific to avoid race conditions)
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string configFile = "/tmp/gradient_test_20x20_slope_" + std::to_string(rank) + ".yaml";
    std::ofstream tmpYaml(configFile);
    tmpYaml << yaml_content;
    tmpYaml.close();
    
    InputConfigParser testParser(configFile);
    MeshObject testMesh(testParser);
    
    int mpiSize = testMesh.getMpiSize();
    int numLocal = (mpiSize > 1) ? testMesh.getNumLocalCells() : testMesh.getNumCells();
    int numGhost = (mpiSize > 1) ? testMesh.getNumGhostCells() : 0;
    
    double slope = 10.0;
    
    // Initialize pressure field to be linear in x with given slope
    FieldArray pressureField("pressure", DimType::SCALAR, numLocal, numGhost);
    auto& pressureData = pressureField.getData();
    
    // Set field values in local+ghost cells
    const auto& cellsToInit = (mpiSize > 1) ? testMesh.getLocalCells() : testMesh.getCells();
    for (const auto& cell : cellsToInit) {
        int cellID = (mpiSize > 1) ? cell.getLocalId() : cell.getFlatId();
        auto centroid = cell.getCentroid();
        pressureData[cellID] = slope * centroid[0];  // Set pressure to slope * x-coordinate
    }
    
    // Compute gradient
    FieldArray gradPressureField("grad_pressure", DimType::VECTOR, numLocal, numGhost);
    
    // Exchange ghost cells before gradient computation in parallel mode
    if (mpiSize > 1) {
        pressureField.exchangeGhostCells(testMesh);
    }
    
    ComputeCellGradient gradientOp(&testMesh, &pressureField, &gradPressureField);
    gradientOp.computeScalarGradient();
    
    auto& gradData = gradPressureField.getData();
    
    // For uniform grid, expect gradient to be approximately 'slope' in x direction and 0 in y direction
    // in interior cells. Each rank checks only its local interior cells
    const auto& interiorCellsToCheck = (mpiSize > 1) ? 
        testMesh.getLocalInteriorCells() : testMesh.getInteriorCells();
    
    for (const auto* interiorCell : interiorCellsToCheck) {
        int cellID = (mpiSize > 1) ? interiorCell->getLocalId() : interiorCell->getFlatId();
        double grad_x = gradData[cellID * MAX_DIM];          // x component of gradient
        double grad_y = gradData[cellID * MAX_DIM + 1];      // y component of gradient
        
        EXPECT_NEAR(grad_x, slope, 1e-5) 
            << "X-gradient mismatch at interior cell " << cellID 
            << " (expected " << slope << ", got " << grad_x << ")";
        EXPECT_NEAR(grad_y, 0.0, 1e-5) 
            << "Y-gradient mismatch at interior cell " << cellID;
    }
    
    // Also check gradient at ghost cells (if running in parallel)
    gradPressureField.exchangeGhostCells(testMesh);
    if (mpiSize > 1) {
        const auto& ghostCells = testMesh.getGhostCells();
        for (size_t ghostIdx = 0; ghostIdx < ghostCells.size(); ++ghostIdx) {
            int ghostLocalId = numLocal + ghostIdx;
            double actual_grad_x = gradData[ghostLocalId * MAX_DIM];
            double actual_grad_y = gradData[ghostLocalId * MAX_DIM + 1];
            if (ghostCells[ghostIdx].getIndices()[0] == 0 || ghostCells[ghostIdx].getIndices()[0] == testMesh.getCellCount()[0] - 1 ||
                ghostCells[ghostIdx].getIndices()[1] == 0 || ghostCells[ghostIdx].getIndices()[1] == testMesh.getCellCount()[1] - 1) {
                    continue; //ignore true domain boundaries
            }
            EXPECT_NEAR(actual_grad_x, slope, 1e-4) 
                << "X-gradient mismatch at ghost cell (" << ghostCells[ghostIdx].getIndices()[0] << "," << ghostCells[ghostIdx].getIndices()[1] << ")"
                << " local ID = " << ghostLocalId;
            EXPECT_NEAR(actual_grad_y, 0.0, 1e-4) 
                << "Y-gradient mismatch at ghost cell (" << ghostCells[ghostIdx].getIndices()[0] << "," << ghostCells[ghostIdx].getIndices()[1] << ")"
                << " local ID = " << ghostLocalId;
        }
    }
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    
    // These tests now work in both sequential and parallel modes
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
