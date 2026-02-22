#include <petsc.h>
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include "../include/MeshObject.h"
#include "../include/YamlParser.h"
#include <cmath>

using namespace Vinci4D;

class MeshObjectTest : public ::testing::Test {
protected:
    std::string tmpdir;
    
    void SetUp() override {
        // Create temp directory for test YAML files
        tmpdir = "/tmp/vinci4d_mesh_test_XXXXXX";
        char* result = mkdtemp(&tmpdir[0]);
        ASSERT_NE(result, nullptr) << "Failed to create temp directory";
    }
    
    void TearDown() override {
        // Clean up temp files
        system(("rm -rf " + tmpdir).c_str());
    }
    
    std::string writeYamlConfig(const std::string& filename, int nx, int ny,
                                double x_min, double x_max, double y_min, double y_max) {
        std::string filepath = tmpdir + "/" + filename;
        std::ofstream file(filepath);
        
        YAML::Node config;
        config["mesh_parameters"]["x_range"].push_back(x_min);
        config["mesh_parameters"]["x_range"].push_back(x_max);
        config["mesh_parameters"]["y_range"].push_back(y_min);
        config["mesh_parameters"]["y_range"].push_back(y_max);
        config["mesh_parameters"]["num_cells_x"] = nx;
        config["mesh_parameters"]["num_cells_y"] = ny;
        
        file << config;
        file.close();
        return filepath;
    }
};

TEST_F(MeshObjectTest, ThreeByThreeCellCenters) {
    std::string path = writeYamlConfig("mesh_3x3.yaml", 3, 3, 0, 1, 0, 1);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    int nx = 3, ny = 3;
    const auto& cells = mesh.getCells();
    
    // Expected centers at 1/6, 1/2, 5/6 for range [0,1] with 3 cells
    std::vector<double> expected = {1.0/6.0, 0.5, 5.0/6.0};
    
    // Extract x centers from first row (j=0)
    for (int i = 0; i < nx; i++) {
        double x_center = cells[i].getCentroid()[0];
        EXPECT_NEAR(x_center, expected[i], 1e-7) 
            << "X center at index " << i << " mismatch";
    }
    
    // Extract y centers from first column (i=0) across rows
    for (int j = 0; j < ny; j++) {
        double y_center = cells[j * nx].getCentroid()[1];
        EXPECT_NEAR(y_center, expected[j], 1e-7) 
            << "Y center at row " << j << " mismatch";
    }
}

TEST_F(MeshObjectTest, FourByEightUnitSquareCoordsAndCenters) {
    std::string path = writeYamlConfig("mesh_4x8.yaml", 4, 8, 0, 1, 0, 1);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    int nx = 4, ny = 8;
    const auto& cells = mesh.getCells();
    
    // Expected node coordinates
    std::vector<double> expected_x_nodes = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> expected_y_nodes;
    for (int i = 0; i <= ny; i++) {
        expected_y_nodes.push_back(static_cast<double>(i) / ny);
    }
    
    // Expected cell centers are midpoints between nodes
    std::vector<double> expected_x_centers;
    for (size_t i = 0; i < expected_x_nodes.size() - 1; i++) {
        expected_x_centers.push_back((expected_x_nodes[i] + expected_x_nodes[i+1]) / 2.0);
    }
    
    std::vector<double> expected_y_centers;
    for (size_t i = 0; i < expected_y_nodes.size() - 1; i++) {
        expected_y_centers.push_back((expected_y_nodes[i] + expected_y_nodes[i+1]) / 2.0);
    }
    
    // Extract x centers from first row (j=0)
    for (int i = 0; i < nx; i++) {
        double x_center = cells[i].getCentroid()[0];
        EXPECT_NEAR(x_center, expected_x_centers[i], 1e-12) 
            << "X center at index " << i << " mismatch";
    }
    
    // Extract y centers from first column (i=0) across rows
    for (int j = 0; j < ny; j++) {
        double y_center = cells[j * nx].getCentroid()[1];
        EXPECT_NEAR(y_center, expected_y_centers[j], 1e-12) 
            << "Y center at row " << j << " mismatch";
    }
}

TEST_F(MeshObjectTest, ThreeByaSixFaces) {
    std::string path = writeYamlConfig("mesh_3x6.yaml", 3, 6, 0, 1, 0, 1);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    int nx = 3, ny = 6;
    const auto& faces = mesh.getFaces();
    const auto& internal = mesh.getInternalFaces();
    const auto& boundary = mesh.getBoundaryFaces();
    
    // For 3x6: compute expected counts
    // vertical internal = (Nx-1)*Ny = (3-1)*6 = 12
    // horizontal internal = Nx*(Ny-1) = 3*(6-1) = 15
    // internal = 12 + 15 = 27
    // boundary = 2*Ny + 2*Nx = 2*6 + 2*3 = 18
    // total = 27 + 18 = 45
    EXPECT_EQ(internal.size(), 27) << "Should have 27 internal faces";
    EXPECT_EQ(boundary.size(), 18) << "Should have 18 boundary faces";
    EXPECT_EQ(faces.size(), 45UL) << "Should have 45 total faces";
    
    double x_min = 0, x_max = 1, y_min = 0, y_max = 1;
    double dx = (x_max - x_min) / nx;
    double dy = (y_max - y_min) / ny;
    
    // Check internal faces properties
    for (const auto& f : internal) {
        int left = f->getLeftCell();
        int right = f->getRightCell();
        EXPECT_GE(left, 0) << "Internal face should have valid left cell";
        EXPECT_GE(right, 0) << "Internal face should have valid right cell";
        
        auto normal = f->getNormalVector();
        double area = f->getArea();
        if (std::abs(normal[0]) > 1e-9) {
            EXPECT_NEAR(area, dy, 1e-10) << "Vertical face area should be dy";
        } else {
            EXPECT_NEAR(area, dx, 1e-10) << "Horizontal face area should be dx";
        }
    }
    
    // Check boundary faces properties
    for (const auto& f : boundary) {
        int left = f->getLeftCell();
        int right = f->getRightCell();
        EXPECT_GE(left, 0) << "Boundary face should have valid left cell";
        EXPECT_EQ(right, -1) << "Boundary face should not have right cell (should be -1)";
    }
}

TEST_F(MeshObjectTest, TenByFifteenCellNumbering) {
    std::string path = writeYamlConfig("mesh_10x15.yaml", 10, 15, 0, 1, 0, 1);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    int nx = 10, ny = 15;
    const auto& cells = mesh.getCells();
    
    EXPECT_EQ(cells.size(), static_cast<size_t>(nx * ny)) 
        << "Should have " << (nx*ny) << " cells";
    
    // Verify every cell's flattened id and indices mapping (row-major)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int flat = j * nx + i;
            const auto& cell = cells[flat];
            
            auto indices = cell.getIndices();
            EXPECT_EQ(indices[0], i) << "Cell " << flat << " i index mismatch";
            EXPECT_EQ(indices[1], j) << "Cell " << flat << " j index mismatch";
            EXPECT_GT(cell.getVolume(), 0) << "Cell " << flat << " volume should be positive";
        }
    }
}

TEST_F(MeshObjectTest, FiveByFiveInteriorCells) {
    std::string path = writeYamlConfig("mesh_5x5_interior.yaml", 5, 5, 0, 1, 0, 1);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    int nx = 5, ny = 5;
    const auto& interior = mesh.getInteriorCells();
    
    // For 5x5 grid, interior = (5-2) x (5-2) = 3x3 = 9
    int expected_interior = (nx - 2) * (ny - 2);
    EXPECT_EQ(interior.size(), static_cast<size_t>(expected_interior)) 
        << "Should have " << expected_interior << " interior cells";
    
    // Verify all interior cells have 1 <= i <= 3 and 1 <= j <= 3
    for (const auto& cell : interior) {
        auto indices = cell->getIndices();
        EXPECT_GE(indices[0], 1) << "Interior cell i should be >= 1";
        EXPECT_LE(indices[0], 3) << "Interior cell i should be <= 3";
        EXPECT_GE(indices[1], 1) << "Interior cell j should be >= 1";
        EXPECT_LE(indices[1], 3) << "Interior cell j should be <= 3";
    }
}

TEST_F(MeshObjectTest, FiveByFiveBoundaryCells) {
    std::string path = writeYamlConfig("mesh_5x5_boundary.yaml", 5, 5, 0, 1, 0, 1);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    int nx = 5, ny = 5;
    const auto& boundary = mesh.getBoundaryCells();
    
    // Total - interior = 25 - 9 = 16 boundary cells
    int expected_boundary = (nx * ny) - ((nx - 2) * (ny - 2));
    EXPECT_EQ(boundary.size(), static_cast<size_t>(expected_boundary)) 
        << "Should have " << expected_boundary << " boundary cells";
    
    // Verify all boundary cells have at least one index on edge
    for (const auto& cell : boundary) {
        auto indices = cell->getIndices();
        bool on_edge = (indices[0] == 0 || indices[0] == nx - 1 || 
                       indices[1] == 0 || indices[1] == ny - 1);
        EXPECT_TRUE(on_edge) << "Boundary cell should have at least one index on edge";
    }
}

TEST_F(MeshObjectTest, SixByFiveGetNumFaces) {
    int nx = 6, ny = 5;
    std::string path = writeYamlConfig("mesh_6x5.yaml", nx, ny, 0, 1, 0, 1);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    // Calculate expected unique faces
    // Total edge-face slots: 4 * (nx * ny)
    // Shared edges: horizontal (nx-1)*ny + vertical (ny-1)*nx
    // Unique faces: 4*nx*ny - ((nx-1)*ny + (ny-1)*nx)
    int total_edge_faces = 4 * (nx * ny);
    int horizontal_shared = (nx - 1) * ny;
    int vertical_shared = (ny - 1) * nx;
    int expected_faces = total_edge_faces - (horizontal_shared + vertical_shared);
    
    int actual_faces = mesh.getNumFaces();
    EXPECT_EQ(actual_faces, expected_faces) 
        << "Total faces should be " << expected_faces;
}

TEST_F(MeshObjectTest, TenByFiveBoundaries) {
    int nx = 10, ny = 5;
    double x_min = 0, x_max = 1, y_min = 0, y_max = 1;
    std::string path = writeYamlConfig("mesh_10x5.yaml", nx, ny, x_min, x_max, y_min, y_max);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    const auto& left = mesh.getLeftBoundaryFaces();
    const auto& right = mesh.getRightBoundaryFaces();
    const auto& top = mesh.getTopBoundaryFaces();
    const auto& bottom = mesh.getBottomBoundaryFaces();
    
    EXPECT_EQ(left.size(), static_cast<size_t>(ny)) << "Left boundary should have " << ny << " faces";
    EXPECT_EQ(right.size(), static_cast<size_t>(ny)) << "Right boundary should have " << ny << " faces";
    EXPECT_EQ(top.size(), static_cast<size_t>(nx)) << "Top boundary should have " << nx << " faces";
    EXPECT_EQ(bottom.size(), static_cast<size_t>(nx)) << "Bottom boundary should have " << nx << " faces";
    
    // Check left boundary face positions
    for (const auto& f : left) {
        auto center = f->getFaceCenter();
        EXPECT_NEAR(center[0], x_min, 1e-10) << "Left boundary face x should be at x_min";
        EXPECT_GE(center[1], y_min) << "Left boundary face y should be >= y_min";
        EXPECT_LE(center[1], y_max) << "Left boundary face y should be <= y_max";
    }
    
    // Check right boundary face positions
    for (const auto& f : right) {
        auto center = f->getFaceCenter();
        EXPECT_NEAR(center[0], x_max, 1e-10) << "Right boundary face x should be at x_max";
        EXPECT_GE(center[1], y_min) << "Right boundary face y should be >= y_min";
        EXPECT_LE(center[1], y_max) << "Right boundary face y should be <= y_max";
    }
    
    // Check top boundary face positions
    for (const auto& f : top) {
        auto center = f->getFaceCenter();
        EXPECT_NEAR(center[1], y_max, 1e-10) << "Top boundary face y should be at y_max";
        EXPECT_GE(center[0], x_min) << "Top boundary face x should be >= x_min";
        EXPECT_LE(center[0], x_max) << "Top boundary face x should be <= x_max";
    }
    
    // Check bottom boundary face positions
    for (const auto& f : bottom) {
        auto center = f->getFaceCenter();
        EXPECT_NEAR(center[1], y_min, 1e-10) << "Bottom boundary face y should be at y_min";
        EXPECT_GE(center[0], x_min) << "Bottom boundary face x should be >= x_min";
        EXPECT_LE(center[0], x_max) << "Bottom boundary face x should be <= x_max";
    }
}

TEST_F(MeshObjectTest, GetXRange) {
    std::vector<double> x_range = {-2.5, 3.0};
    std::string path = writeYamlConfig("mesh_get_x_range.yaml", 4, 4, 
                                       x_range[0], x_range[1], 0, 1);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    auto retrieved_range = mesh.getXRange();
    EXPECT_NEAR(retrieved_range[0], x_range[0], 1e-10) << "X range min mismatch";
    EXPECT_NEAR(retrieved_range[1], x_range[1], 1e-10) << "X range max mismatch";
}

TEST_F(MeshObjectTest, GetYRange) {
    std::vector<double> y_range = {-1.0, 2.0};
    std::string path = writeYamlConfig("mesh_get_y_range.yaml", 4, 4, 
                                       0, 1, y_range[0], y_range[1]);
    InputConfigParser parser(path);
    MeshObject mesh(parser);
    
    auto retrieved_range = mesh.getYRange();
    EXPECT_NEAR(retrieved_range[0], y_range[0], 1e-10) << "Y range min mismatch";
    EXPECT_NEAR(retrieved_range[1], y_range[1], 1e-10) << "Y range max mismatch";
}


int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
