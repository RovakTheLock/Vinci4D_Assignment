#include <petsc.h>
#include <gtest/gtest.h>
#include "../include/MeshObject.h"
#include "../include/YamlParser.h"

using namespace Vinci4D;

class MeshObjectTest : public ::testing::Test {
protected:
    void SetUp() override {
        // PETSc is initialized in main()
    }
};

TEST_F(MeshObjectTest, MeshDimensions) {
    InputConfigParser parser("../config/input.yaml");
    MeshObject mesh(parser);
    
    auto cellCount = mesh.getCellCount();
    EXPECT_EQ(cellCount[0], 50) << "X cell count should be 50";
    EXPECT_EQ(cellCount[1], 50) << "Y cell count should be 50";
}

TEST_F(MeshObjectTest, TotalCellCount) {
    InputConfigParser parser("../config/input.yaml");
    MeshObject mesh(parser);
    
    int totalCells = mesh.getNumCells();
    EXPECT_EQ(totalCells, 2500) << "Total cells should be 2500";
}

TEST_F(MeshObjectTest, InteriorAndBoundaryCells) {
    InputConfigParser parser("../config/input.yaml");
    MeshObject mesh(parser);
    
    int totalCells = mesh.getNumCells();
    int interiorCells = mesh.getInteriorCells().size();
    int boundaryCells = mesh.getBoundaryCells().size();
    
    EXPECT_GT(interiorCells, 0) << "Should have interior cells";
    EXPECT_GT(boundaryCells, 0) << "Should have boundary cells";
    EXPECT_LE(interiorCells + boundaryCells, totalCells) << "Cell counts should be consistent";
}

TEST_F(MeshObjectTest, FaceCounts) {
    InputConfigParser parser("../config/input.yaml");
    MeshObject mesh(parser);
    
    int totalFaces = mesh.getNumFaces();
    int internalFaces = mesh.getInternalFaces().size();
    int boundaryFaces = mesh.getBoundaryFaces().size();
    
    EXPECT_GT(totalFaces, 0) << "Should have faces";
    EXPECT_EQ(totalFaces, internalFaces + boundaryFaces) << "Face counts should be consistent";
}

TEST_F(MeshObjectTest, CellCentroids) {
    InputConfigParser parser("../config/input.yaml");
    MeshObject mesh(parser);
    
    const Cell* cell0 = mesh.getCellByFlatId(0);
    ASSERT_NE(cell0, nullptr) << "Cell 0 should exist";
    
    auto centroid = cell0->getCentroid();
    EXPECT_GT(centroid[0], 0) << "X centroid should be positive";
    EXPECT_GT(centroid[1], 0) << "Y centroid should be positive";
}

TEST_F(MeshObjectTest, FaceAreas) {
    InputConfigParser parser("../config/input.yaml");
    MeshObject mesh(parser);
    
    const auto& faces = mesh.getInternalFaces();
    ASSERT_FALSE(faces.empty()) << "Should have internal faces";
    
    double area = faces[0]->getArea();
    EXPECT_GT(area, 0) << "Face area should be positive";
}

TEST_F(MeshObjectTest, BoundaryFacesByLocation) {
    InputConfigParser parser("../config/input.yaml");
    MeshObject mesh(parser);
    
    EXPECT_GT(mesh.getLeftBoundaryFaces().size(), 0) << "Should have left boundary faces";
    EXPECT_GT(mesh.getRightBoundaryFaces().size(), 0) << "Should have right boundary faces";
    EXPECT_GT(mesh.getTopBoundaryFaces().size(), 0) << "Should have top boundary faces";
    EXPECT_GT(mesh.getBottomBoundaryFaces().size(), 0) << "Should have bottom boundary faces";
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
