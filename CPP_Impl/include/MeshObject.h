#ifndef MESH_OBJECT_H
#define MESH_OBJECT_H

#include "YamlParser.h"
#include "QuadElement.h"
#include <vector>
#include <memory>

namespace Vinci4D {

/**
 * @brief Creates a structured 2D grid based on mesh configuration parameters
 * 
 * Generates grid coordinates, cells, and faces (internal and boundary).
 */
class MeshObject {
public:
    /**
     * @brief Construct a MeshObject from configuration parser
     * 
     * @param configParser Parser containing mesh parameters
     */
    explicit MeshObject(const InputConfigParser& configParser);
    
    // Getters for mesh parameters
    std::array<double, 2> getXRange() const;
    std::array<double, 2> getYRange() const;
    std::array<int, 2> getCellCount() const;
    
    // Getters for coordinate arrays
    const std::vector<double>& getXCoordinates() const { return xCoords_; }
    const std::vector<double>& getYCoordinates() const { return yCoords_; }
    
    // Getters for cells
    const std::vector<Cell>& getCells() const { return cells_; }
    const std::vector<Cell*>& getInteriorCells() const { return interiorCells_; }
    const std::vector<Cell*>& getBoundaryCells() const { return boundaryCells_; }
    int getNumCells() const { return cells_.size(); }
    Cell* getCellByFlatId(int flatId);
    const Cell* getCellByFlatId(int flatId) const;
    
    // Getters for faces
    const std::vector<Face>& getFaces() const { return faces_; }
    const std::vector<Face*>& getInternalFaces() const { return internalFaces_; }
    const std::vector<Face*>& getBoundaryFaces() const { return boundaryFaces_; }
    int getNumFaces() const { return faces_.size(); }
    
    // Getters for boundary faces by location
    const std::vector<Face*>& getLeftBoundaryFaces() const { return leftBoundaryFaces_; }
    const std::vector<Face*>& getRightBoundaryFaces() const { return rightBoundaryFaces_; }
    const std::vector<Face*>& getTopBoundaryFaces() const { return topBoundaryFaces_; }
    const std::vector<Face*>& getBottomBoundaryFaces() const { return bottomBoundaryFaces_; }

private:
    void generateGrid();
    void generateFaces();
    
    InputConfigParser configParser_;
    
    // Coordinate arrays
    std::vector<double> xCoords_;
    std::vector<double> yCoords_;
    
    // Cells (actual storage)
    std::vector<Cell> cells_;
    std::vector<Cell*> interiorCells_;
    std::vector<Cell*> boundaryCells_;
    
    // Faces (actual storage)
    std::vector<Face> faces_;
    std::vector<Face*> internalFaces_;
    std::vector<Face*> boundaryFaces_;
    
    // Boundary faces by location
    std::vector<Face*> leftBoundaryFaces_;
    std::vector<Face*> rightBoundaryFaces_;
    std::vector<Face*> topBoundaryFaces_;
    std::vector<Face*> bottomBoundaryFaces_;
};

} // namespace Vinci4D

#endif // MESH_OBJECT_H
