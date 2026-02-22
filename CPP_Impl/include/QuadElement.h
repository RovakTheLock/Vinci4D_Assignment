#ifndef QUAD_ELEMENT_H
#define QUAD_ELEMENT_H

#include <tuple>
#include <array>

namespace Vinci4D {

/**
 * @brief Represents a face in a finite volume mesh
 * 
 * Stores references to left and right cell indices for flux computations.
 */
class Face {
public:
    /**
     * @brief Construct a Face
     * 
     * @param faceId Unique identifier for the face
     * @param leftCellIdx Index of left cell (flattened ID)
     * @param rightCellIdx Index of right cell (flattened ID), -1 for boundary faces
     * @param faceCoords Midpoint coordinates of the face (x, y)
     * @param normalVector Outward normal vector (nx, ny)
     */
    Face(int faceId, int leftCellIdx, int rightCellIdx, 
         std::array<double, 2> faceCoords, std::array<double, 2> normalVector);
    
    // Getters
    int getId() const { return id_; }
    int getLeftCell() const { return leftCellFlatId_; }
    int getRightCell() const { return rightCellFlatId_; }
    std::array<double, 2> getFaceCenter() const { return faceCoords_; }
    std::array<double, 2> getNormalVector() const { return normalVector_; }
    double getArea() const { return area_; }
    bool isBoundaryFace() const { return isBoundary_; }
    double getMassFlux() const { return massFlux_; }
    
    // Setters
    void setArea(double area) { area_ = area; }
    void setMassFlux(double flux) { massFlux_ = flux; }
    
    // Public for direct access (matching Python implementation)
    double massFlux_;

private:
    int id_;
    int leftCellFlatId_;
    int rightCellFlatId_;
    std::array<double, 2> faceCoords_;
    std::array<double, 2> normalVector_;
    double area_;
    bool isBoundary_;
};

/**
 * @brief Represents a cell in a finite volume mesh
 * 
 * Stores a flattened array ID and the cell volume with optional (i,j) indices.
 * For parallel execution, also stores a local ID within rank's subdomain.
 */
class Cell {
public:
    /**
     * @brief Construct a Cell
     * 
     * @param flatId Flattened cell ID (e.g., row-major index)
     * @param volume Cell volume (area in 2D)
     * @param indices Optional (i, j) integer indices
     * @param centroid Optional (x, y) centroid coordinates
     */
    Cell(int flatId, double volume, 
         std::array<int, 2> indices = {-1, -1}, 
         std::array<double, 2> centroid = {0.0, 0.0});
    
    // Getters
    int getFlatId() const { return flatId_; }
    int getLocalId() const { return localId_; }
    double getVolume() const { return volume_; }
    std::array<int, 2> getIndices() const { return indices_; }
    std::array<double, 2> getCentroid() const { return centroid_; }
    
    // Setters
    void setCentroid(const std::array<double, 2>& centroid) { centroid_ = centroid; }
    void setLocalId(int localId) { localId_ = localId; }

private:
    int flatId_;
    int localId_;  // Local ID within rank (for MPI)
    double volume_;
    std::array<int, 2> indices_;
    std::array<double, 2> centroid_;
};

} // namespace Vinci4D

#endif // QUAD_ELEMENT_H
