#ifndef MESH_OBJECT_H
#define MESH_OBJECT_H

#include "YamlParser.h"
#include "QuadElement.h"
#include <vector>
#include <memory>
#include <map>
#include <mpi.h>

namespace Vinci4D {

/**
 * @brief Creates a structured 2D grid based on mesh configuration parameters
 * 
 * Generates grid coordinates, cells, and faces (internal and boundary).
 * Supports MPI parallelization with domain decomposition and ghost cells.
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
    
    // Getters for cells (sequential interface - deprecated in parallel)
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
    
    // MPI parallel interface
    int getMpiRank() const { return mpiRank_; }
    int getMpiSize() const { return mpiSize_; }
    int getNumLocalCells() const { return localCells_.size(); }
    int getNumGhostCells() const { return ghostCells_.size(); }
    int getGlobalNumCells() const { return globalNx_ * globalNy_; }
    const std::vector<Cell>& getLocalCells() const { return localCells_; }
    const std::vector<Cell>& getGhostCells() const { return ghostCells_; }
    const std::vector<Cell*>& getLocalInteriorCells() const { return localInteriorCells_; }
    const std::vector<int>& getNeighborRanks() const { return neighborRanks_; }
    int getLocalStartX() const { return localStartX_; }
    int getLocalEndX() const { return localEndX_; }
    int getLocalStartY() const { return localStartY_; }
    int getLocalEndY() const { return localEndY_; }
    
    // Convert between global and local indices
    int globalToLocal(int globalIdx) const;
    int localToGlobal(int localIdx) const;
    bool isLocalCell(int globalIdx) const;
    
    // Ghost cell communication
    void exchangeGhostCells(class FieldArray& field) const;
    
    // Ghost cell communication data structure and accessor
    struct GhostCellInfo {
        int neighborRank;
        std::vector<int> sendGlobalIds;  // Global IDs to send
        std::vector<int> recvGlobalIds;  // Global IDs to receive
        std::vector<int> sendLocalIds;   // Local IDs to send
        std::vector<int> recvLocalIds;   // Local IDs (in ghost array) to receive into
    };
    
    const std::vector<GhostCellInfo>& getGhostCommInfo() const { return ghostCommInfo_; }

private:
    void generateGrid();
    void generateFaces();
    void initializeMPI();
    void decomposeDomain();
    void identifyGhostCells();
    void identifyNeighbors();
    
    InputConfigParser configParser_;
    
    // Coordinate arrays
    std::vector<double> xCoords_;
    std::vector<double> yCoords_;
    
    // Cells (actual storage - for sequential compatibility)
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
    
    // MPI parallelization members
    MPI_Comm comm_;
    int mpiRank_;
    int mpiSize_;
    
    // Domain decomposition (2D Cartesian)
    int procGridX_;  // Number of procs in X direction
    int procGridY_;  // Number of procs in Y direction
    int rankX_;      // This rank's position in X
    int rankY_;      // This rank's position in Y
    
    // Global mesh dimensions
    int globalNx_;
    int globalNy_;
    
    // Local domain boundaries (inclusive)
    int localStartX_;
    int localEndX_;
    int localStartY_;
    int localEndY_;
    
    // Parallel cell storage
    std::vector<Cell> localCells_;     // Cells owned by this rank
    std::vector<Cell> ghostCells_;     // Ghost cells from neighbors
    std::vector<Cell*> localInteriorCells_;  // Interior cells excluding boundaries
    
    // Index mapping
    std::map<int, int> globalToLocalMap_;  // Global flat ID -> local index
    std::map<int, int> localToGlobalMap_;  // Local index -> global flat ID
    
    // Neighbor information
    std::vector<int> neighborRanks_;   // List of neighbor ranks
    enum Direction { NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3 };
    int neighborRank_[4];  // Neighbor ranks in each direction (-1 if none)
    
    // Ghost cell communication data
    std::vector<GhostCellInfo> ghostCommInfo_;
};

} // namespace Vinci4D

#endif // MESH_OBJECT_H
