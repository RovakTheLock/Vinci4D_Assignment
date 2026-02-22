#include "MeshObject.h"
#include "FieldsHolder.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <set>
#include <map>
#include <mpi.h>

namespace Vinci4D {

MeshObject::MeshObject(const InputConfigParser& configParser)
    : configParser_(configParser), comm_(MPI_COMM_WORLD) {
    
    // Initialize MPI
    initializeMPI();
    
    // Generate grid (sequential for now, decompose after)
    generateGrid();
    generateFaces();
    
    // Perform domain decomposition for parallel execution
    decomposeDomain();
    identifyGhostCells();
    identifyNeighbors();
    
    if (mpiRank_ == 0) {
        std::cout << "Parallel mesh initialized on " << mpiSize_ << " ranks" << std::endl;
        std::cout << "Domain decomposition: " << procGridX_ << " x " << procGridY_ << std::endl;
    }
}

void MeshObject::initializeMPI() {
    MPI_Comm_rank(comm_, &mpiRank_);
    MPI_Comm_size(comm_, &mpiSize_);
    
    // Initialize neighbor array to no neighbors
    for (int i = 0; i < 4; ++i) {
        neighborRank_[i] = -1;
    }
    
    // Get global mesh dimensions
    globalNx_ = configParser_.getNumCellsX();
    globalNy_ = configParser_.getNumCellsY();
}

void MeshObject::decomposeDomain() {
    // Use MPI_Dims_create to get optimal 2D decomposition (minimizes surface area)
    int dims[2] = {0, 0};
    MPI_Dims_create(mpiSize_, 2, dims);
    procGridX_ = dims[0];
    procGridY_ = dims[1];
    
    // Determine this rank's position in the 2D grid
    rankX_ = mpiRank_ % procGridX_;
    rankY_ = mpiRank_ / procGridX_;
    
    // Calculate local domain boundaries (inclusive)
    // Distribute cells as evenly as possible
    int cellsPerProcX = globalNx_ / procGridX_;
    int extraCellsX = globalNx_ % procGridX_;
    
    localStartX_ = rankX_ * cellsPerProcX + std::min(rankX_, extraCellsX);
    int localNx = cellsPerProcX + (rankX_ < extraCellsX ? 1 : 0);
    localEndX_ = localStartX_ + localNx - 1;
    
    int cellsPerProcY = globalNy_ / procGridY_;
    int extraCellsY = globalNy_ % procGridY_;
    
    localStartY_ = rankY_ * cellsPerProcY + std::min(rankY_, extraCellsY);
    int localNy = cellsPerProcY + (rankY_ < extraCellsY ? 1 : 0);
    localEndY_ = localStartY_ + localNy - 1;
    
    // Extract local cells from global cell array
    int localIdx = 0;
    for (int j = localStartY_; j <= localEndY_; ++j) {
        for (int i = localStartX_; i <= localEndX_; ++i) {
            int globalIdx = j * globalNx_ + i;
            if (globalIdx < static_cast<int>(cells_.size())) {
                Cell localCell = cells_[globalIdx];
                localCell.setLocalId(localIdx);
                localCells_.push_back(localCell);
                
                // Build index mapping
                globalToLocalMap_[globalIdx] = localIdx;
                localToGlobalMap_[localIdx] = globalIdx;
                
                localIdx++;
            }
        }
    }
    
    // Identify local interior cells (not on physical boundary)
    // Store pointers into localCells_ after it's fully populated
    for (size_t idx = 0; idx < localCells_.size(); ++idx) {
        auto indices = localCells_[idx].getIndices();
        int i = indices[0];
        int j = indices[1];
        
        // Interior if not on physical boundary
        bool isInterior = (i > 0 && i < globalNx_ - 1 && j > 0 && j < globalNy_ - 1);
        
        if (isInterior) {
            localInteriorCells_.push_back(&localCells_[idx]);
        }
    }
    
    std::cout << "Rank " << mpiRank_ << ": local domain [" 
                << localStartX_ << ":" << localEndX_ << ", "
                << localStartY_ << ":" << localEndY_ << "], "
                << localCells_.size() << " cells" << std::endl;
}

void MeshObject::identifyGhostCells() {
    // Identify cells we need from neighbors (1 layer of ghost cells)
    std::map<int, bool> ghostCellsNeeded;
    
    for (const auto& cell : localCells_) {
        auto indices = cell.getIndices();
        int i = indices[0];
        int j = indices[1];
        
        // Check 4 neighbors (N, S, E, W)
        int neighbors[4][2] = {{i, j+1}, {i, j-1}, {i+1, j}, {i-1, j}};
        
        for (int n = 0; n < 4; ++n) {
            int ni = neighbors[n][0];
            int nj = neighbors[n][1];
            
            // Skip if out of global bounds
            if (ni < 0 || ni >= globalNx_ || nj < 0 || nj >= globalNy_) {
                continue;
            }
            
            int neighborGlobalIdx = nj * globalNx_ + ni;
            
            // If neighbor is not in our local domain, it's a ghost cell
            if (globalToLocalMap_.find(neighborGlobalIdx) == globalToLocalMap_.end()) {
                ghostCellsNeeded[neighborGlobalIdx] = true;
            }
        }
    }
    
    // Create ghost cells
    int ghostIdx = 0;
    for (const auto& pair : ghostCellsNeeded) {
        int globalIdx = pair.first;
        Cell ghostCell = cells_[globalIdx];
        ghostCell.setLocalId(localCells_.size() + ghostIdx);
        ghostCells_.push_back(ghostCell);
        
        // Add to global-to-local mapping
        globalToLocalMap_[globalIdx] = localCells_.size() + ghostIdx;
        
        ghostIdx++;
    }
}

void MeshObject::identifyNeighbors() {
    // Identify neighbor ranks in 4 directions
    // East neighbor (rankX + 1)
    if (rankX_ < procGridX_ - 1) {
        neighborRank_[EAST] = rankY_ * procGridX_ + (rankX_ + 1);
        neighborRanks_.push_back(neighborRank_[EAST]);
    }
    
    // West neighbor (rankX - 1)
    if (rankX_ > 0) {
        neighborRank_[WEST] = rankY_ * procGridX_ + (rankX_ - 1);
        neighborRanks_.push_back(neighborRank_[WEST]);
    }
    
    // North neighbor (rankY + 1)
    if (rankY_ < procGridY_ - 1) {
        neighborRank_[NORTH] = (rankY_ + 1) * procGridX_ + rankX_;
        neighborRanks_.push_back(neighborRank_[NORTH]);
    }
    
    // South neighbor (rankY - 1)
    if (rankY_ > 0) {
        neighborRank_[SOUTH] = (rankY_ - 1) * procGridX_ + rankX_;
        neighborRanks_.push_back(neighborRank_[SOUTH]);
    }
    
    // Build ghost cell communication patterns
    for (int dir = 0; dir < 4; ++dir) {
        if (neighborRank_[dir] == -1) continue;
        
        GhostCellInfo commInfo;
        commInfo.neighborRank = neighborRank_[dir];
        
        // Identify cells to send/receive for this neighbor
        for (const auto& cell : localCells_) {
            auto indices = cell.getIndices();
            int i = indices[0];
            int j = indices[1];
            
            bool isBoundaryCell = false;
            
            // Check if this cell is on the boundary with this neighbor
            if (dir == EAST && i == localEndX_) isBoundaryCell = true;
            if (dir == WEST && i == localStartX_) isBoundaryCell = true;
            if (dir == NORTH && j == localEndY_) isBoundaryCell = true;
            if (dir == SOUTH && j == localStartY_) isBoundaryCell = true;
            
            if (isBoundaryCell) {
                commInfo.sendLocalIds.push_back(cell.getLocalId());
                commInfo.sendGlobalIds.push_back(cell.getFlatId());
            }
        }
        
        // Identify ghost cells to receive from this neighbor
        for (const auto& ghostCell : ghostCells_) {
            auto indices = ghostCell.getIndices();
            int i = indices[0];
            int j = indices[1];
            
            bool isFromThisNeighbor = false;
            
            // Check if this ghost cell comes from this neighbor
            if (dir == EAST && i == localEndX_ + 1) isFromThisNeighbor = true;
            if (dir == WEST && i == localStartX_ - 1) isFromThisNeighbor = true;
            if (dir == NORTH && j == localEndY_ + 1) isFromThisNeighbor = true;
            if (dir == SOUTH && j == localStartY_ - 1) isFromThisNeighbor = true;
            
            if (isFromThisNeighbor) {
                commInfo.recvLocalIds.push_back(ghostCell.getLocalId());
                commInfo.recvGlobalIds.push_back(ghostCell.getFlatId());
            }
        }
        
        if (!commInfo.sendLocalIds.empty() || !commInfo.recvLocalIds.empty()) {
            ghostCommInfo_.push_back(commInfo);
        }
    }
}

int MeshObject::globalToLocal(int globalIdx) const {
    auto it = globalToLocalMap_.find(globalIdx);
    if (it != globalToLocalMap_.end()) {
        return it->second;
    }
    return -1;
}

int MeshObject::localToGlobal(int localIdx) const {
    auto it = localToGlobalMap_.find(localIdx);
    if (it != localToGlobalMap_.end()) {
        return it->second;
    }
    return -1;
}

bool MeshObject::isLocalCell(int globalIdx) const {
    int j = globalIdx / globalNx_;
    int i = globalIdx % globalNx_;
    return (i >= localStartX_ && i <= localEndX_ && j >= localStartY_ && j <= localEndY_);
}

void MeshObject::exchangeGhostCells(FieldArray& field) const {
    if (mpiSize_ == 1) {
        // No communication needed for single rank
        return;
    }
    
    int numComponents = field.getNumComponents();
    auto& data = field.getData();
    
    // Build a map for quick lookup of communication info by neighbor rank
    std::map<int, const GhostCellInfo*> commMap;
    for (const auto& commInfo : ghostCommInfo_) {
        commMap[commInfo.neighborRank] = &commInfo;
    }
    
    // Synchronize before exchange
    MPI_Barrier(comm_);
    
    // Every rank tries to communicate with every other rank
    // If no data to exchange, counts will be 0
    // This is the safest approach - everyone participates uniformly
    for (int target_rank = 0; target_rank < mpiSize_; ++target_rank) {
        if (target_rank == mpiRank_) {
            continue;  // Skip self
        }
        
        int sendCount = 0;
        int recvCount = 0;
        
        // Find if we have data for this target
        auto it = commMap.find(target_rank);
        if (it != commMap.end()) {
            sendCount = it->second->sendLocalIds.size();
            recvCount = it->second->recvLocalIds.size();
        }
        
        // Always allocate valid buffers 
        std::vector<double> sendBuffer(std::max(1, sendCount * numComponents), 0.0);
        std::vector<double> recvBuffer(std::max(1, recvCount * numComponents), 0.0);
        
        // Pack send data if we have any
        if (sendCount > 0 && it != commMap.end()) {
            const auto& commInfo = *(it->second);
            for (int i = 0; i < sendCount; ++i) {
                int localId = commInfo.sendLocalIds[i];
                for (int comp = 0; comp < numComponents; ++comp) {
                    sendBuffer[i * numComponents + comp] = data[localId * numComponents + comp];
                }
            }
        }
        
        // Deterministic tag based on rank pair
        int tag = std::min(mpiRank_, target_rank) * mpiSize_ + std::max(mpiRank_, target_rank);
        
        // Safe sendrecv that works even with 0-length messages
        MPI_Sendrecv(
            sendBuffer.data(), sendCount * numComponents, MPI_DOUBLE, target_rank, tag,
            recvBuffer.data(), recvCount * numComponents, MPI_DOUBLE, target_rank, tag,
            comm_, MPI_STATUS_IGNORE
        );
        
        // Unpack received data if we expected to receive
        if (recvCount > 0 && it != commMap.end()) {
            const auto& commInfo = *(it->second);
            for (int i = 0; i < recvCount; ++i) {
                int localId = commInfo.recvLocalIds[i];
                for (int comp = 0; comp < numComponents; ++comp) {
                    data[localId * numComponents + comp] = recvBuffer[i * numComponents + comp];
                }
            }
        }
    }
    
    // Synchronize after exchange
    MPI_Barrier(comm_);
}

std::array<double, 2> MeshObject::getXRange() const {
    return configParser_.getXRange();
}

std::array<double, 2> MeshObject::getYRange() const {
    return configParser_.getYRange();
}

std::array<int, 2> MeshObject::getCellCount() const {
    return {configParser_.getNumCellsX(), configParser_.getNumCellsY()};
}

Cell* MeshObject::getCellByFlatId(int flatId) {
    if (flatId >= 0 && flatId < static_cast<int>(cells_.size())) {
        return &cells_[flatId];
    }
    return nullptr;
}

const Cell* MeshObject::getCellByFlatId(int flatId) const {
    if (flatId >= 0 && flatId < static_cast<int>(cells_.size())) {
        return &cells_[flatId];
    }
    return nullptr;
}

void MeshObject::generateGrid() {
    // Get parameters
    auto xRange = configParser_.getXRange();
    auto yRange = configParser_.getYRange();
    int numCellsX = configParser_.getNumCellsX();
    int numCellsY = configParser_.getNumCellsY();
    
    // Generate 1D coordinate arrays
    xCoords_.resize(numCellsX + 1);
    yCoords_.resize(numCellsY + 1);
    
    double dx = (xRange[1] - xRange[0]) / numCellsX;
    double dy = (yRange[1] - yRange[0]) / numCellsY;
    
    for (int i = 0; i <= numCellsX; ++i) {
        xCoords_[i] = xRange[0] + i * dx;
    }
    
    for (int j = 0; j <= numCellsY; ++j) {
        yCoords_[j] = yRange[0] + j * dy;
    }
    
    // Compute cell centers and create Cell objects
    double cellVolume = dx * dy;
    cells_.reserve(numCellsX * numCellsY);
    
    for (int j = 0; j < numCellsY; ++j) {
        for (int i = 0; i < numCellsX; ++i) {
            int flatId = j * numCellsX + i;
            double cx = (xCoords_[i] + xCoords_[i + 1]) / 2.0;
            double cy = (yCoords_[j] + yCoords_[j + 1]) / 2.0;
            
            cells_.emplace_back(flatId, cellVolume, 
                               std::array<int, 2>{i, j},
                               std::array<double, 2>{cx, cy});
        }
    }
    
    // Identify interior and boundary cells
    for (int j = 0; j < numCellsY; ++j) {
        for (int i = 0; i < numCellsX; ++i) {
            int flatId = j * numCellsX + i;
            
            // Interior cells
            if (i > 0 && i < numCellsX - 1 && j > 0 && j < numCellsY - 1) {
                interiorCells_.push_back(&cells_[flatId]);
            }
            
            // Boundary cells
            if (i == 0 || i == numCellsX - 1 || j == 0 || j == numCellsY - 1) {
                boundaryCells_.push_back(&cells_[flatId]);
            }
        }
    }
    
    std::cout << "Grid generated: " << numCellsX << " x " << numCellsY << " cells" << std::endl;
}

void MeshObject::generateFaces() {
    int numCellsX = configParser_.getNumCellsX();
    int numCellsY = configParser_.getNumCellsY();
    
    auto xRange = configParser_.getXRange();
    auto yRange = configParser_.getYRange();
    double dx = (xRange[1] - xRange[0]) / numCellsX;
    double dy = (yRange[1] - yRange[0]) / numCellsY;
    
    // Pre-allocate faces vector to prevent reallocation invalidating pointers
    // Total faces: vertical + horizontal
    // Vertical: (numCellsX + 1) * numCellsY
    // Horizontal: numCellsX * (numCellsY + 1)
    int estimatedFaces = (numCellsX + 1) * numCellsY + numCellsX * (numCellsY + 1);
    faces_.reserve(estimatedFaces);
    
    int faceId = 0;
    
    // Vertical faces (separating cells in x-direction)
    for (int i = 0; i <= numCellsX; ++i) {
        for (int j = 0; j < numCellsY; ++j) {
            double xCoord = xCoords_[i];
            
            // Get y-coordinate (use centroid of adjacent cell)
            int refCell = (i > 0) ? (j * numCellsX + (i - 1)) : (j * numCellsX);
            double yCoord = cells_[refCell].getCentroid()[1];
            
            // Interior vertical faces
            if (i > 0 && i < numCellsX) {
                int leftCell = j * numCellsX + (i - 1);
                int rightCell = j * numCellsX + i;
                
                faces_.emplace_back(faceId++, leftCell, rightCell,
                                   std::array<double, 2>{xCoord, yCoord},
                                   std::array<double, 2>{1.0, 0.0});
                faces_.back().setArea(dy);
                internalFaces_.push_back(&faces_.back());
            }
            // Left boundary faces
            else if (i == 0) {
                int adjCell = j * numCellsX;
                faces_.emplace_back(faceId++, adjCell, -1,
                                   std::array<double, 2>{xCoord, yCoord},
                                   std::array<double, 2>{-1.0, 0.0});
                faces_.back().setArea(dy);
                boundaryFaces_.push_back(&faces_.back());
                leftBoundaryFaces_.push_back(&faces_.back());
            }
            // Right boundary faces
            else if (i == numCellsX) {
                int adjCell = j * numCellsX + (numCellsX - 1);
                faces_.emplace_back(faceId++, adjCell, -1,
                                   std::array<double, 2>{xCoord, yCoord},
                                   std::array<double, 2>{1.0, 0.0});
                faces_.back().setArea(dy);
                boundaryFaces_.push_back(&faces_.back());
                rightBoundaryFaces_.push_back(&faces_.back());
            }
        }
    }
    
    // Horizontal faces (separating cells in y-direction)
    for (int j = 0; j <= numCellsY; ++j) {
        for (int i = 0; i < numCellsX; ++i) {
            double yCoord = yCoords_[j];
            
            // Get x-coordinate (use centroid of adjacent cell)
            int refCell = (j > 0) ? ((j - 1) * numCellsX + i) : i;
            double xCoord = cells_[refCell].getCentroid()[0];
            
            // Interior horizontal faces
            if (j > 0 && j < numCellsY) {
                int bottomCell = (j - 1) * numCellsX + i;
                int topCell = j * numCellsX + i;
                
                faces_.emplace_back(faceId++, bottomCell, topCell,
                                   std::array<double, 2>{xCoord, yCoord},
                                   std::array<double, 2>{0.0, 1.0});
                faces_.back().setArea(dx);
                internalFaces_.push_back(&faces_.back());
            }
            // Bottom boundary faces
            else if (j == 0) {
                int adjCell = i;
                faces_.emplace_back(faceId++, adjCell, -1,
                                   std::array<double, 2>{xCoord, yCoord},
                                   std::array<double, 2>{0.0, -1.0});
                faces_.back().setArea(dx);
                boundaryFaces_.push_back(&faces_.back());
                bottomBoundaryFaces_.push_back(&faces_.back());
            }
            // Top boundary faces
            else if (j == numCellsY) {
                int adjCell = (numCellsY - 1) * numCellsX + i;
                faces_.emplace_back(faceId++, adjCell, -1,
                                   std::array<double, 2>{xCoord, yCoord},
                                   std::array<double, 2>{0.0, 1.0});
                faces_.back().setArea(dx);
                boundaryFaces_.push_back(&faces_.back());
                topBoundaryFaces_.push_back(&faces_.back());
            }
        }
    }
    
    std::cout << "Faces generated: " << internalFaces_.size() << " internal, "
              << boundaryFaces_.size() << " boundary" << std::endl;
}

} // namespace Vinci4D
