#include "MeshObject.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace Vinci4D {

MeshObject::MeshObject(const InputConfigParser& configParser)
    : configParser_(configParser) {
    generateGrid();
    generateFaces();
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
