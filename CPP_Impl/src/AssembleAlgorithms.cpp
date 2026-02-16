#include "AssembleAlgorithms.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <vector>

namespace Vinci4D {

AssembleCellVectorTimeTerm::AssembleCellVectorTimeTerm(
    const std::string& name, FieldArray* fieldsHolderNew, FieldArray* fieldsHolderOld,
    LinearSystem* linearSystem, MeshObject* meshObject, double dt)
    : AssembleSystemBase(name, fieldsHolderNew, linearSystem, meshObject),
      dt_(dt), fieldHolderOld_(fieldsHolderOld) {
    
    if (fieldsHolderNew->getType() != DimType::VECTOR) {
        throw std::runtime_error("AssembleCellVectorTimeTerm requires a vector field for state NEW");
    }
    if (fieldsHolderOld->getType() != DimType::VECTOR) {
        throw std::runtime_error("AssembleCellVectorTimeTerm requires a vector field for state OLD");
    }
}

void AssembleCellVectorTimeTerm::assemble() {
    // Use local cells for parallel execution
    bool isParallel = (meshObject_->getMpiSize() > 1);
    const auto& cells = isParallel ? 
                        meshObject_->getLocalCells() : meshObject_->getCells();
    auto& dataNew = fieldsHolder_->getData();
    const auto& dataOld = fieldHolderOld_->getData();
    int numComponents = fieldsHolder_->getNumComponents();
    
    for (const auto& cell : cells) {
        // In parallel mode: use local/global IDs
        // In sequential mode: flatId serves as both local and global ID
        int localId = isParallel ? cell.getLocalId() : cell.getFlatId();
        int globalId = isParallel ? meshObject_->localToGlobal(localId) : cell.getFlatId();
        double cellVolume = cell.getVolume();
        
        for (int comp = 0; comp < numComponents; ++comp) {
            double lhsFactor = cellVolume / dt_;
            int localIndex = localId * numComponents + comp;
            int globalIndex = globalId * numComponents + comp;
            double value = dataNew[localIndex] - dataOld[localIndex];
            
            linearSystem_->addLhs(globalIndex, globalIndex, -lhsFactor);
            linearSystem_->addRhs(globalIndex, value * lhsFactor);
        }
    }
}

AssembleInteriorVectorDiffusionToLinSystem::AssembleInteriorVectorDiffusionToLinSystem(
    const std::string& name, FieldArray* fieldsHolder, LinearSystem* linearSystem,
    MeshObject* meshObject, double diffusionCoeff)
    : AssembleSystemBase(name, fieldsHolder, linearSystem, meshObject),
      diffusionCoeff_(diffusionCoeff) {
    
    if (fieldsHolder->getType() != DimType::VECTOR) {
        throw std::runtime_error("AssembleInteriorVectorDiffusionToLinSystem requires a vector field");
    }
}

void AssembleInteriorVectorDiffusionToLinSystem::assemble() {
    // Exchange ghost cells before assembly
    fieldsHolder_->exchangeGhostCells(*meshObject_);
    
    const auto& internalFaces = meshObject_->getInternalFaces();
    auto& data = fieldsHolder_->getData();
    
    int numCells = meshObject_->getNumCells();
    bool isParallel = (meshObject_->getMpiSize() > 1);
    
    for (const auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        // Debug: check if cell IDs are valid
        if (leftCellID < 0 || leftCellID >= numCells || 
            rightCellID < 0 || rightCellID >= numCells) {
            std::cerr << "Invalid cell IDs in internal face: leftID=" << leftCellID 
                      << " rightID=" << rightCellID << " (numCells=" << numCells << ")" << std::endl;
            throw std::runtime_error("Invalid cell ID in internal face");
        }
        
        // Skip faces not owned by this rank in parallel
        if (isParallel && !meshObject_->isLocalCell(leftCellID)) {
            continue;
        }
        
        // Get cell centroids
        const Cell* leftCell = meshObject_->getCellByFlatId(leftCellID);
        const Cell* rightCell = meshObject_->getCellByFlatId(rightCellID);
        
        if (!leftCell || !rightCell) {
            throw std::runtime_error("Invalid cell ID in internal face");
        }
        
        auto leftCentroid = leftCell->getCentroid();
        auto rightCentroid = rightCell->getCentroid();
        auto normal = face->getNormalVector();
        
        // Compute distance between cell centers along normal direction
        double distance = 0.0;
        for (int i = 0; i < MAX_DIM; ++i) {
            distance += (rightCentroid[i] - leftCentroid[i]) * normal[i];
        }
        
        double faceArea = face->getArea();
        double diffusionFactor = diffusionCoeff_ * faceArea / distance;
        
        // Use global indices for parallel assembly
        int globalLeftID = isParallel ? leftCellID : leftCellID;
        int globalRightID = isParallel ? rightCellID : rightCellID;
        int localLeftID = isParallel ? meshObject_->globalToLocal(leftCellID) : leftCellID;
        int localRightID = isParallel ? meshObject_->globalToLocal(rightCellID) : rightCellID;
        
        // Assemble for each component
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            int localRowLeft = localLeftID * MAX_DIM + comp;
            int localRowRight = localRightID * MAX_DIM + comp;
            int globalRowLeft = globalLeftID * MAX_DIM + comp;
            int globalRowRight = globalRightID * MAX_DIM + comp;
            
            double phiLeft = (localLeftID >= 0) ? data[localRowLeft] : 0.0;
            double phiRight = (localRightID >= 0) ? data[localRowRight] : 0.0;
            double flux = diffusionFactor * (phiRight - phiLeft);
            
            // Contribution to left cell
            linearSystem_->addLhs(globalRowLeft, globalRowLeft, -diffusionFactor);
            linearSystem_->addLhs(globalRowLeft, globalRowRight, diffusionFactor);
            linearSystem_->addRhs(globalRowLeft, -flux);
            
            // Contribution to right cell
            linearSystem_->addLhs(globalRowRight, globalRowRight, -diffusionFactor);
            linearSystem_->addLhs(globalRowRight, globalRowLeft, diffusionFactor);
            linearSystem_->addRhs(globalRowRight, flux);
        }
    }
}

AssembleCellVectorPressureGradientToLinSystem::AssembleCellVectorPressureGradientToLinSystem(
    const std::string& name, FieldArray* fieldsHolder, LinearSystem* linearSystem,
    MeshObject* meshObject)
    : AssembleSystemBase(name, fieldsHolder, linearSystem, meshObject) {
    
    if (fieldsHolder->getType() != DimType::VECTOR) {
        throw std::runtime_error("AssembleCellVectorPressureGradient requires a vector field");
    }
}

void AssembleCellVectorPressureGradientToLinSystem::assemble() {
    bool isParallel = (meshObject_->getMpiSize() > 1);
    const auto& cells = isParallel ? meshObject_->getLocalCells() : meshObject_->getCells();
    auto& data = fieldsHolder_->getData();
    int numComponents = fieldsHolder_->getNumComponents();
    
    for (const auto& cell : cells) {
        int localId = isParallel ? cell.getLocalId() : cell.getFlatId();
        int globalId = isParallel ? meshObject_->localToGlobal(localId) : cell.getFlatId();
        double cellVolume = cell.getVolume();
        
        for (int comp = 0; comp < numComponents; ++comp) {
            int localIndex = localId * numComponents + comp;
            int globalIndex = globalId * numComponents + comp;
            double value = data[localIndex] * cellVolume;
            linearSystem_->addRhs(globalIndex, value);
        }
    }
}

AssembleDirichletBoundaryVectorDiffusionToLinSystem::AssembleDirichletBoundaryVectorDiffusionToLinSystem(
    const std::string& name, FieldArray* fieldsHolder, LinearSystem* linearSystem,
    MeshObject* meshObject, Boundary boundaryType,
    const std::array<double, 2>& boundaryValue, double diffusionCoeff)
    : AssembleSystemBase(name, fieldsHolder, linearSystem, meshObject),
      boundaryType_(boundaryType), boundaryValue_(boundaryValue), diffusionCoeff_(diffusionCoeff) {
    
    if (fieldsHolder->getType() != DimType::VECTOR) {
        throw std::runtime_error("Boundary vector diffusion requires a vector field");
    }
    
    // Populate boundary faces based on boundary type
    if (boundaryType == Boundary::LEFT) {
        boundaryFaces_ = meshObject->getLeftBoundaryFaces();
    } else if (boundaryType == Boundary::RIGHT) {
        boundaryFaces_ = meshObject->getRightBoundaryFaces();
    } else if (boundaryType == Boundary::TOP) {
        boundaryFaces_ = meshObject->getTopBoundaryFaces();
    } else if (boundaryType == Boundary::BOTTOM) {
        boundaryFaces_ = meshObject->getBottomBoundaryFaces();
    }
}

void AssembleDirichletBoundaryVectorDiffusionToLinSystem::assemble() {
    auto& data = fieldsHolder_->getData();
    bool isParallel = (meshObject_->getMpiSize() > 1);
    
    for (auto* face : boundaryFaces_) {
        int cellID = face->getLeftCell();
        if (cellID < 0 || cellID >= meshObject_->getNumCells()) continue;
        
        // Skip if not local cell in parallel
        if (isParallel && !meshObject_->isLocalCell(cellID)) {
            continue;
        }
        
        const Cell* cell = meshObject_->getCellByFlatId(cellID);
        if (!cell) continue;
        
        double faceArea = face->getArea();
        auto normal = face->getNormalVector();
        auto cellCentroid = cell->getCentroid();
        auto faceCentroid = face->getFaceCenter();
        
        // Distance from cell centroid to face centroid
        double distX = faceCentroid[0] - cellCentroid[0];
        double distY = faceCentroid[1] - cellCentroid[1];
        double normalDistance = normal[0] * distX + normal[1] * distY;
        
        double lhsFactor = (diffusionCoeff_ * faceArea) / normalDistance;
        
        int localId = isParallel ? meshObject_->globalToLocal(cellID) : cellID;
        int globalId = isParallel ? cellID : cellID;
        
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            int localIndex = localId * MAX_DIM + comp;
            int globalRowIdx = globalId * MAX_DIM + comp;
            double gradDotArea = (boundaryValue_[comp] - data[localIndex]) * lhsFactor;
            linearSystem_->addRhs(globalRowIdx, -gradDotArea);
            linearSystem_->addLhs(globalRowIdx, globalRowIdx, -lhsFactor);
        }
    }
}

AssembleInteriorVectorAdvectionToLinSystem::AssembleInteriorVectorAdvectionToLinSystem(
    const std::string& name, FieldArray* fieldsHolder, LinearSystem* linearSystem,
    MeshObject* meshObject)
    : AssembleSystemBase(name, fieldsHolder, linearSystem, meshObject) {
    
    if (fieldsHolder->getType() != DimType::VECTOR) {
        throw std::runtime_error("Interior vector advection requires a vector field");
    }
}

void AssembleInteriorVectorAdvectionToLinSystem::assemble() {
    fieldsHolder_->exchangeGhostCells(*meshObject_);
    
    const auto& internalFaces = meshObject_->getInternalFaces();
    auto& data = fieldsHolder_->getData();
    bool isParallel = (meshObject_->getMpiSize() > 1);
    
    for (const auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        if (leftCellID < 0 || rightCellID < 0) continue;
        if (leftCellID >= meshObject_->getNumCells() || rightCellID >= meshObject_->getNumCells()) continue;
        
        // Skip if not owned by this rank
        if (isParallel && !meshObject_->isLocalCell(leftCellID)) {
            continue;
        }
        
        int localLeftID = isParallel ? meshObject_->globalToLocal(leftCellID) : leftCellID;
        int localRightID = isParallel ? meshObject_->globalToLocal(rightCellID) : rightCellID;
        
        double massFlux = face->getMassFlux();
        double mdot_L = (massFlux + std::abs(massFlux)) / 2.0;  // Upwind for left
        double mdot_R = (massFlux - std::abs(massFlux)) / 2.0;  // Upwind for right
        
        for (int i = 0; i < MAX_DIM; ++i) {
            double fieldLeft = (localLeftID >= 0) ? data[localLeftID * MAX_DIM + i] : 0.0;
            double fieldRight = (localRightID >= 0) ? data[localRightID * MAX_DIM + i] : 0.0;
            double rhs = fieldLeft * mdot_L + fieldRight * mdot_R;
            
            int globalRowLeft = leftCellID * MAX_DIM + i;
            int globalRowRight = rightCellID * MAX_DIM + i;
            
            linearSystem_->addRhs(globalRowLeft, rhs);
            linearSystem_->addRhs(globalRowRight, -rhs);
            linearSystem_->addLhs(globalRowLeft, globalRowLeft, -mdot_L);
            linearSystem_->addLhs(globalRowLeft, globalRowRight, -mdot_R);
            linearSystem_->addLhs(globalRowRight, globalRowLeft, mdot_L);
            linearSystem_->addLhs(globalRowRight, globalRowRight, mdot_R);
        }
    }
}

AssembleInteriorPressurePoissonSystem::AssembleInteriorPressurePoissonSystem(
    const std::string& name, FieldArray* fieldsHolder, LinearSystem* linearSystem,
    MeshObject* meshObject, double dt)
    : AssembleSystemBase(name, fieldsHolder, linearSystem, meshObject), dt_(dt) {
    
    if (fieldsHolder->getType() != DimType::SCALAR) {
        throw std::runtime_error("Pressure Poisson requires a scalar field");
    }
}

void AssembleInteriorPressurePoissonSystem::assemble() {
    const auto& internalFaces = meshObject_->getInternalFaces();
    bool isParallel = (meshObject_->getMpiSize() > 1);
    
    for (const auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        if (leftCellID < 0 || rightCellID < 0) continue;
        if (leftCellID >= meshObject_->getNumCells() || rightCellID >= meshObject_->getNumCells()) continue;
        
        // Skip if not owned by this rank
        if (isParallel && !meshObject_->isLocalCell(leftCellID)) {
            continue;
        }
        
        const Cell* leftCell = meshObject_->getCellByFlatId(leftCellID);
        const Cell* rightCell = meshObject_->getCellByFlatId(rightCellID);
        if (!leftCell || !rightCell) continue;
        
        double faceArea = face->getArea();
        auto normal = face->getNormalVector();
        auto leftCent = leftCell->getCentroid();
        auto rightCent = rightCell->getCentroid();
        
        double distX = rightCent[0] - leftCent[0];
        double distY = rightCent[1] - leftCent[1];
        double normalLength = normal[0] * distX + normal[1] * distY;
        
        double lhsFactor = dt_ * faceArea / normalLength;
        double massFlux = face->getMassFlux();
        
        linearSystem_->addRhs(leftCellID, massFlux);
        linearSystem_->addRhs(rightCellID, -massFlux);
        linearSystem_->addLhs(leftCellID, rightCellID, lhsFactor);
        linearSystem_->addLhs(leftCellID, leftCellID, -lhsFactor);
        linearSystem_->addLhs(rightCellID, rightCellID, -lhsFactor);
        linearSystem_->addLhs(rightCellID, leftCellID, lhsFactor);
    }
}

void AssembleInteriorPressurePoissonSystem::assembleRhs() {
    const auto& internalFaces = meshObject_->getInternalFaces();
    bool isParallel = (meshObject_->getMpiSize() > 1);
    
    for (const auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        double massFlux = face->getMassFlux();

        if (leftCellID < 0 || rightCellID < 0) continue;
        if (leftCellID >= meshObject_->getNumCells() || rightCellID >= meshObject_->getNumCells()) continue;
        if (isParallel && !meshObject_->isLocalCell(leftCellID)) {
            continue;
        }
        
        linearSystem_->addRhs(leftCellID, massFlux);
        linearSystem_->addRhs(rightCellID, -massFlux);
    }
}

} // namespace Vinci4D
