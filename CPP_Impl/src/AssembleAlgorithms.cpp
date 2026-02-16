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
    const auto& cells = meshObject_->getCells();
    auto& dataNew = fieldsHolder_->getData();
    const auto& dataOld = fieldHolderOld_->getData();
    int numComponents = fieldsHolder_->getNumComponents();
    
    for (const auto& cell : cells) {
        int cellID = cell.getFlatId();
        double cellVolume = cell.getVolume();
        
        for (int comp = 0; comp < numComponents; ++comp) {
            double lhsFactor = cellVolume / dt_;
            int rowIndex = cellID * numComponents + comp;
            double value = dataNew[rowIndex] - dataOld[rowIndex];
            
            linearSystem_->addLhs(rowIndex, rowIndex, -lhsFactor);
            linearSystem_->addRhs(rowIndex, value * lhsFactor);
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
    const auto& internalFaces = meshObject_->getInternalFaces();
    auto& data = fieldsHolder_->getData();
    
    int numCells = meshObject_->getNumCells();
    
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
        
        // Assemble for each component
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            int rowLeft = leftCellID * MAX_DIM + comp;
            int rowRight = rightCellID * MAX_DIM + comp;
            
            double phiLeft = data[rowLeft];
            double phiRight = data[rowRight];
            double flux = diffusionFactor * (phiRight - phiLeft);
            
            // Contribution to left cell
            linearSystem_->addLhs(rowLeft, rowLeft, -diffusionFactor);
            linearSystem_->addLhs(rowLeft, rowRight, diffusionFactor);
            linearSystem_->addRhs(rowLeft, -flux);
            
            // Contribution to right cell
            linearSystem_->addLhs(rowRight, rowRight, -diffusionFactor);
            linearSystem_->addLhs(rowRight, rowLeft, diffusionFactor);
            linearSystem_->addRhs(rowRight, flux);
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
    const auto& cells = meshObject_->getCells();
    auto& data = fieldsHolder_->getData();
    int numComponents = fieldsHolder_->getNumComponents();
    
    for (const auto& cell : cells) {
        int cellID = cell.getFlatId();
        double cellVolume = cell.getVolume();
        
        for (int comp = 0; comp < numComponents; ++comp) {
            int rowIndex = cellID * numComponents + comp;
            double value = data[rowIndex] * cellVolume;
            linearSystem_->addRhs(rowIndex, value);
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
    
    for (auto* face : boundaryFaces_) {
        int cellID = face->getLeftCell();
        if (cellID < 0 || cellID >= meshObject_->getNumCells()) continue;
        
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
        
        for (int comp = 0; comp < MAX_DIM; ++comp) {
            int rowIdx = cellID * MAX_DIM + comp;
            double gradDotArea = (boundaryValue_[comp] - data[rowIdx]) * lhsFactor;
            linearSystem_->addRhs(rowIdx, -gradDotArea);
            linearSystem_->addLhs(rowIdx, rowIdx, -lhsFactor);
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
    const auto& internalFaces = meshObject_->getInternalFaces();
    auto& data = fieldsHolder_->getData();
    
    for (const auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        if (leftCellID < 0 || rightCellID < 0) continue;
        if (leftCellID >= meshObject_->getNumCells() || rightCellID >= meshObject_->getNumCells()) continue;
        
        double massFlux = face->getMassFlux();
        double mdot_L = (massFlux + std::abs(massFlux)) / 2.0;  // Upwind for left
        double mdot_R = (massFlux - std::abs(massFlux)) / 2.0;  // Upwind for right
        
        for (int i = 0; i < MAX_DIM; ++i) {
            double fieldLeft = data[leftCellID * MAX_DIM + i];
            double fieldRight = data[rightCellID * MAX_DIM + i];
            double rhs = fieldLeft * mdot_L + fieldRight * mdot_R;
            
            int rowIndexLeft = leftCellID * MAX_DIM + i;
            int rowIndexRight = rightCellID * MAX_DIM + i;
            
            linearSystem_->addRhs(rowIndexLeft, rhs);
            linearSystem_->addRhs(rowIndexRight, -rhs);
            linearSystem_->addLhs(rowIndexLeft, rowIndexLeft, -mdot_L);
            linearSystem_->addLhs(rowIndexLeft, rowIndexRight, -mdot_R);
            linearSystem_->addLhs(rowIndexRight, rowIndexLeft, mdot_L);
            linearSystem_->addLhs(rowIndexRight, rowIndexRight, mdot_R);
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
    
    for (const auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        if (leftCellID < 0 || rightCellID < 0) continue;
        if (leftCellID >= meshObject_->getNumCells() || rightCellID >= meshObject_->getNumCells()) continue;
        
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
    
    for (const auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        double massFlux = face->getMassFlux();
        
        linearSystem_->addRhs(leftCellID, massFlux);
        linearSystem_->addRhs(rightCellID, -massFlux);
    }
}

} // namespace Vinci4D
