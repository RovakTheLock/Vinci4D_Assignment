#include "Operations.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Vinci4D {

void PerformanceTimer::startTimer(const std::string& eventName) {
    eventName_ = eventName;
    started_ = true;
    startTime_ = std::chrono::high_resolution_clock::now();
}

void PerformanceTimer::endTimer() {
    if (!started_) {
        std::cerr << "Warning: Timer was not started" << std::endl;
        return;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime_);
    double seconds = duration.count() / 1e6;
    
    std::cout << "For event " << eventName_ << ", timer: " << seconds << " seconds" << std::endl;
    
    started_ = false;
    eventName_ = "";
}

void LogObject::reportLog(int nonlinIterCounter) {
    std::cout << "For nonlinear iteration = " << nonlinIterCounter << std::endl;
    
    for (auto* system : systemVector_) {
        double residual = system->getRhsNorm();
        std::cout << "\tFor system " << system->getName() << ": RHS residual = " << residual << std::endl;
        nonlinearResiduals_[system->getName()] = residual;
    }
}

std::map<std::string, double> LogObject::getResiduals() {
    for (auto* system : systemVector_) {
        double residual = system->getRhsNorm();
        nonlinearResiduals_[system->getName()] = residual;
    }
    return nonlinearResiduals_;
}

double CFLTimeStepCompute::computeTimeStep() {
    auto xRange = meshObject_->getXRange();
    auto yRange = meshObject_->getYRange();
    auto cellCount = meshObject_->getCellCount();
    
    double minDx = (xRange[1] - xRange[0]) / cellCount[0];
    double minDy = (yRange[1] - yRange[0]) / cellCount[1];
    double minCellSpacing = std::min(minDx, minDy);
    
    // Assume unit velocity for now
    double maxVelocity = 1.0;
    
    return CFL_ * minCellSpacing / maxVelocity;
}

ComputeInteriorMassFlux::ComputeInteriorMassFlux(
    MeshObject* meshObject, FieldArray* velocityField,
    FieldArray* pressureField, FieldArray* cellPressureGradientField, double dt)
    : meshObject_(meshObject),
      velocityFieldArray_(velocityField),
      pressureFieldArray_(pressureField),
      cellPressureGradientFieldArray_(cellPressureGradientField),
      dt_(dt),
      scaling_(dt) {}

void ComputeInteriorMassFlux::computeMassFlux() {
    const auto& internalFaces = meshObject_->getInternalFaces();
    auto& velocityData = velocityFieldArray_->getData();
    auto& pressureData = pressureFieldArray_->getData();
    auto& gradPressureData = cellPressureGradientFieldArray_->getData();
    
    int numCells = meshObject_->getNumCells();
    
    for (auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        // Validate cell IDs
        if (leftCellID < 0 || leftCellID >= numCells || rightCellID < 0 || rightCellID >= numCells) {
            throw std::runtime_error("Invalid cell ID in internal face");
        }
        
        // Get velocity values
        double leftVelX = velocityData[leftCellID * MAX_DIM + 0];
        double leftVelY = velocityData[leftCellID * MAX_DIM + 1];
        double rightVelX = velocityData[rightCellID * MAX_DIM + 0];
        double rightVelY = velocityData[rightCellID * MAX_DIM + 1];
        
        // Get pressure gradient values
        double leftGradPX = gradPressureData[leftCellID * MAX_DIM + 0];
        double leftGradPY = gradPressureData[leftCellID * MAX_DIM + 1];
        double rightGradPX = gradPressureData[rightCellID * MAX_DIM + 0];
        double rightGradPY = gradPressureData[rightCellID * MAX_DIM + 1];

        // Get pressure values
        double leftPressure = pressureData[leftCellID];
        double rightPressure = pressureData[rightCellID];
        
        // Average velocities and pressure gradients
        double avgVelX = 0.5 * (leftVelX + rightVelX);
        double avgVelY = 0.5 * (leftVelY + rightVelY);
        double avgGradPX = 0.5 * (leftGradPX + rightGradPX);
        double avgGradPY = 0.5 * (leftGradPY + rightGradPY);

        // Compute face normal distance between cell centroids
        const Cell* leftCell = meshObject_->getCellByFlatId(leftCellID);
        const Cell* rightCell = meshObject_->getCellByFlatId(rightCellID);
        if (!leftCell || !rightCell) {
            continue;
        }
        auto leftCentroid = leftCell->getCentroid();
        auto rightCentroid = rightCell->getCentroid();
        auto normal = face->getNormalVector();
        double distance = 0.0;
        for (int i = 0; i < MAX_DIM; ++i) {
            distance += (rightCentroid[i] - leftCentroid[i]) * normal[i];
        }
        if (std::abs(distance) < 1e-12) {
            continue;
        }

        // Rhie-Chow interpolation: add face pressure gradient correction
        double uDotN = avgVelX * normal[0] + avgVelY * normal[1];
        double cellGradPDotN = avgGradPX * normal[0] + avgGradPY * normal[1];
        double faceGradPDotN = (rightPressure - leftPressure) / distance;
        double massFlux = (uDotN + scaling_ * (cellGradPDotN - faceGradPDotN)) * face->getArea();
        
        // Store in face (direct access like Python version)
        face->massFlux_ = massFlux;
    }
}

void ComputeInteriorMassFlux::correctMassFlux(FieldArray* dP) {
    const auto& internalFaces = meshObject_->getInternalFaces();
    auto& dPData = dP->getData();
    
    int numCells = meshObject_->getNumCells();
    
    for (auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        // Validate cell IDs
        if (leftCellID < 0 || leftCellID >= numCells || rightCellID < 0 || rightCellID >= numCells) {
            throw std::runtime_error("Invalid cell ID in internal face");
        }
        
        // Get pressure correction values
        double leftDP = dPData[leftCellID];
        double rightDP = dPData[rightCellID];
        
        // Get cell centroids
        const Cell* leftCell = meshObject_->getCellByFlatId(leftCellID);
        const Cell* rightCell = meshObject_->getCellByFlatId(rightCellID);
        
        if (!leftCell || !rightCell) {
            throw std::runtime_error("Invalid cell ID in internal face");
        }
        
        auto leftCentroid = leftCell->getCentroid();
        auto rightCentroid = rightCell->getCentroid();
        auto normal = face->getNormalVector();
        
        // Compute distance along normal
        double distance = 0.0;
        for (int i = 0; i < MAX_DIM; ++i) {
            distance += (rightCentroid[i] - leftCentroid[i]) * normal[i];
        }
        
        // Compute pressure correction gradient
        double faceGradPressureCorrection = (rightDP - leftDP) / distance;
        
        // Correct mass flux
        face->massFlux_ -= scaling_ * faceGradPressureCorrection;
    }
}

void ComputeCellGradient::computeScalarGradient() {
    outField_->initializeConstant(0.0);
    
    int numComponents = outField_->getNumComponents();
    auto& inData = inField_->getData();
    auto& outData = outField_->getData();
    
    // Interior faces contribution
    for (const auto* face : meshObject_->getInternalFaces()) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        if (leftCellID < 0 || rightCellID < 0) continue;
        if (leftCellID >= meshObject_->getNumCells() || rightCellID >= meshObject_->getNumCells()) continue;
        
        double leftValue = inData[leftCellID];
        double rightValue = inData[rightCellID];
        double faceValue = 0.5 * (leftValue + rightValue);
        
        auto normal = face->getNormalVector();
        double area = face->getArea();
        double areaX = normal[0] * area;
        double areaY = normal[1] * area;
        
        // Left cell contribution
        outData[leftCellID * numComponents + 0] += faceValue * areaX;
        outData[leftCellID * numComponents + 1] += faceValue * areaY;
        
        // Right cell contribution (opposite normal)
        outData[rightCellID * numComponents + 0] -= faceValue * areaX;
        outData[rightCellID * numComponents + 1] -= faceValue * areaY;
    }
    
    // Boundary faces contribution
    for (const auto* face : meshObject_->getBoundaryFaces()) {
        int cellID = face->getLeftCell();
        if (cellID < 0 || cellID >= meshObject_->getNumCells()) continue;
        
        double faceValue = inData[cellID];
        auto normal = face->getNormalVector();
        double area = face->getArea();
        
        outData[cellID * numComponents + 0] += faceValue * normal[0] * area;
        outData[cellID * numComponents + 1] += faceValue * normal[1] * area;
    }
    
    // Scale by cell volume
    for (const auto& cell : meshObject_->getCells()) {
        int cellID = cell.getFlatId();
        double cellVolume = cell.getVolume();
        
        outData[cellID * numComponents + 0] /= cellVolume;
        outData[cellID * numComponents + 1] /= cellVolume;
    }
}

} // namespace Vinci4D
