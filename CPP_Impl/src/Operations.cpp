#include "Operations.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <mpi.h>

namespace Vinci4D {

namespace {
bool isRootRank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}
}

void PerformanceTimer::startTimer(const std::string& eventName) {
    eventName_ = eventName;
    started_ = true;
    startTime_ = std::chrono::high_resolution_clock::now();
}

void PerformanceTimer::endTimer() {
    if (!started_) {
        if (isRootRank()) {
            std::cerr << "Warning: Timer was not started" << std::endl;
        }
        return;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime_);
    double seconds = duration.count() / 1e6;
    
    if (isRootRank()) {
        std::cout << "For event " << eventName_ << ", timer: " << seconds << " seconds" << std::endl;
    }
    eventTimings_[eventName_].push_back(seconds);
    
    started_ = false;
    eventName_ = "";
}

void PerformanceTimer::report_timing_statistics() const {
    if (!isRootRank()) {
        return;
    }
    std::cout << "\n=== Timing Statistics ===" << std::endl;
    for (const auto& pair : eventTimings_) {
        const auto& times = pair.second;
        double total = std::accumulate(times.begin(), times.end(), 0.0);
        double average = total / times.size();
        std::cout << "Event: " << pair.first << ", Average Time: " << average << " seconds over " << times.size() << " events." << std::endl;
    }
}

void LogObject::reportLog(int nonlinIterCounter) {
    if (!isRootRank()) {
        return;
    }
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
    // Exchange ghost cells before computing face-based quantities
    velocityFieldArray_->exchangeGhostCells(*meshObject_);
    pressureFieldArray_->exchangeGhostCells(*meshObject_);
    cellPressureGradientFieldArray_->exchangeGhostCells(*meshObject_);
    
    const auto& internalFaces = meshObject_->getInternalFaces();
    auto& velocityData = velocityFieldArray_->getData();
    auto& pressureData = pressureFieldArray_->getData();
    auto& gradPressureData = cellPressureGradientFieldArray_->getData();
    
    int numCells = meshObject_->getNumCells();
    bool isParallel = (meshObject_->getMpiSize() > 1);
    
    for (auto* face : internalFaces) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        // Validate cell IDs
        if (leftCellID < 0 || leftCellID >= numCells || rightCellID < 0 || rightCellID >= numCells) {
            throw std::runtime_error("Invalid cell ID in internal face");
        }

        // Only compute faces owned by this rank in parallel
        if (isParallel && !meshObject_->isLocalCell(leftCellID)) {
            continue;
        }

        int localLeftID = isParallel ? meshObject_->globalToLocal(leftCellID) : leftCellID;
        int localRightID = isParallel ? meshObject_->globalToLocal(rightCellID) : rightCellID;
        if (localLeftID < 0 || localRightID < 0) {
            continue;
        }
        
        // Get velocity values
        double leftVelX = velocityData[localLeftID * MAX_DIM + 0];
        double leftVelY = velocityData[localLeftID * MAX_DIM + 1];
        double rightVelX = velocityData[localRightID * MAX_DIM + 0];
        double rightVelY = velocityData[localRightID * MAX_DIM + 1];
        
        // Get pressure gradient values
        double leftGradPX = gradPressureData[localLeftID * MAX_DIM + 0];
        double leftGradPY = gradPressureData[localLeftID * MAX_DIM + 1];
        double rightGradPX = gradPressureData[localRightID * MAX_DIM + 0];
        double rightGradPY = gradPressureData[localRightID * MAX_DIM + 1];

        // Get pressure values
        double leftPressure = pressureData[localLeftID];
        double rightPressure = pressureData[localRightID];
        
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

void ComputeCellGradient::computeScalarGradient() {
    // Exchange ghost cells before computing gradients
    inField_->exchangeGhostCells(*meshObject_);
    
    outField_->initializeConstant(0.0);
    
    int numComponents = outField_->getNumComponents();
    auto& inData = inField_->getData();
    auto& outData = outField_->getData();
    
    bool isParallel = (meshObject_->getMpiSize() > 1);

    // Interior faces contribution
    for (const auto* face : meshObject_->getInternalFaces()) {
        int leftCellID = face->getLeftCell();
        int rightCellID = face->getRightCell();
        
        if (leftCellID < 0 || rightCellID < 0) continue;
        if (leftCellID >= meshObject_->getNumCells() || rightCellID >= meshObject_->getNumCells()) continue;
        
        int localLeftID = isParallel ? meshObject_->globalToLocal(leftCellID) : leftCellID;
        int localRightID = isParallel ? meshObject_->globalToLocal(rightCellID) : rightCellID;
        if (localLeftID < 0 || localRightID < 0) {
            continue;
        }

        double leftValue = inData[localLeftID];
        double rightValue = inData[localRightID];
        double faceValue = 0.5 * (leftValue + rightValue);
        
        auto normal = face->getNormalVector();
        double area = face->getArea();
        double areaX = normal[0] * area;
        double areaY = normal[1] * area;
        
        // Left cell contribution
        if (!isParallel || meshObject_->isLocalCell(leftCellID)) {
            outData[localLeftID * numComponents + 0] += faceValue * areaX;
            outData[localLeftID * numComponents + 1] += faceValue * areaY;
        }
        
        // Right cell contribution (opposite normal)
        if (!isParallel || meshObject_->isLocalCell(rightCellID)) {
            outData[localRightID * numComponents + 0] -= faceValue * areaX;
            outData[localRightID * numComponents + 1] -= faceValue * areaY;
        }
    }
    
    // Boundary faces contribution
    for (const auto* face : meshObject_->getBoundaryFaces()) {
        int cellID = face->getLeftCell();
        if (cellID < 0 || cellID >= meshObject_->getNumCells()) continue;
        if (isParallel && !meshObject_->isLocalCell(cellID)) continue;

        int localCellID = isParallel ? meshObject_->globalToLocal(cellID) : cellID;
        if (localCellID < 0) continue;
        
        double faceValue = inData[localCellID];
        auto normal = face->getNormalVector();
        double area = face->getArea();
        
        outData[localCellID * numComponents + 0] += faceValue * normal[0] * area;
        outData[localCellID * numComponents + 1] += faceValue * normal[1] * area;
    }
    
    // Scale by cell volume
    const auto& cells = isParallel ? meshObject_->getLocalCells() : meshObject_->getCells();
    for (const auto& cell : cells) {
        int localCellID = isParallel ? cell.getLocalId() : cell.getFlatId();
        double cellVolume = cell.getVolume();
        
        outData[localCellID * numComponents + 0] /= cellVolume;
        outData[localCellID * numComponents + 1] /= cellVolume;
    }
}

} // namespace Vinci4D
