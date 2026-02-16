#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "FieldsHolder.h"
#include "MeshObject.h"
#include "LinearSystem.h"
#include <chrono>
#include <string>
#include <map>

namespace Vinci4D {

/**
 * @brief Performance timer for timing events
 */
class PerformanceTimer {
public:
    PerformanceTimer() : eventName_(""), started_(false) {}
    
    void startTimer(const std::string& eventName);
    void endTimer();
    void report_timing_statistics() const;

private:
    std::string eventName_;
    std::map<std::string, std::vector<double>> eventTimings_;
    bool started_;
    std::chrono::high_resolution_clock::time_point startTime_;
};

/**
 * @brief Log object for tracking residuals
 */
class LogObject {
public:
    explicit LogObject(const std::vector<LinearSystem*>& systemVector)
        : systemVector_(systemVector) {}
    
    void reportLog(int nonlinIterCounter);
    std::map<std::string, double> getResiduals();

private:
    std::vector<LinearSystem*> systemVector_;
    std::map<std::string, double> nonlinearResiduals_;
};

/**
 * @brief Compute time step based on CFL condition
 */
class CFLTimeStepCompute {
public:
    CFLTimeStepCompute(MeshObject* meshObject, double CFL)
        : meshObject_(meshObject), CFL_(CFL) {}
    
    double computeTimeStep();

private:
    MeshObject* meshObject_;
    double CFL_;
};

/**
 * @brief Compute mass flux across interior faces
 */
class ComputeInteriorMassFlux {
public:
    ComputeInteriorMassFlux(MeshObject* meshObject, FieldArray* velocityField,
                           FieldArray* pressureField, 
                           FieldArray* cellPressureGradientField, double dt);
    
    void computeMassFlux();
    void correctMassFlux(FieldArray* dP);

private:
    MeshObject* meshObject_;
    FieldArray* velocityFieldArray_;
    FieldArray* pressureFieldArray_;
    FieldArray* cellPressureGradientFieldArray_;
    double dt_;
    double scaling_;
};

/**
 * @brief Compute cell gradients for scalar or vector fields
 */
class ComputeCellGradient {
public:
    /**
     * @brief Constructor for gradient computation
     * @param meshObject Mesh for face iteration
     * @param inField Input scalar field
     * @param outField Output vector field (gradient)
     */
    ComputeCellGradient(MeshObject* meshObject, FieldArray* inField, FieldArray* outField)
        : meshObject_(meshObject), inField_(inField), outField_(outField) {}
    
    /**
     * @brief Compute the gradient of a scalar field at cell centers
     */
    void computeScalarGradient();

private:
    MeshObject* meshObject_;
    FieldArray* inField_;
    FieldArray* outField_;
};

} // namespace Vinci4D

#endif // OPERATIONS_H
