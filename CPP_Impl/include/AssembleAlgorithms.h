#ifndef ASSEMBLE_ALGORITHMS_H
#define ASSEMBLE_ALGORITHMS_H

#include "LinearSystem.h"
#include "FieldsHolder.h"
#include "MeshObject.h"
#include <memory>
#include <array>

namespace Vinci4D {

/**
 * @brief Boundary types for Dirichlet boundary conditions
 */
enum class Boundary {
    LEFT = 0,
    RIGHT = 1,
    TOP = 2,
    BOTTOM = 3
};

/**
 * @brief Base class for assembly algorithms
 */
class AssembleSystemBase {
public:
    AssembleSystemBase(const std::string& name, FieldArray* fieldsHolder,
                      LinearSystem* linearSystem, MeshObject* meshObject)
        : name_(name), fieldsHolder_(fieldsHolder),
          linearSystem_(linearSystem), meshObject_(meshObject) {}
    
    virtual ~AssembleSystemBase() = default;
    
    void zero() { linearSystem_->zero(); }
    void zeroRhs() { linearSystem_->zeroRhs(); }
    
    virtual void assemble() = 0;
    virtual void assembleRhs() { assemble(); }
    
    std::string getName() const { return name_; }

protected:
    std::string name_;
    FieldArray* fieldsHolder_;
    LinearSystem* linearSystem_;
    MeshObject* meshObject_;
};

/**
 * @brief Assemble time derivative term for vector field
 */
class AssembleCellVectorTimeTerm : public AssembleSystemBase {
public:
    AssembleCellVectorTimeTerm(const std::string& name, FieldArray* fieldsHolderNew,
                               FieldArray* fieldsHolderOld, LinearSystem* linearSystem,
                               MeshObject* meshObject, double dt);
    
    void assemble() override;

private:
    double dt_;
    FieldArray* fieldHolderOld_;
};

/**
 * @brief Assemble interior diffusion term for vector field
 */
class AssembleInteriorVectorDiffusionToLinSystem : public AssembleSystemBase {
public:
    AssembleInteriorVectorDiffusionToLinSystem(const std::string& name, FieldArray* fieldsHolder,
                                               LinearSystem* linearSystem, MeshObject* meshObject,
                                               double diffusionCoeff = 1.0);
    
    void assemble() override;

private:
    double diffusionCoeff_;
};

/**
 * @brief Assemble Dirichlet boundary diffusion for vector field
 */
class AssembleDirichletBoundaryVectorDiffusionToLinSystem : public AssembleSystemBase {
public:
    AssembleDirichletBoundaryVectorDiffusionToLinSystem(
        const std::string& name, FieldArray* fieldsHolder, LinearSystem* linearSystem,
        MeshObject* meshObject, Boundary boundaryType,
        const std::array<double, 2>& boundaryValue, double diffusionCoeff = 1.0);
    
    void assemble() override;

private:
    Boundary boundaryType_;
    std::array<double, 2> boundaryValue_;
    double diffusionCoeff_;
    std::vector<Face*> boundaryFaces_;
};

/**
 * @brief Assemble interior advection term for vector field
 */
class AssembleInteriorVectorAdvectionToLinSystem : public AssembleSystemBase {
public:
    AssembleInteriorVectorAdvectionToLinSystem(const std::string& name, FieldArray* fieldsHolder,
                                               LinearSystem* linearSystem, MeshObject* meshObject);
    
    void assemble() override;
};

/**
 * @brief Assemble pressure gradient contribution
 */
class AssembleCellVectorPressureGradientToLinSystem : public AssembleSystemBase {
public:
    AssembleCellVectorPressureGradientToLinSystem(const std::string& name, FieldArray* fieldsHolder,
                                                   LinearSystem* linearSystem, MeshObject* meshObject);
    
    void assemble() override;
};

/**
 * @brief Assemble interior pressure Poisson system
 */
class AssembleInteriorPressurePoissonSystem : public AssembleSystemBase {
public:
    AssembleInteriorPressurePoissonSystem(const std::string& name, FieldArray* fieldsHolder,
                                          LinearSystem* linearSystem, MeshObject* meshObject, double dt);
    
    void assemble() override;
    void assembleRhs() override;

private:
    double dt_;
};

} // namespace Vinci4D

#endif // ASSEMBLE_ALGORITHMS_H
