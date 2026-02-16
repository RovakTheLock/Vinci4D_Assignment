#ifndef FIELDS_HOLDER_H
#define FIELDS_HOLDER_H

#include <vector>
#include <string>

namespace Vinci4D {

constexpr int MAX_DIM = 2;

enum class DimType {
    SCALAR = 0,
    VECTOR = 1
};

enum class FieldNames {
    PRESSURE,
    GRAD_PRESSURE,
    VELOCITY_NEW,
    VELOCITY_OLD,
    VELOCITY_VERY_OLD,
    MASS_FLUX_FACE,
    PRESSURE_CORRECTION,
    GRAD_PRESSURE_CORRECTION
};

/**
 * @brief Convert FieldNames enum to string
 */
std::string fieldNameToString(FieldNames name);

/**
 * @brief Field array that stores scalar or vector field data
 * 
 * Supports ghost cells for MPI parallel execution.
 * Data layout: [local cells...][ghost cells...]
 */
class FieldArray {
public:
    /**
     * @brief Construct a FieldArray
     * 
     * @param name Name of the field
     * @param fieldType Type of the field (SCALAR or VECTOR)
     * @param numLocalPoints Number of local points (cells or faces) owned by this rank
     * @param numGhostPoints Number of ghost points (default 0 for sequential)
     */
    FieldArray(const std::string& name, DimType fieldType, int numLocalPoints, int numGhostPoints = 0);
    
    // Getters
    std::string getName() const { return name_; }
    DimType getType() const { return fieldType_; }
    int getNumComponents() const { return numComponents_; }
    int getNumLocalPoints() const { return numLocalPoints_; }
    int getNumGhostPoints() const { return numGhostPoints_; }
    std::vector<double>& getData() { return data_; }
    const std::vector<double>& getData() const { return data_; }
    
    /**
     * @brief Initialize the field with a constant value
     */
    void initializeConstant(double value);
    
    /**
     * @brief Increment the field by another field array scaled by a factor
     * 
     * @param other The FieldArray to add to this field
     * @param scale Scaling factor to apply to the other field (default 1.0)
     */
    void increment(const FieldArray& other, double scale = 1.0);
    
    /**
     * @brief Copy data to another FieldArray
     */
    void copyTo(FieldArray& other) const;
    
    /**
     * @brief Swap data arrays with another FieldArray
     */
    void swapFields(FieldArray& other);
    
    /**
     * @brief Exchange ghost cell data with neighbor ranks
     * 
     * @param mesh MeshObject containing communication patterns
     */
    void exchangeGhostCells(const class MeshObject& mesh);

private:
    std::string name_;
    DimType fieldType_;
    int numComponents_;
    int numLocalPoints_;
    int numGhostPoints_;
    std::vector<double> data_;
};

} // namespace Vinci4D

#endif // FIELDS_HOLDER_H
