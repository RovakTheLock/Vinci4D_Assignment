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
 */
class FieldArray {
public:
    /**
     * @brief Construct a FieldArray
     * 
     * @param name Name of the field
     * @param fieldType Type of the field (SCALAR or VECTOR)
     * @param numPoints Number of points (cells or faces)
     */
    FieldArray(const std::string& name, DimType fieldType, int numPoints);
    
    // Getters
    std::string getName() const { return name_; }
    DimType getType() const { return fieldType_; }
    int getNumComponents() const { return numComponents_; }
    std::vector<double>& getData() { return data_; }
    const std::vector<double>& getData() const { return data_; }
    
    /**
     * @brief Initialize the field with a constant value
     */
    void initializeConstant(double value);
    
    /**
     * @brief Increment the field by a scaled value
     */
    void increment(double value, double scale = 1.0);
    
    /**
     * @brief Copy data to another FieldArray
     */
    void copyTo(FieldArray& other) const;
    
    /**
     * @brief Swap data arrays with another FieldArray
     */
    void swapFields(FieldArray& other);

private:
    std::string name_;
    DimType fieldType_;
    int numComponents_;
    std::vector<double> data_;
};

} // namespace Vinci4D

#endif // FIELDS_HOLDER_H
