#include "FieldsHolder.h"
#include <stdexcept>
#include <algorithm>

namespace Vinci4D {

std::string fieldNameToString(FieldNames name) {
    switch(name) {
        case FieldNames::PRESSURE: return "pressure";
        case FieldNames::GRAD_PRESSURE: return "grad_pressure";
        case FieldNames::VELOCITY_NEW: return "velocity_np1";
        case FieldNames::VELOCITY_OLD: return "velocity_n";
        case FieldNames::VELOCITY_VERY_OLD: return "velocity_nm1";
        case FieldNames::MASS_FLUX_FACE: return "mass_flux_face";
        case FieldNames::PRESSURE_CORRECTION: return "pressure_correction";
        case FieldNames::GRAD_PRESSURE_CORRECTION: return "grad_pressure_correction";
        default: return "unknown";
    }
}

FieldArray::FieldArray(const std::string& name, DimType fieldType, int numPoints)
    : name_(name),
      fieldType_(fieldType),
      numComponents_(fieldType == DimType::SCALAR ? 1 : MAX_DIM) {
    data_.resize(numPoints * numComponents_, 0.0);
}

void FieldArray::initializeConstant(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void FieldArray::increment(double value, double scale) {
    for (auto& val : data_) {
        val += value * scale;
    }
}

void FieldArray::copyTo(FieldArray& other) const {
    if (fieldType_ != other.fieldType_) {
        throw std::runtime_error("Can only copy fields of the same type");
    }
    if (data_.size() != other.data_.size()) {
        throw std::runtime_error("Can only copy fields with the same shape");
    }
    other.data_ = data_;
}

void FieldArray::swapFields(FieldArray& other) {
    if (fieldType_ != other.fieldType_) {
        throw std::runtime_error("Can only swap fields of the same type");
    }
    if (data_.size() != other.data_.size()) {
        throw std::runtime_error("Can only swap fields with the same shape");
    }
    data_.swap(other.data_);
}

} // namespace Vinci4D
