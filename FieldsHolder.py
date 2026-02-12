import numpy as np
from enum import Enum


MAX_DIM=2
class DimType(Enum):
    SCALAR = 0
    VECTOR = 1

class FieldNames(Enum):
    PRESSURE = "pressure"
    GRAD_PRESSURE = "grad_pressure"
    VELOCITY_NEW = "velocity_np1"
    VELOCITY_OLD = "velocity_n"
    VELOCITY_VERY_OLD = "velocity_nm1"
    MASS_FLUX_FACE = "mass_flux_face"

class FieldArray:
    """
    Creates a field array based on type enum type SCALAR (pressure) or VECTOR (velocity)
    """
    
    def __init__(self, name, fieldType, num_points):
        """
        Initialize field with name and type of field
        
        Args:
            name (str): Name of the field
            fieldType (DimType): Type of the field (SCALAR or VECTOR)
        """
        
        self.name_ = name
        self.fieldType_ = fieldType
        self.numComponents_ = 1 if fieldType == DimType.SCALAR else MAX_DIM  # Assuming 2D vector fields for VECTOR type
        self.data_ = np.zeros(num_points * self.numComponents_)

    def get_name(self):
        """Get the name of the field"""
        return self.name_
    
    def get_type(self):
        """Get the type of the field as a string ('Scalar' or 'Vector')"""
        return self.fieldType_
    
    def get_num_components(self):
        """Get the number of components in the field (1 for scalar, 2 for vector)"""
        return self.numComponents_
    
    def get_data(self):
        """Get the underlying data array"""
        return self.data_
    
    def initialize_constant(self,value):
        """Initialize the field data with a constant value"""
        self.data_.fill(value)

    def increment(self, value, scale=1.0):
        """Increment the field data by a constant value"""
        self.data_ += value*scale
    
    def copy_to(self, other):
        """Copy the data to another FieldArray (must be the same type and shape)"""
        assert self.fieldType_ == other.fieldType_, "Can only copy fields of the same type"
        assert self.data_.shape == other.data_.shape, "Can only copy fields with the same shape"
        np.copyto(other.data_, self.data_)

    def swap_fields(self, other):
        """Swap the data arrays with another FieldArray (useful for time-stepping)"""
        assert self.fieldType_ == other.fieldType_, "Can only swap fields of the same type"
        assert self.data_.shape == other.data_.shape, "Can only swap fields with the same shape"
        self.data_, other.data_ = other.data_, self.data_
    
    def __repr__(self):
        return f"FieldArray(name='{self.name_}', type='{self.get_type()}', num_components={self.numComponents_})\nData: {self.data_}"
    