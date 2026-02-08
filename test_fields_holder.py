import unittest
import numpy as np
from FieldsHolder import FieldArray, DimType, FieldNames


class TestFieldArray(unittest.TestCase):
    
    def test_scalar_array_pressure(self):
        """Test scalar field array for pressure"""
        num_cells = 100
        pressure = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, num_cells)
        
        # Check name
        self.assertEqual(pressure.get_name(), FieldNames.PRESSURE.value)
        
        # Check type is "Scalar"
        self.assertEqual(pressure.get_type(), "Scalar")
        
        # Check number of components is 1
        self.assertEqual(pressure.get_num_components(), 1)
        
        # Check all data values are 0
        self.assertEqual(len(pressure.get_data()), num_cells)
        np.testing.assert_array_equal(pressure.get_data(), np.zeros(num_cells))

    def test_scalar_array_face_flux(self):
        """Test scalar field array for face flux"""
        num_faces = 500
        face_flux = FieldArray(FieldNames.FACE_FLUX.value, DimType.SCALAR, num_faces)
        
        # Check name
        self.assertEqual(face_flux.get_name(), FieldNames.FACE_FLUX.value)
        
        # Check type is "Scalar"
        self.assertEqual(face_flux.get_type(), "Scalar")
        
        # Check number of components is 1
        self.assertEqual(face_flux.get_num_components(), 1)
        
        # Check all data values are 0
        self.assertEqual(len(face_flux.get_data()), num_faces)
        np.testing.assert_array_equal(face_flux.get_data(), np.zeros(num_faces))
    
    def test_vector_array_velocity(self):
        """Test vector field array for velocity"""
        num_cells = 100
        velocity = FieldArray(FieldNames.VELOCITY.value, DimType.VECTOR, num_cells)
        
        # Check name
        self.assertEqual(velocity.get_name(), FieldNames.VELOCITY.value)
        
        # Check type is "Vector"
        self.assertEqual(velocity.get_type(), "Vector")
        
        # Check number of components is 2
        self.assertEqual(velocity.get_num_components(), 2)
        
        # Check all data values are 0 (should have 2*num_cells elements)
        self.assertEqual(len(velocity.get_data()), num_cells * 2)
        np.testing.assert_array_equal(velocity.get_data(), np.zeros(num_cells * 2))
    
    def test_initialize_constant_pressure(self):
        """Test initialize_constant method for pressure field"""
        num_cells = 50
        pressure = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, num_cells)
        
        # Initialize all pressure values to 2.0
        constant_value = 2.0
        pressure.initialize_constant(constant_value)
        
        # Verify all values are 2.0
        expected = np.full(num_cells, constant_value)
        np.testing.assert_array_equal(pressure.get_data(), expected)


if __name__ == '__main__':
    unittest.main()
