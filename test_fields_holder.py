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
        self.assertEqual(pressure.get_type(), DimType.SCALAR)
        
        # Check number of components is 1
        self.assertEqual(pressure.get_num_components(), 1)
        
        # Check all data values are 0
        self.assertEqual(len(pressure.get_data()), num_cells)
        np.testing.assert_array_equal(pressure.get_data(), np.zeros(num_cells))

    def test_scalar_array_face_flux(self):
        """Test scalar field array for face flux"""
        num_faces = 500
        face_flux = FieldArray(FieldNames.MASS_FLUX_FACE.value, DimType.SCALAR, num_faces)
        
        # Check name
        self.assertEqual(face_flux.get_name(), FieldNames.MASS_FLUX_FACE.value)
        
        # Check type is "Scalar"
        self.assertEqual(face_flux.get_type(), DimType.SCALAR)
        
        # Check number of components is 1
        self.assertEqual(face_flux.get_num_components(), 1)
        
        # Check all data values are 0
        self.assertEqual(len(face_flux.get_data()), num_faces)
        np.testing.assert_array_equal(face_flux.get_data(), np.zeros(num_faces))
    
    def test_vector_array_velocity(self):
        """Test vector field array for velocity"""
        num_cells = 100
        velocity = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, num_cells)
        
        # Check name
        self.assertEqual(velocity.get_name(), FieldNames.VELOCITY_NEW.value)
        
        # Check type is "Vector"
        self.assertEqual(velocity.get_type(), DimType.VECTOR)
        
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


    def test_vector_array_velocity_swap(self):
        """Test swap_fields method for velocity field"""
        num_cells = 100
        velocity_new = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, num_cells)
        velocity_old = FieldArray(FieldNames.VELOCITY_OLD.value, DimType.VECTOR, num_cells)
        
        # Initialize velocity_new with some values
        velocity_new.initialize_constant(1.0)
        
        # Initialize velocity_old with different values
        velocity_old.initialize_constant(2.0)
        
        # Swap fields
        velocity_new.swap_fields(velocity_old)
        
        # Verify that the data has been swapped correctly
        expected_new = np.full(num_cells * 2, 2.0)  # Should now be 2.0 (old values of velocity_old)
        expected_old = np.full(num_cells * 2, 1.0)  # Should now be 1.0 (old values of velocity_new)
        
        np.testing.assert_array_equal(velocity_new.get_data(), expected_new)
        np.testing.assert_array_equal(velocity_old.get_data(), expected_old)

    def test_vector_array_velocity_increment(self):
        """Test increment method for velocity field"""
        num_cells = 100
        velocity = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, num_cells)
        
        # Initialize velocity with some values
        velocity.initialize_constant(1.0)
        
        # Increment by a constant value
        increment_value = 2.0
        velocity.increment(increment_value, scale=1.0)
        
        # Verify all values are now 3.0 (1.0 + 2.0)
        expected = np.full(num_cells * 2, 3.0)
        np.testing.assert_array_equal(velocity.get_data(), expected)
    def test_vector_array_copy_to(self):
        """Test copy_to method for velocity field"""
        num_cells = 100
        expectedValue = 123.456
        velocity_source = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, num_cells)
        velocity_dest = FieldArray(FieldNames.VELOCITY_OLD.value, DimType.VECTOR, num_cells)
        
        # Initialize source velocity with some values
        velocity_source.initialize_constant(expectedValue)
        
        # Copy to destination
        velocity_source.copy_to(velocity_dest)
        
        # Verify that the data has been copied correctly
        expected = np.full(num_cells * 2, expectedValue)  # Should be the same as source
        np.testing.assert_array_equal(velocity_dest.get_data(), expected)



if __name__ == '__main__':
    unittest.main()
