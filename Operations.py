import numpy as np
import FieldsHolder as FH
from MeshObject import MeshObject

class ComputeCellGradient:
    """Class to compute cell gradients for a given field array"""
    def __init__(self, mesh_object, in_field_array, out_field_array):
        self.mesh_object_ : MeshObject = mesh_object
        self.field_array_ : FH.FieldArray = in_field_array
        self.out_field_array_ : FH.FieldArray = out_field_array
    
    def compute_scalar_gradient(self):
        """Compute the gradient of the field array for each cell in the mesh"""
        self.out_field_array_.initialize_constant(0.0)  # Initialize output array to zero
        numComponents = self.out_field_array_.get_num_components()
        assert numComponents == 2, "Output field array must be a vector field with 2 components for 2D gradient of a scalar"
        for face in self.mesh_object_.get_internal_faces():
            left_cell_id = face.get_left_cell()
            right_cell_id = face.get_right_cell()
            
            # Get the field values for the left and right cells
            left_value = self.field_array_.get_data()[left_cell_id]
            right_value = self.field_array_.get_data()[right_cell_id]
            
            # Compute the gradient contribution for this face
            faceValue = 0.5*(right_value + left_value)
            
            # Add the contribution to the output field array for both cells
            areaVector = (face.get_normal_vector()[0]*face.get_area(), face.get_normal_vector()[1]*face.get_area())
            self.out_field_array_.get_data()[left_cell_id*numComponents + 0] += faceValue*areaVector[0]  # x component contribution for left cell
            self.out_field_array_.get_data()[left_cell_id*numComponents + 1] += faceValue*areaVector[1]  # y component contribution for left cell
            self.out_field_array_.get_data()[right_cell_id*numComponents + 0] -= faceValue*areaVector[0]  # Opposite contribution for right cell, inverted normal
            self.out_field_array_.get_data()[right_cell_id*numComponents + 1] -= faceValue*areaVector[1]  # Opposite contribution for right cell, inverted normal

        for face in self.mesh_object_.get_boundary_faces():
            directionFactor = 1.0
            if (face.get_left_cell() is not None):
                cellID = face.get_left_cell()  # For boundary faces, only one cell contributes
            else:
                cellID = face.get_right_cell()
                directionFactor = -1.0
            faceValue = self.field_array_.get_data()[cellID]  # For boundary faces, use the value from the single adjacent cell


            # Add the contribution to the output field array for both cells
            areaVector = (face.get_normal_vector()[0]*face.get_area(), face.get_normal_vector()[1]*face.get_area())
            self.out_field_array_.get_data()[cellID*numComponents + 0] += directionFactor*faceValue*areaVector[0]  # x component contribution for left cell
            self.out_field_array_.get_data()[cellID*numComponents + 1] += directionFactor*faceValue*areaVector[1]  # y component contribution for left cell
        
        # scale cell output by volume of the cell to get average gradient
        for cell in self.mesh_object_.get_cells():
            cell_id = cell.get_flat_id()
            cell_volume = cell.get_volume()
            self.out_field_array_.get_data()[cell_id*numComponents + 0] /= cell_volume
            self.out_field_array_.get_data()[cell_id*numComponents + 1] /= cell_volume

