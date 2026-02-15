import numpy as np
from FieldsHolder import FieldArray, MAX_DIM
from MeshObject import MeshObject
import time
from LinearSystem import LinearSystem

class PerformanceTimer:
	def __init__(self):
		self.myStartTimer_ = 0.0
		self.myEndTimer_ = 0.0
		self.myEventName_ = None
	def start_timer(self, eventName):
		self.myEventName_ = eventName
		self.myStartTimer_ = time.perf_counter()
	def end_timer(self):
		assert self.myEventName_ != None, "No event was started"
		self.myEndTimer_ = time.perf_counter()
		print(f"For event {self.myEventName_}, timer: {self.myEndTimer_ - self.myStartTimer_:.6f} seconds ")
		self.myEventName_ = None


class LogObject:
	def __init__(self,systemVector):
		self.mySystemVector_ : list[LinearSystem] = systemVector
		self.myNonlinearResiduals_ = {}
	def report_log(self, nonlinIterCounter):
		print(f"For nonlinear iteration = {nonlinIterCounter}")
		for i in range(len(self.mySystemVector_)):
			system = self.mySystemVector_[i]
			residual = np.linalg.norm(system.get_rhs())
			print(f"\tFor system {system.get_name()}: RHS residual = {residual}")
			self.myNonlinearResiduals_[system.name_] = residual
	def get_residuals(self):
		for i in range(len(self.mySystemVector_)):
			system = self.mySystemVector_[i]
			residual = np.linalg.norm(system.get_rhs())
			self.myNonlinearResiduals_[system.name_] = residual
		return self.myNonlinearResiduals_




class CFLTimeStepCompute:
    def __init__(self, mesh_object, CFL):
        self.myMeshObject_ : MeshObject = mesh_object
        self.CFL_ = CFL
    def compute_time_step(self):
        """Assume a unit velocity at top boundary for now, and uniform cell spacing in x and y"""
        xRange = self.myMeshObject_.get_x_range()
        yRange = self.myMeshObject_.get_y_range()
        numCellsX, numCellsY = self.myMeshObject_.get_cell_count()
        minDx = (xRange[1] - xRange[0])/numCellsX
        minDy = (yRange[1] - yRange[0])/numCellsY
        minCellSpacing = min(minDx, minDy)
        maxVelocity = 1.0
        min_time_step = self.CFL_ * minCellSpacing / maxVelocity
        return min_time_step

class ComputeInteriorMassFlux:
    def __init__(self, mesh_object, velocity_field_array, pressure_field_array, cell_pressure_gradient_field_array, dt):
        self.myMeshObject_ : MeshObject = mesh_object
        self.velocity_field_array_ : FieldArray = velocity_field_array
        self.pressure_field_array_ : FieldArray = pressure_field_array
        self.cell_pressure_gradient_field_array_ : FieldArray = cell_pressure_gradient_field_array # assume cell gradient has already been computed. . .
        self.dt_ = dt
        self.scaling_ = self.dt_
    def correct_mass_flux(self,dP : FieldArray):
        for face in self.myMeshObject_.get_internal_faces():
            leftCellID = face.get_left_cell()
            rightCellID = face.get_right_cell()


            # Get the pressure valuse for the left and right cells
            left_dP = dP.get_data()[leftCellID]
            right_dP = dP.get_data()[rightCellID]
            normalVector = face.get_normal_vector()
            cellLeft = self.myMeshObject_.get_cell_by_flat_id(leftCellID)
            cellRight = self.myMeshObject_.get_cell_by_flat_id(rightCellID)
            distance = 0.0
            for i in range(MAX_DIM):
                distance += (cellRight.get_centroid()[i] - cellLeft.get_centroid()[i])*normalVector[i]
            faceGradPressureCorrection_dot_n = (right_dP - left_dP)/distance
            # Store the computed mass flux in the face object (you may want to have a separate field array for this in practice)
            face.massFlux_ -= self.scaling_ * faceGradPressureCorrection_dot_n

    def compute_mass_flux(self):
        """Compute the mass flux across each face in the mesh based on the velocity field and pressure gradient"""
        for face in self.myMeshObject_.get_internal_faces():
            leftCellID = face.get_left_cell()
            rightCellID = face.get_right_cell()
            
            # Get the velocity values for the left and right cells
            leftVelocity_x = self.velocity_field_array_.get_data()[leftCellID*MAX_DIM + 0]
            leftVelocity_y = self.velocity_field_array_.get_data()[leftCellID*MAX_DIM + 1]
            rightVelocity_x = self.velocity_field_array_.get_data()[rightCellID*MAX_DIM + 0]
            rightVelocity_y = self.velocity_field_array_.get_data()[rightCellID*MAX_DIM + 1]

            # Get the pressure gradient values for the left and right cells
            leftGradPressure_x = self.cell_pressure_gradient_field_array_.get_data()[leftCellID*MAX_DIM + 0]
            leftGradPressure_y = self.cell_pressure_gradient_field_array_.get_data()[leftCellID*MAX_DIM + 1]
            rightGradPressure_x = self.cell_pressure_gradient_field_array_.get_data()[rightCellID*MAX_DIM + 0]
            rightGradPressure_y = self.cell_pressure_gradient_field_array_.get_data()[rightCellID*MAX_DIM + 1]

            # Get the pressure valuse for the left and right cells
            leftPressure = self.pressure_field_array_.get_data()[leftCellID]
            rightPressure = self.pressure_field_array_.get_data()[rightCellID]

            # Compute the mass flux contribution for this face based on rhie-chow interpolation.
            normalVector = face.get_normal_vector()
            
            # Average velocity/gradP at the face
            faceVelocity_x = 0.5*(leftVelocity_x + rightVelocity_x)
            faceVelocity_y = 0.5*(leftVelocity_y + rightVelocity_y)
            cellGradPressure_x = 0.5*(leftGradPressure_x + rightGradPressure_x)
            cellGradPressure_y = 0.5*(leftGradPressure_y + rightGradPressure_y)

            cellLeft = self.myMeshObject_.get_cell_by_flat_id(leftCellID)
            cellRight = self.myMeshObject_.get_cell_by_flat_id(rightCellID)
            distance = 0.0
            for i in range(MAX_DIM):
                distance += (cellRight.get_centroid()[i] - cellLeft.get_centroid()[i])*normalVector[i]
            faceGradPressure_dot_n = (rightPressure - leftPressure)/distance
            u_dot_n = faceVelocity_x*normalVector[0] + faceVelocity_y*normalVector[1]
            cellGradP_dot_n = cellGradPressure_x*normalVector[0] + cellGradPressure_y*normalVector[1]
            # Compute mass flux as velocity dot normal times area of the face
            mass_flux = (u_dot_n + self.scaling_ *((cellGradP_dot_n-faceGradPressure_dot_n))) * face.get_area()
            
            # Store the computed mass flux in the face object (you may want to have a separate field array for this in practice)
            face.massFlux_ = mass_flux

class ComputeCellGradient:
    """Class to compute cell gradients for a given field array"""
    def __init__(self, mesh_object, in_field_array, out_field_array):
        self.mesh_object_ : MeshObject = mesh_object
        self.field_array_ : FieldArray = in_field_array
        self.out_field_array_ : FieldArray = out_field_array
    
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
            cellID = face.get_left_cell()  # For boundary faces, only one cell contributes
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

