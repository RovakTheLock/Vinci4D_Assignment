import unittest
import tempfile
import os
import yaml
import numpy as np
from YamlParser import InputConfigParser
from MeshObject import MeshObject
from QuadElement import Face, Cell
from FieldsHolder import FieldArray, FieldNames, DimType, MAX_DIM
from Operations import ComputeInteriorMassFlux, ComputeCellGradient, CFLTimeStepCompute
import math


class TestOperations(unittest.TestCase):
	def setUp(self):
		self.tmpdir = tempfile.mkdtemp()

	def tearDown(self):
		for f in os.listdir(self.tmpdir):
			os.remove(os.path.join(self.tmpdir, f))
		os.rmdir(self.tmpdir)
	def test_cfl_time_step_compute(self):
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 2],
				'y_range': [0, 1],
				'num_cells_x': 4,
				'num_cells_y': 2
			}
		}
		path = os.path.join(self.tmpdir, 'mesh_cfl.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		cfl_value = 0.5
		time_step_op = CFLTimeStepCompute(mesh, cfl_value)

		dt = time_step_op.compute_time_step()
		expected_dt = 0.5 * min((2 - 0) / 4, (1 - 0) / 2)
		self.assertAlmostEqual(dt, expected_dt, places=12)
	def test_3x3_cell_grad_op_scalar(self):
		"""Test that we compute the right gradient given a linear field with slope 1"""
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': 3,
				'num_cells_y': 3
			}
		}
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		## initialize pressure field to be linear in x for testing gradient
		for i in range(mesh.get_num_cells()):
			pressureField.get_data()[i] = mesh.get_cells()[i].get_centroid()[0]  # Set pressure to x-coordinate of cell center
		gradPressureField = FieldArray(FieldNames.GRAD_PRESSURE.value, DimType.VECTOR, mesh.get_num_cells())
		gradient_op = ComputeCellGradient(mesh, pressureField, gradPressureField)
		gradient_op.compute_scalar_gradient()
        # for uniform grid, we expect the gradient to be approximately 1 in x direction and 0 in y direction in interior cells. Boundary cells will have different values due to one-sided difference...
		for interiorCell in mesh.get_interior_cells():
			cellIndex = interiorCell.get_flat_id()
			grad_x = gradPressureField.get_data()[2*cellIndex]  # x component of gradient
			grad_y = gradPressureField.get_data()[2*cellIndex + 1]  # y component of gradient
			self.assertAlmostEqual(grad_x, 1.0, places=5)
			self.assertAlmostEqual(grad_y, 0.0, places=5)
		

	def test_6x6_cell_grad_op_scalar_with_slope(self):
		"""Test that we compute the right gradient given a linear field, extended domain with a non-1 slope."""
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': 6,
				'num_cells_y': 6
			}
		}
		slope = 10.0
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		## initialize pressure field to be linear in x for testing gradient
		for i in range(mesh.get_num_cells()):
			pressureField.get_data()[i] = slope*mesh.get_cells()[i].get_centroid()[0]  # Set pressure to x-coordinate of cell center
		gradPressureField = FieldArray(FieldNames.GRAD_PRESSURE.value, DimType.VECTOR, mesh.get_num_cells())
		gradient_op = ComputeCellGradient(mesh, pressureField, gradPressureField)
		gradient_op.compute_scalar_gradient()
        # for uniform grid, we expect the gradient to be approximately 'slope' in x direction and 0 in y direction in interior cells. Boundary cells will have different values due to one-sided difference...
		for interiorCell in mesh.get_interior_cells():
			cellIndex = interiorCell.get_flat_id()
			grad_x = gradPressureField.get_data()[2*cellIndex]  # x component of gradient
			grad_y = gradPressureField.get_data()[2*cellIndex + 1]  # y component of gradient
			self.assertAlmostEqual(grad_x, slope, places=5)
			self.assertAlmostEqual(grad_y, 0.0, places=5)
	def test_3x3_cell_mass_flux_zero_pressure(self):
		"""Test that we get a summed mass flux of zero for a div-free velocity with zero pressure gradients."""
		numCellsX=3
		numCellsY=3
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': numCellsX,
				'num_cells_y': numCellsY
			}
		}
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		velocityField = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, mesh.get_num_cells())
		## initialize pressure field to be linear in x for testing gradient
		for i in range(mesh.get_num_cells()):
			cell_x = mesh.get_cells()[i].get_centroid()[0]
			cell_y = mesh.get_cells()[i].get_centroid()[1]
			velocityField.get_data()[MAX_DIM*i] = math.pi*math.sin(math.pi*cell_x)*math.cos(math.pi*cell_y) 
			velocityField.get_data()[MAX_DIM*i + 1] = -math.pi*math.cos(math.pi*cell_x)*math.sin(math.pi*cell_y)
		pressureField.initialize_constant(0.0)
		gradPressureField = FieldArray(FieldNames.GRAD_PRESSURE.value, DimType.VECTOR, mesh.get_num_cells())
		gradient_op = ComputeCellGradient(mesh, pressureField, gradPressureField)
		gradient_op.compute_scalar_gradient()
		massFluxAlg = ComputeInteriorMassFlux(mesh, velocityField, pressureField, gradPressureField, dt=0.01)
		massFluxAlg.compute_mass_flux()
		faceListTest = np.zeros(mesh.get_num_cells())
		for face in mesh.get_internal_faces():
			leftCellID = face.get_left_cell()
			rightCellID = face.get_right_cell()
			faceListTest[leftCellID] += face.massFlux_
			faceListTest[rightCellID] -= face.massFlux_
		for cell in faceListTest:
			self.assertAlmostEqual(cell, 0.0, places=5)
	def test_3x3_cell_mass_flux_with_pressure(self):
		numCellsX=3
		numCellsY=3
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': numCellsX,
				'num_cells_y': numCellsY
			}
		}
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		velocityField = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, mesh.get_num_cells())
		## initialize pressure field to be linear in x for testing gradient
		for i in range(mesh.get_num_cells()):
			cell_x = mesh.get_cells()[i].get_centroid()[0]
			cell_y = mesh.get_cells()[i].get_centroid()[1]
			pressureField.get_data()[i] = math.pi/2*(math.cos(2*math.pi*cell_x) + math.cos(2*math.pi*cell_y))
			velocityField.get_data()[MAX_DIM*i] = math.pi*math.sin(math.pi*cell_x)*math.cos(math.pi*cell_y) 
			velocityField.get_data()[MAX_DIM*i + 1] = -math.pi*math.cos(math.pi*cell_x)*math.sin(math.pi*cell_y)
		gradPressureField = FieldArray(FieldNames.GRAD_PRESSURE.value, DimType.VECTOR, mesh.get_num_cells())
		gradient_op = ComputeCellGradient(mesh, pressureField, gradPressureField)
		gradient_op.compute_scalar_gradient()
		massFluxAlg = ComputeInteriorMassFlux(mesh, velocityField, pressureField, gradPressureField, dt=0.01)
		massFluxAlg.compute_mass_flux()
		faceListTest = np.zeros(mesh.get_num_cells())
		for face in mesh.get_internal_faces():
			leftCellID = face.get_left_cell()
			rightCellID = face.get_right_cell()
			faceListTest[leftCellID] += face.massFlux_
			faceListTest[rightCellID] -= face.massFlux_