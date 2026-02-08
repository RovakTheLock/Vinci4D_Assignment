import unittest
import tempfile
import os
import yaml
import numpy as np
from YamlParser import InputConfigParser
from MeshObject import MeshObject
from QuadElement import Face, Cell
import FieldsHolder as FH
import Operations as Ops


class TestOperations(unittest.TestCase):
	def setUp(self):
		self.tmpdir = tempfile.mkdtemp()

	def tearDown(self):
		for f in os.listdir(self.tmpdir):
			os.remove(os.path.join(self.tmpdir, f))
		os.rmdir(self.tmpdir)
	def test_3x3_cell_grad_op_scalar(self):
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
		pressureField = FH.FieldArray(FH.FieldNames.PRESSURE.value, FH.DimType.SCALAR, mesh.get_num_cells())
		## initialize pressure field to be linear in x for testing gradient
		for i in range(mesh.get_num_cells()):
			pressureField.get_data()[i] = mesh.get_cells()[i].get_centroid()[0]  # Set pressure to x-coordinate of cell center
		gradPressureField = FH.FieldArray(FH.FieldNames.GRAD_PRESSURE.value, FH.DimType.VECTOR, mesh.get_num_cells())
		gradient_op = Ops.ComputeCellGradient(mesh, pressureField, gradPressureField)
		gradient_op.compute_scalar_gradient()
        # for uniform grid, we expect the gradient to be approximately 1 in x direction and 0 in y direction in interior cells. Boundary cells will have different values due to one-sided difference...
		for interiorCell in mesh.get_interior_cells():
			cellIndex = interiorCell.get_flat_id()
			grad_x = gradPressureField.get_data()[2*cellIndex]  # x component of gradient
			grad_y = gradPressureField.get_data()[2*cellIndex + 1]  # y component of gradient
			self.assertAlmostEqual(grad_x, 1.0, places=5)
			self.assertAlmostEqual(grad_y, 0.0, places=5)
		

	def test_6x6_cell_grad_op_scalar_with_slope(self):
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
		pressureField = FH.FieldArray(FH.FieldNames.PRESSURE.value, FH.DimType.SCALAR, mesh.get_num_cells())
		## initialize pressure field to be linear in x for testing gradient
		for i in range(mesh.get_num_cells()):
			pressureField.get_data()[i] = slope*mesh.get_cells()[i].get_centroid()[0]  # Set pressure to x-coordinate of cell center
		gradPressureField = FH.FieldArray(FH.FieldNames.GRAD_PRESSURE.value, FH.DimType.VECTOR, mesh.get_num_cells())
		gradient_op = Ops.ComputeCellGradient(mesh, pressureField, gradPressureField)
		gradient_op.compute_scalar_gradient()
        # for uniform grid, we expect the gradient to be approximately 'slope' in x direction and 0 in y direction in interior cells. Boundary cells will have different values due to one-sided difference...
		for interiorCell in mesh.get_interior_cells():
			cellIndex = interiorCell.get_flat_id()
			grad_x = gradPressureField.get_data()[2*cellIndex]  # x component of gradient
			grad_y = gradPressureField.get_data()[2*cellIndex + 1]  # y component of gradient
			self.assertAlmostEqual(grad_x, slope, places=5)
			self.assertAlmostEqual(grad_y, 0.0, places=5)
		