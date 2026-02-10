import unittest
import tempfile
import os
import yaml
import numpy as np
from YamlParser import InputConfigParser
from MeshObject import MeshObject
from QuadElement import Face, Cell
import FieldsHolder as FH
from AssembleAlgorithms import AssembleInteriorScalarDiffusionToLinSystem, AssembleDirichletBoundaryScalarDiffusionToLinSystem, Boundary
from LinearSystem import LinearSystem


class TestOperations(unittest.TestCase):
	def setUp(self):
		self.tmpdir = tempfile.mkdtemp()

	def tearDown(self):
		for f in os.listdir(self.tmpdir):
			os.remove(os.path.join(self.tmpdir, f))
		os.rmdir(self.tmpdir)

	def test_3x3_interior_diffusion(self):
		cells_x = 3
		cells_y = 3
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': cells_x,
				'num_cells_y': cells_y
			}
		}
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FH.FieldArray(FH.FieldNames.PRESSURE.value, FH.DimType.SCALAR, mesh.get_num_cells())
		pressureField.initialize_constant(0.)  # Initialize pressure field to zero
		system = LinearSystem(mesh.get_num_cells(), "test_system", sparse=False)
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", mesh.get_num_cells(), pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlg.assemble()
		
        # The boundary isn't set up, but we know the center interior cell should look like  [ 0. -1.  0. -1.  4. -1.  0. -1.  0.] from a finite difference perspective.
		cellIndexCheck = 4
		exactDiffComponent = [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0]
		for i in range(mesh.get_num_cells()):
			centerCellValue_i = system.get_lhs()[cellIndexCheck, i]
			self.assertAlmostEqual(centerCellValue_i, exactDiffComponent[i], places=5)	
		
	def test_3x3_full_diffusion(self):
		cells_x = 3
		cells_y = 3
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': cells_x,
				'num_cells_y': cells_y
			}
		}
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FH.FieldArray(FH.FieldNames.PRESSURE.value, FH.DimType.SCALAR, mesh.get_num_cells())
		pressureField.initialize_constant(0.)  # Initialize pressure field to zero
		system = LinearSystem(mesh.get_num_cells(), "test_system", sparse=False)
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", mesh.get_num_cells(), pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlgLeft = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", mesh.get_num_cells(), pressureField, system, mesh, boundaryType=Boundary.LEFT, boundaryValue=0.0, diffusionCoeff=1.0)
		diffusionScalarAlgRight = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", mesh.get_num_cells(), pressureField, system, mesh, boundaryType=Boundary.RIGHT, boundaryValue=1.0, diffusionCoeff=1.0)
		allAlgs = [diffusionScalarAlg, diffusionScalarAlgLeft, diffusionScalarAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dP = system.solve()
		print(pressureField)
		print(dP)
		
		