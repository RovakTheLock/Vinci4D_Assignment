import unittest
import tempfile
import os
import yaml
import numpy as np
from YamlParser import InputConfigParser
from MeshObject import MeshObject
from QuadElement import Face, Cell
from FieldsHolder import FieldArray, DimType, FieldNames, MAX_DIM
from AssembleAlgorithms import AssembleInteriorScalarDiffusionToLinSystem, AssembleDirichletBoundaryScalarDiffusionToLinSystem, Boundary, AssembleCellVectorTimeTerm
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
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
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
		rightValue = 1.0
		leftValue = 0.0
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		pressureField.initialize_constant(0.)  # Initialize pressure field to zero
		system = LinearSystem(mesh.get_num_cells(), "test_system", sparse=False)
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", mesh.get_num_cells(), pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlgLeft = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", mesh.get_num_cells(), pressureField, system, mesh, boundaryType=Boundary.LEFT, boundaryValue=leftValue, diffusionCoeff=1.0)
		diffusionScalarAlgRight = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", mesh.get_num_cells(), pressureField, system, mesh, boundaryType=Boundary.RIGHT, boundaryValue=rightValue, diffusionCoeff=1.0)
		allAlgs = [diffusionScalarAlg, diffusionScalarAlgLeft, diffusionScalarAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dP = system.solve()
		
        ## we should know what the RHS value should be on the boundary for this setup....
		rightBoundaryFaceCells =  mesh.get_right_boundary()
		faceArea = rightBoundaryFaceCells[0].get_area()
		rightBoundaryCellRHS = (rightValue - 0)/(faceArea/2)*faceArea
		for face in rightBoundaryFaceCells:
			rightCellID = face.get_right_cell()
			leftCellID = face.get_left_cell()
			self.assertEqual(rightCellID, None)  
			self.assertAlmostEqual(system.get_rhs()[leftCellID], rightBoundaryCellRHS, places=5)
		
        # linear profile for dP, our solution
		for cell in mesh.get_cells():
			cellID = cell.get_flat_id()
			expectedValue = cell.get_centroid()[0]  # Linear profile from 0 to 1 across the domain
			self.assertAlmostEqual(dP[cellID], expectedValue, places=5)
        
		
	def test_3x3_full_diffusion_nonzero(self):
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
		rightValue = 1.0
		leftValue = 10.0
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		pressureField.initialize_constant(0.)  # Initialize pressure field to zero
		system = LinearSystem(mesh.get_num_cells(), "test_system", sparse=False)
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", mesh.get_num_cells(), pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlgLeft = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", mesh.get_num_cells(), pressureField, system, mesh, boundaryType=Boundary.LEFT, boundaryValue=leftValue, diffusionCoeff=1.0)
		diffusionScalarAlgRight = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", mesh.get_num_cells(), pressureField, system, mesh, boundaryType=Boundary.RIGHT, boundaryValue=rightValue, diffusionCoeff=1.0)
		allAlgs = [diffusionScalarAlg, diffusionScalarAlgLeft, diffusionScalarAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dP = system.solve()
		
        ## we should know what the RHS value should be on the boundary for this setup....
		rightBoundaryFaceCells =  mesh.get_right_boundary()
		faceArea = rightBoundaryFaceCells[0].get_area()
		rightBoundaryCellRHS = (rightValue - 0)/(faceArea/2)*faceArea
		for face in rightBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			self.assertAlmostEqual(system.get_rhs()[leftCellID], rightBoundaryCellRHS, places=5)

		leftBoundaryFaceCells =  mesh.get_left_boundary()
		leftBoundaryCellRHS = -(0 - leftValue)/(faceArea/2)*faceArea # normal points in negative direction for left boundary
		for face in leftBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			self.assertAlmostEqual(system.get_rhs()[leftCellID], leftBoundaryCellRHS, places=5)
		
        # linear profile for dP, our 
		for cell in mesh.get_cells():
			cellID = cell.get_flat_id()
			slope = (rightValue-leftValue)/1.0
			expectedValue = cell.get_centroid()[0]*slope + leftValue  # Linear profile from rightValue to leftValue across the domain
			self.assertAlmostEqual(dP[cellID], expectedValue, places=5)
			
	def test_3x3_full_diffusion_nonzero_vertical(self):
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
		bottomValue = 1.0
		topValue = 10.0
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		pressureField.initialize_constant(0.)  # Initialize pressure field to zero
		system = LinearSystem(mesh.get_num_cells(), "test_system", sparse=False)
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", mesh.get_num_cells(), pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlgLeft = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", mesh.get_num_cells(), pressureField, system, mesh, boundaryType=Boundary.TOP, boundaryValue=topValue, diffusionCoeff=1.0)
		diffusionScalarAlgRight = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", mesh.get_num_cells(), pressureField, system, mesh, boundaryType=Boundary.BOTTOM, boundaryValue=bottomValue, diffusionCoeff=1.0)
		allAlgs = [diffusionScalarAlg, diffusionScalarAlgLeft, diffusionScalarAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dP = system.solve()
		
        ## we should know what the RHS value should be on the boundary for this setup....
		bottomBoundaryFaceCells =  mesh.get_bottom_boundary()
		faceArea = bottomBoundaryFaceCells[0].get_area()
		bottomBoundaryCellRHS = -(0 - bottomValue)/(faceArea/2)*faceArea
		for face in bottomBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			self.assertAlmostEqual(system.get_rhs()[leftCellID], bottomBoundaryCellRHS, places=5)
			
		topBoundaryFaceCells =  mesh.get_top_boundary()
		faceArea = topBoundaryFaceCells[0].get_area()
		topBoundaryCellRHS = (topValue - 0)/(faceArea/2)*faceArea
		for face in topBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			self.assertAlmostEqual(system.get_rhs()[leftCellID], topBoundaryCellRHS, places=5)
		
        # linear profile for dP, our 
		for cell in mesh.get_cells():
			cellID = cell.get_flat_id()
			slope = (topValue-bottomValue)/1.0
			expectedValue = cell.get_centroid()[1]*slope + bottomValue  # Linear profile from rightValue to leftValue across the domain
			self.assertAlmostEqual(dP[cellID], expectedValue, places=5)

	def test_3x3_time_term(self):
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
		dt = 1.0e-3
		exactVolume = 1.0/(cells_x*cells_y)
		expectedValue = exactVolume/dt
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		velocityNp1 = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, mesh.get_num_cells())
		velocityN = FieldArray(FieldNames.VELOCITY_OLD.value, DimType.VECTOR, mesh.get_num_cells())
		system = LinearSystem(mesh.get_num_cells()*MAX_DIM, "test_system", sparse=False)
		timeTermAlg = AssembleCellVectorTimeTerm("Velocity_time_term", mesh.get_num_cells(), velocityNp1, velocityN, system, mesh, dt)
		timeTermAlg.zero()
		timeTermAlg.assemble()
		for cell in mesh.get_cells():
			cellID = cell.get_flat_id()
			for comp in range(MAX_DIM):
				rowIndex = cellID*MAX_DIM + comp
				self.assertAlmostEqual(system.get_lhs()[rowIndex, rowIndex], expectedValue, places=5)
	