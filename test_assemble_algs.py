import unittest
import tempfile
import os
import yaml
import numpy as np
from YamlParser import InputConfigParser
from MeshObject import MeshObject
from QuadElement import Face, Cell
from FieldsHolder import FieldArray, DimType, FieldNames, MAX_DIM
from Operations import ComputeInteriorMassFlux, ComputeCellGradient
from AssembleAlgorithms import AssembleInteriorScalarDiffusionToLinSystem, AssembleDirichletBoundaryScalarDiffusionToLinSystem, Boundary, \
	AssembleCellVectorTimeTerm, AssembleInteriorVectorDiffusionToLinSystem, AssembleDirichletBoundaryVectorDiffusionToLinSystem, AssembleInteriorVectorAdvectionToLinSystem
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
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlg.assemble()
		
        # The boundary isn't set up, but we know the center interior cell should look like  [ 0. -1.  0. -1.  4. -1.  0. -1.  0.] from a finite difference perspective.
		cellIndexCheck = 4
		exactDiffComponent = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]
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
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlgLeft = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", pressureField, system, mesh, boundaryType=Boundary.LEFT, boundaryValue=leftValue, diffusionCoeff=1.0)
		diffusionScalarAlgRight = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", pressureField, system, mesh, boundaryType=Boundary.RIGHT, boundaryValue=rightValue, diffusionCoeff=1.0)
		allAlgs = [diffusionScalarAlg, diffusionScalarAlgLeft, diffusionScalarAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dP = system.solve()
		
        ## we should know what the RHS value should be on the boundary for this setup....
		rightBoundaryFaceCells =  mesh.get_right_boundary()
		faceArea = rightBoundaryFaceCells[0].get_area()
		rightBoundaryCellRHS = -(rightValue - 0)/(faceArea/2)*faceArea
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
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlgLeft = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", pressureField, system, mesh, boundaryType=Boundary.LEFT, boundaryValue=leftValue, diffusionCoeff=1.0)
		diffusionScalarAlgRight = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", pressureField, system, mesh, boundaryType=Boundary.RIGHT, boundaryValue=rightValue, diffusionCoeff=1.0)
		allAlgs = [diffusionScalarAlg, diffusionScalarAlgLeft, diffusionScalarAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dP = system.solve()
		
        ## we should know what the RHS value should be on the boundary for this setup....
		rightBoundaryFaceCells =  mesh.get_right_boundary()
		faceArea = rightBoundaryFaceCells[0].get_area()
		rightBoundaryCellRHS = -(rightValue - 0)/(faceArea/2)*faceArea
		for face in rightBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			self.assertAlmostEqual(system.get_rhs()[leftCellID], rightBoundaryCellRHS, places=5)

		leftBoundaryFaceCells =  mesh.get_left_boundary()
		leftBoundaryCellRHS = (0 - leftValue)/(faceArea/2)*faceArea # normal points in negative direction for left boundary
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
		diffusionScalarAlg = AssembleInteriorScalarDiffusionToLinSystem("Pressure_diffusion_test", pressureField, system, mesh, diffusionCoeff=1.0)
		diffusionScalarAlgLeft = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", pressureField, system, mesh, boundaryType=Boundary.TOP, boundaryValue=topValue, diffusionCoeff=1.0)
		diffusionScalarAlgRight = AssembleDirichletBoundaryScalarDiffusionToLinSystem("Pressure_diffusion_test_boundary", pressureField, system, mesh, boundaryType=Boundary.BOTTOM, boundaryValue=bottomValue, diffusionCoeff=1.0)
		allAlgs = [diffusionScalarAlg, diffusionScalarAlgLeft, diffusionScalarAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dP = system.solve()
		
        ## we should know what the RHS value should be on the boundary for this setup....
		bottomBoundaryFaceCells =  mesh.get_bottom_boundary()
		faceArea = bottomBoundaryFaceCells[0].get_area()
		bottomBoundaryCellRHS = (0 - bottomValue)/(faceArea/2)*faceArea
		for face in bottomBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			self.assertAlmostEqual(system.get_rhs()[leftCellID], bottomBoundaryCellRHS, places=5)
			
		topBoundaryFaceCells =  mesh.get_top_boundary()
		faceArea = topBoundaryFaceCells[0].get_area()
		topBoundaryCellRHS = -(topValue - 0)/(faceArea/2)*faceArea
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
		expectedValue = -exactVolume/dt
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		newValue=2
		oldValue=0
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		velocityNp1 = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, mesh.get_num_cells())
		velocityN = FieldArray(FieldNames.VELOCITY_OLD.value, DimType.VECTOR, mesh.get_num_cells())
		velocityNp1.initialize_constant(newValue)
		velocityN.initialize_constant(oldValue)
		system = LinearSystem(mesh.get_num_cells()*MAX_DIM, "test_system", sparse=False)
		timeTermAlg = AssembleCellVectorTimeTerm("Velocity_time_term", velocityNp1, velocityN, system, mesh, dt)
		timeTermAlg.zero()
		timeTermAlg.assemble()
		for cell in mesh.get_cells():
			cellID = cell.get_flat_id()
			for comp in range(MAX_DIM):
				rowIndex = cellID*MAX_DIM + comp
				self.assertAlmostEqual(system.get_lhs()[rowIndex, rowIndex], expectedValue, places=5)
				self.assertAlmostEqual(system.get_rhs()[rowIndex], -expectedValue*(newValue-oldValue), places=5)

	def test_3x3_full_vector_diffusion_nonzero_x(self):
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
		rightValue = [1.0,0.0]
		leftValue = [10.0,0.0]
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		velocityField = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, mesh.get_num_cells())
		velocityField.initialize_constant(0.)  # Initialize field to zero
		system = LinearSystem(mesh.get_num_cells()*MAX_DIM, "test_system", sparse=False)
		diffusionVectorAlg = AssembleInteriorVectorDiffusionToLinSystem("Velocity_diffusion_test", velocityField, system, mesh, diffusionCoeff=1.0)
		diffusionVectorAlgLeft = AssembleDirichletBoundaryVectorDiffusionToLinSystem("Velocity_diffusion_test_boundary", velocityField, system, mesh, boundaryType=Boundary.LEFT, boundaryValue=leftValue, diffusionCoeff=1.0)
		diffusionVectorAlgRight = AssembleDirichletBoundaryVectorDiffusionToLinSystem("Velocity_diffusion_test_boundary", velocityField, system, mesh, boundaryType=Boundary.RIGHT, boundaryValue=rightValue, diffusionCoeff=1.0)
		allAlgs = [diffusionVectorAlg, diffusionVectorAlgLeft, diffusionVectorAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dU = system.solve()
		
        ## we should know what the RHS value should be on the boundary for this setup....
		rightBoundaryFaceCells =  mesh.get_right_boundary()
		faceArea = rightBoundaryFaceCells[0].get_area()
		rightBoundaryCellRHS = [-(rightValue[i] - 0)/(faceArea/2)*faceArea for i in range(MAX_DIM)]
		for face in rightBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			for comp in range(MAX_DIM):
				self.assertAlmostEqual(system.get_rhs()[leftCellID*MAX_DIM + comp], rightBoundaryCellRHS[comp], places=5)

		leftBoundaryFaceCells =  mesh.get_left_boundary()
		leftBoundaryCellRHS = [(0 - leftValue[i])/(faceArea/2)*faceArea for i in range(MAX_DIM)] # normal points in negative direction for left boundary
		for face in leftBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			for comp in range(MAX_DIM):
				self.assertAlmostEqual(system.get_rhs()[leftCellID*MAX_DIM + comp], leftBoundaryCellRHS[comp], places=5)
		
        # linear profile for dP, our 
		for cell in mesh.get_cells():
			cellID = cell.get_flat_id()
			slope = [(rightValue[i]-leftValue[i])/1.0 for i in range(MAX_DIM)]
			expectedValue = [cell.get_centroid()[0]*slope[i] + leftValue[i] for i in range(MAX_DIM)]  # Linear profile from rightValue to leftValue across the domain
			for comp in range(MAX_DIM):
				self.assertAlmostEqual(dU[cellID*MAX_DIM + comp], expectedValue[comp], places=5)
	
	def test_3x3_full_vector_diffusion_nonzero_y(self):
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
		rightValue = [0.0,10.0]
		leftValue = [0.0,2.0]
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		velocityField = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, mesh.get_num_cells())
		velocityField.initialize_constant(0.)  # Initialize field to zero
		system = LinearSystem(mesh.get_num_cells()*MAX_DIM, "test_system", sparse=False)
		diffusionVectorAlg = AssembleInteriorVectorDiffusionToLinSystem("Velocity_diffusion_test", velocityField, system, mesh, diffusionCoeff=1.0)
		diffusionVectorAlgLeft = AssembleDirichletBoundaryVectorDiffusionToLinSystem("Velocity_diffusion_test_boundary", velocityField, system, mesh, boundaryType=Boundary.LEFT, boundaryValue=leftValue, diffusionCoeff=1.0)
		diffusionVectorAlgRight = AssembleDirichletBoundaryVectorDiffusionToLinSystem("Velocity_diffusion_test_boundary", velocityField, system, mesh, boundaryType=Boundary.RIGHT, boundaryValue=rightValue, diffusionCoeff=1.0)
		allAlgs = [diffusionVectorAlg, diffusionVectorAlgLeft, diffusionVectorAlgRight]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		dU = system.solve()
		
        ## we should know what the RHS value should be on the boundary for this setup....
		rightBoundaryFaceCells =  mesh.get_right_boundary()
		faceArea = rightBoundaryFaceCells[0].get_area()
		rightBoundaryCellRHS = [-(rightValue[i] - 0)/(faceArea/2)*faceArea for i in range(MAX_DIM)]
		for face in rightBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			for comp in range(MAX_DIM):
				self.assertAlmostEqual(system.get_rhs()[leftCellID*MAX_DIM + comp], rightBoundaryCellRHS[comp], places=5)

		leftBoundaryFaceCells =  mesh.get_left_boundary()
		leftBoundaryCellRHS = [(0 - leftValue[i])/(faceArea/2)*faceArea for i in range(MAX_DIM)] # normal points in negative direction for left boundary
		for face in leftBoundaryFaceCells:
			leftCellID = face.get_left_cell()
			for comp in range(MAX_DIM):
				self.assertAlmostEqual(system.get_rhs()[leftCellID*MAX_DIM + comp], leftBoundaryCellRHS[comp], places=5)
		
        # linear profile for dP, our 
		for cell in mesh.get_cells():
			cellID = cell.get_flat_id()
			slope = [(rightValue[i]-leftValue[i])/1.0 for i in range(MAX_DIM)]
			expectedValue = [cell.get_centroid()[0]*slope[i] + leftValue[i] for i in range(MAX_DIM)]  # Linear profile from rightValue to leftValue across the domain
			for comp in range(MAX_DIM):
				self.assertAlmostEqual(dU[cellID*MAX_DIM + comp], expectedValue[comp], places=5)

	def test_3x3_interior_vector_advection(self):
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
		velocityField = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, mesh.get_num_cells())
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		velocityField.initialize_constant(0.)  # Initialize field to zero
		pressureField.initialize_constant(0.)
		for cell in mesh.get_cells():
			cellID = cell.get_flat_id()
			velocityField.get_data()[cellID*MAX_DIM] = 1.0  
			velocityField.get_data()[cellID*MAX_DIM+1] = 1.0  
		gradPressureField = FieldArray(FieldNames.GRAD_PRESSURE.value, DimType.VECTOR, mesh.get_num_cells())
		gradient_op = ComputeCellGradient(mesh, pressureField, gradPressureField)
		gradient_op.compute_scalar_gradient()
		massFluxAlg = ComputeInteriorMassFlux(mesh, velocityField, pressureField, gradPressureField, dt=0.01)
		massFluxAlg.compute_mass_flux()

		system = LinearSystem(mesh.get_num_cells()*MAX_DIM, "test_system", sparse=False)
		advectionVectorAlg = AssembleInteriorVectorAdvectionToLinSystem("Velocity_advection_test", velocityField, system, mesh)
		allAlgs = [advectionVectorAlg]
		for alg in allAlgs:
			alg.zero()
		for alg in allAlgs:
			alg.assemble()
		# make sure the lhs and rhs are consistent, should be the same with a reversed sign if this is a no flux BC
		testLHS = system.get_lhs() @ velocityField.get_data()
		for i in range(testLHS.shape[0]):
			self.assertAlmostEqual(-testLHS[i], system.get_rhs()[i], places=5)