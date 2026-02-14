import unittest
import tempfile
import os
import yaml
import numpy as np
from YamlParser import InputConfigParser
from MeshObject import MeshObject
from FieldsHolder import FieldArray, DimType, FieldNames, MAX_DIM
from AssembleAlgorithms import Boundary, AssembleCellVectorTimeTerm, AssembleInteriorVectorDiffusionToLinSystem, AssembleInteriorPressurePoissonSystem, \
	AssembleDirichletBoundaryVectorDiffusionToLinSystem, AssembleInteriorVectorAdvectionToLinSystem, AssembleCellVectorPressureGradientToLinSystem
from LinearSystem import LinearSystem
from Operations import ComputeInteriorMassFlux, ComputeCellGradient
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys


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


class TestOperations(unittest.TestCase):
	def setUp(self):
		self.tmpdir = tempfile.mkdtemp()

	def tearDown(self):
		for f in os.listdir(self.tmpdir):
			os.remove(os.path.join(self.tmpdir, f))
		os.rmdir(self.tmpdir)


	def test_nxn_solve(self):
		'''
		test on a nxn grid for a lid-driven cavity setup.
		'''
		cells_x = 40
		cells_y = 40
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': cells_x,
				'num_cells_y': cells_y
			}
		}
		vTop = 1.0
		topValue = [vTop,0.0]
		rightValue = [0.0,0.0]
		leftValue = [0.0,0.0]
		bottomValue = [0.0,0.0]
		Re = 100000
		diffusionCoefficient = 1/Re 
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		velocityNp1 = FieldArray(FieldNames.VELOCITY_NEW.value, DimType.VECTOR, mesh.get_num_cells())
		velocityN = FieldArray(FieldNames.VELOCITY_OLD.value, DimType.VECTOR, mesh.get_num_cells())
		pressureField = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
		gradPressureField = FieldArray(FieldNames.GRAD_PRESSURE.value, DimType.VECTOR, mesh.get_num_cells())
		dPressure = FieldArray(FieldNames.PRESSURE_CORRECTION.value, DimType.SCALAR, mesh.get_num_cells())
		grad_dPressure = FieldArray(FieldNames.GRAD_PRESSURE_CORRECTION.value, DimType.VECTOR, mesh.get_num_cells())
		velocityNp1.initialize_constant(0.)  # Initialize field to zero
		velocityN.initialize_constant(0.)  # Initialize field to zero
		pressureField.initialize_constant(0.)  # Initialize field to zero
		gradPressureField.initialize_constant(0.)  # Initialize field to zero
		dPressure.initialize_constant(0.)  # Initialize field to zero
		dt = 0.1 #dt, for fast convergence, needs to be either Re*dx*dx/2 or dx/U
		velocitySystem = LinearSystem(mesh.get_num_cells()*MAX_DIM, "velocity_system", sparse=False)
		pressureSystem = LinearSystem(mesh.get_num_cells(), "pressure_system", sparse=False)
		timeTermAlg = AssembleCellVectorTimeTerm("Velocity_time_term", velocityNp1, velocityN, velocitySystem, mesh, dt)
		pressureGradAlg = AssembleCellVectorPressureGradientToLinSystem("Pressure_gradient_source", gradPressureField, velocitySystem, mesh)
		diffusionVectorAlg = AssembleInteriorVectorDiffusionToLinSystem("Velocity_diffusion_test", velocityNp1, velocitySystem, mesh, diffusionCoeff=diffusionCoefficient)
		diffusionVectorAlgLeft = AssembleDirichletBoundaryVectorDiffusionToLinSystem("Velocity_diffusion_left_boundary",  velocityNp1, velocitySystem, mesh, boundaryType=Boundary.LEFT, boundaryValue=leftValue, diffusionCoeff=diffusionCoefficient)
		diffusionVectorAlgRight = AssembleDirichletBoundaryVectorDiffusionToLinSystem("Velocity_diffusion_right_boundary",  velocityNp1, velocitySystem, mesh, boundaryType=Boundary.RIGHT, boundaryValue=rightValue, diffusionCoeff=diffusionCoefficient)
		diffusionVectorAlgTop = AssembleDirichletBoundaryVectorDiffusionToLinSystem("Velocity_diffusion_top_boundary",  velocityNp1, velocitySystem, mesh, boundaryType=Boundary.TOP, boundaryValue=topValue, diffusionCoeff=diffusionCoefficient)
		diffusionVectorAlgBottom = AssembleDirichletBoundaryVectorDiffusionToLinSystem("Velocity_diffusion_bottom_boundary",  velocityNp1, velocitySystem, mesh, boundaryType=Boundary.BOTTOM, boundaryValue=bottomValue, diffusionCoeff=diffusionCoefficient)
		advectionVelocityAlgInterior = AssembleInteriorVectorAdvectionToLinSystem("Velocity_advection_test",  velocityNp1, velocitySystem, mesh)
		ppeAlg = AssembleInteriorPressurePoissonSystem("Pressure_poisson", pressureField, pressureSystem, mesh, dt)
		pressureGradientOp = ComputeCellGradient(mesh, pressureField, gradPressureField)
		massFluxAlg = ComputeInteriorMassFlux(mesh, velocityNp1, pressureField, gradPressureField, dt)
		velocitySystemAlgs = [diffusionVectorAlg, diffusionVectorAlgLeft, diffusionVectorAlgRight, diffusionVectorAlgTop, diffusionVectorAlgBottom, timeTermAlg, pressureGradAlg, advectionVelocityAlgInterior]
		pressureSystemAlg = [ppeAlg]
		pressureGradientOp.compute_scalar_gradient()
		massFluxAlg.compute_mass_flux()

		pressureCorrectionGradientOp = ComputeCellGradient(mesh, dPressure, grad_dPressure)
		myLogger = LogObject([velocitySystem,pressureSystem])
		directoryName = 'RESULTS'
		numSteps = 1000
		numNonlinearIterations = 10
		alphaV = 0.7
		alphaP = 0.3
		tolerance = 1e-5
		timeOutputFreq = 20
		if not os.path.exists(directoryName):
			os.makedirs(directoryName)
			print(f"Created directory: {directoryName}")
		else:
			print("Directory already exists.")
		for t in range(numSteps):
			print(f"t = {t}")
			print("-------------------------------------------------------------------------------------------")
			for iter in range(numNonlinearIterations):
				# Solve velocity system and do various updates.
				for alg in velocitySystemAlgs:
					alg.zero()
				for alg in velocitySystemAlgs:
					alg.assemble()
				dU = velocitySystem.solve(method='gmres')
				normRHS = np.linalg.norm(velocitySystem.get_rhs())
				velocityNp1.increment(dU, scale=alphaV)
				massFluxAlg.compute_mass_flux()

				# Solve pressure system and do various updates
				for alg in pressureSystemAlg:
					alg.zero()
				for alg in pressureSystemAlg:
					alg.assemble()
				# fix one of the cells to 0 pressure to avoid singularity..
#				pressureSystem.get_lhs()[0,:] = 0
#				pressureSystem.get_lhs()[0,0] = 1.0
#				pressureSystem.get_rhs()[0] = 0
				dPressure.data_ = pressureSystem.solve(method='gmres') # being lazy here..need to expose the underlying member
				pressureField.increment(dPressure.get_data(), scale=alphaP)
				pressureCorrectionGradientOp.compute_scalar_gradient()
				pressureGradientOp.compute_scalar_gradient()
#				massFluxAlg.correct_mass_flux(dPressure)
				velocityNp1.increment(grad_dPressure.get_data(), scale=-alphaV*dt)
				massFluxAlg.compute_mass_flux()
				normDivU = np.linalg.norm(pressureSystem.get_rhs())
				if (normDivU < tolerance and normRHS < tolerance):
					residualDict = myLogger.get_residuals()
					print(f"Momentum and pressure system converged on nonlinear iteration {iter+1}!")
					for system,residual in residualDict.items():
						print(f"\tSystem {system} converged with residual {residual}")
					break
				myLogger.report_log(iter)

				if (normRHS > 1e3 or normDivU > 1e3):
					sys.exit("Residual growth has gotten out of control, killing simulation.")
			timeDiff = velocityNp1.get_data() - velocityN.get_data()
			velocityNp1.copy_to(velocityN)
			print(f"max velocity = {max(velocityNp1.get_data())}")
			print(f"norm time diff = {np.linalg.norm(timeDiff)}")
			print("-------------------------------------------------------------------------------------------")
			# plotting..
			if ((t+1) % timeOutputFreq == 0 ):
				xMesh = mesh.get_x_coordinates()
				yMesh = mesh.get_y_coordinates()
				X,Y = np.meshgrid(xMesh[:-1],yMesh[:-1])
				P_grid = pressureField.get_data().reshape(cells_y,cells_x)
				u_flat = velocityNp1.get_data()[0::2]  # All x-components: [u1_x, u2_x, ...]
				v_flat = velocityNp1.get_data()[1::2]  # All y-components: [u1_y, u2_y, ...]
				U = u_flat.reshape((cells_y,cells_x))
				V = v_flat.reshape((cells_y,cells_x))
				fig, ax = plt.subplots(figsize=(8, 6))
				p_min=-0.25
				p_max=0.25
#				cp = ax.contourf(X,Y,P_grid,cmap='viridis',alpha=0.8)
				cp = ax.contourf(X,Y,np.sqrt(U**2 + V**2),cmap='viridis',alpha=0.8)
				cbar = plt.colorbar(cp)
				ax.quiver(X, Y, U, V, color='white')
				plt.title(f'Time = {t*dt}')
	#speed = np.sqrt(U**2 + V**2)
	#ax.streamplot(X, Y, U, V, color='white', linewidth=1, density=1.2)
				plt.savefig(os.path.join(os.getcwd(),directoryName, f'pressure_and_velocity_at_{t}.png'))
				plt.close()

