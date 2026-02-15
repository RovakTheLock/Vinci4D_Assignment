import unittest
import tempfile
import os
import yaml
import numpy as np
from YamlParser import InputConfigParser
from MeshObject import MeshObject
from QuadElement import Face, Cell


class TestMeshObject(unittest.TestCase):
	def setUp(self):
		self.tmpdir = tempfile.mkdtemp()

	def tearDown(self):
		for f in os.listdir(self.tmpdir):
			os.remove(os.path.join(self.tmpdir, f))
		os.rmdir(self.tmpdir)

	def test_3x3_cell_centers(self):
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

		cells = mesh.get_cells()
		nx = parser.numCellsX_
		ny = parser.numCellsY_

		# Expect centers at 1/6, 1/2, 5/6 for range [0,1] with 3 cells
		expected = np.array([1.0/6.0, 0.5, 5.0/6.0])

		# extract x centers from first row (j=0)
		x_centers = np.array([cells[i].get_centroid()[0] for i in range(nx)])
		# extract y centers from first column (i=0) across rows
		y_centers = np.array([cells[j * nx].get_centroid()[1] for j in range(ny)])

		self.assertEqual(x_centers.shape[0], 3)
		self.assertEqual(y_centers.shape[0], 3)
		np.testing.assert_allclose(x_centers, expected, rtol=1e-7, atol=1e-9)
		np.testing.assert_allclose(y_centers, expected, rtol=1e-7, atol=1e-9)

	def test_4x8_unit_square_coords_and_centers(self):
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': 4,
				'num_cells_y': 8
			}
		}
		path = os.path.join(self.tmpdir, 'mesh_4x8.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()

		x_nodes = mesh.get_x_coordinates()
		y_nodes = mesh.get_y_coordinates()
		cells = mesh.get_cells()
		nx = parser.numCellsX_
		ny = parser.numCellsY_

		# Expected node coordinates
		expected_x_nodes = np.linspace(0.0, 1.0, 4 + 1)  # 5 nodes: 0,0.25,0.5,0.75,1
		expected_y_nodes = np.linspace(0.0, 1.0, 8 + 1)  # 9 nodes: spacing 1/8

		# Expected cell centers are midpoints between nodes
		expected_x_centers = (expected_x_nodes[:-1] + expected_x_nodes[1:]) / 2
		expected_y_centers = (expected_y_nodes[:-1] + expected_y_nodes[1:]) / 2

		# extract x centers from first row (j=0)
		x_centers = np.array([cells[i].get_centroid()[0] for i in range(nx)])
		# extract y centers from first column (i=0) across rows
		y_centers = np.array([cells[j * nx].get_centroid()[1] for j in range(ny)])

		self.assertEqual(x_nodes.shape[0], expected_x_nodes.shape[0])
		self.assertEqual(y_nodes.shape[0], expected_y_nodes.shape[0])
		self.assertEqual(x_centers.shape[0], expected_x_centers.shape[0])
		self.assertEqual(y_centers.shape[0], expected_y_centers.shape[0])

		np.testing.assert_allclose(x_nodes, expected_x_nodes, rtol=1e-12, atol=1e-12)
		np.testing.assert_allclose(y_nodes, expected_y_nodes, rtol=1e-12, atol=1e-12)
		np.testing.assert_allclose(x_centers, expected_x_centers, rtol=1e-12, atol=1e-12)
		np.testing.assert_allclose(y_centers, expected_y_centers, rtol=1e-12, atol=1e-12)

	def test_3x6_faces(self):
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': 3,
				'num_cells_y': 6
			}
		}
		path = os.path.join(self.tmpdir, 'mesh.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_faces()

		faces = mesh.get_faces()
		internal = mesh.get_internal_faces()
		boundary = mesh.get_boundary_faces()

		# For 3x6: compute expected counts
		# vertical internal = (Nx-1)*Ny = (3-1)*6 = 12
		# horizontal internal = Nx*(Ny-1) = 3*(6-1) = 15
		# internal = 12 + 15 = 27
		# boundary = 2*Ny + 2*Nx = 2*6 + 2*3 = 18
		# total = 27 + 18 = 45
		self.assertEqual(len(internal), 27)
		self.assertEqual(len(boundary), 18)
		self.assertEqual(len(faces), 45)

		x_min, x_max = parser.xRange_
		y_min, y_max = parser.yRange_
		nx = parser.numCellsX_
		ny = parser.numCellsY_
		dx = (x_max - x_min) / nx
		dy = (y_max - y_min) / ny

		# Check internal faces have both left and right; boundary faces have right None
		for f in internal:
			left = f.get_left_cell()
			right = f.get_right_cell()
			# should be integers (flattened ids) within valid range
			self.assertIsInstance(left, int)
			self.assertIsInstance(right, int)
			self.assertTrue(0 <= left < (nx * ny))
			self.assertTrue(0 <= right < (nx * ny))
			n = f.get_normal_vector()
			area = f.get_area()
			if abs(n[0]) > 0:
				self.assertTrue(np.isclose(area, dy))
			else:
				self.assertTrue(np.isclose(area, dx))

		for f in boundary:
			left = f.get_left_cell()
			right = f.get_right_cell()
			# left should be flattened int, right should be None
			self.assertIsInstance(left, int)
			self.assertIsNone(right)
			self.assertTrue(0 <= left < (nx * ny))
			n = f.get_normal_vector()
			area = f.get_area()
			if abs(n[0]) > 0:
				self.assertTrue(np.isclose(area, dy))
			else:
				self.assertTrue(np.isclose(area, dx))

			# Verify boundary face lies on correct domain edge
			fc = f.get_face_center()
			# vertical face (normal in x) -> x should equal x_min or x_max
			if abs(n[0]) > 0:
				if n[0] < 0:
					self.assertTrue(np.isclose(fc[0], x_min))
				else:
					self.assertTrue(np.isclose(fc[0], x_max))
			# horizontal face (normal in y) -> y should equal y_min or y_max
			else:
				if n[1] < 0:
					self.assertTrue(np.isclose(fc[1], y_min))
				else:
					self.assertTrue(np.isclose(fc[1], y_max))

	def test_10x15_cell_numbering(self):
		# 10x15 unit-square mesh: verify flattened and (i,j) numbering
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': 10,
				'num_cells_y': 15
			}
		}
		path = os.path.join(self.tmpdir, 'mesh_10x15.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()

		cells = mesh.get_cells()
		nx = parser.numCellsX_
		ny = parser.numCellsY_

		self.assertEqual(len(cells), nx * ny)

		# Verify every cell's flattened id and indices mapping (row-major)
		for j in range(ny):
			for i in range(nx):
				flat = j * nx + i
				cell = cells[flat]
				self.assertEqual(cell.get_flat_id(), flat)
				self.assertEqual(cell.get_indices(), (i, j))
				# also basic type checks
				self.assertIsInstance(cell.get_flat_id(), int)
				self.assertIsInstance(cell.get_volume(), float)

	def test_5x5_interior_cells(self):
		# 5x5 mesh: verify interior cells (3x3 = 9 interior cells)
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': 5,
				'num_cells_y': 5
			}
		}
		path = os.path.join(self.tmpdir, 'mesh_5x5_interior.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()

		interior = mesh.get_interior_cells()
		nx = parser.numCellsX_
		ny = parser.numCellsY_

		# For 5x5 grid, interior = (5-2) x (5-2) = 3x3 = 9
		self.assertEqual(len(interior), (nx - 2) * (ny - 2))

		# Verify all interior cells have 1 <= i <= 3 and 1 <= j <= 3
		for cell in interior:
			i, j = cell.get_indices()
			self.assertTrue(1 <= i <= 3)
			self.assertTrue(1 <= j <= 3)

	def test_5x5_boundary_cells(self):
		# 5x5 mesh: verify boundary cells (perimeter = 16 boundary cells)
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': 5,
				'num_cells_y': 5
			}
		}
		path = os.path.join(self.tmpdir, 'mesh_5x5_boundary.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()

		boundary = mesh.get_boundary_cells()
		nx = parser.numCellsX_
		ny = parser.numCellsY_

		# Total - interior = 25 - 9 = 16 boundary cells
		expected_boundary_count = (nx * ny) - ((nx - 2) * (ny - 2))
		self.assertEqual(len(boundary), expected_boundary_count)

		# Verify all boundary cells have at least one index on edge (0 or nx-1 or ny-1)
		for cell in boundary:
			i, j = cell.get_indices()
			on_edge = (i == 0 or i == nx - 1 or j == 0 or j == ny - 1)
			self.assertTrue(on_edge)

	def test_6x5_get_num_faces(self):

		# 6x5 mesh: verify boundary cells (perimeter = 16 boundary cells)
        # Test the get_num_faces method
		num_cells_x = 6
		num_cells_y = 5
		cfg = {
			'mesh_parameters': {
				'x_range': [0, 1],
				'y_range': [0, 1],
				'num_cells_x': num_cells_x,
				'num_cells_y': num_cells_y 
			}
		}
		path = os.path.join(self.tmpdir, 'mesh_6x5_boundary.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()

        # Calculate expected unique faces
		total_faces = 4 * (num_cells_x * num_cells_y)
		horizontal_shared_faces = (num_cells_x - 1) * num_cells_y
		vertical_shared_faces = (num_cells_y - 1) * num_cells_x
		expected_faces = total_faces - (horizontal_shared_faces + vertical_shared_faces)

		self.assertEqual(mesh.get_num_faces(), expected_faces)

	def test_10x5_boundaries(self):
		# 5x5 mesh: verify boundary cells (perimeter = 16 boundary cells)
		numCellsX_ = 10
		numCellsY_ = 5
		xRange = [0,1]
		yRange = [0,1]
		cfg = {
			'mesh_parameters': {
				'x_range': xRange,
				'y_range': yRange,
				'num_cells_x': numCellsX_,
				'num_cells_y': numCellsY_
			}
		}
		path = os.path.join(self.tmpdir, f'mesh_{numCellsX_}_{numCellsY_}_left_boundary.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		mesh.generate_grid()
		leftBoundaryFaces = mesh.get_left_boundary()
		rightBoundaryFaces = mesh.get_right_boundary()
		topBoundaryFaces = mesh.get_top_boundary()
		bottomBoundaryFaces = mesh.get_bottom_boundary()

		self.assertEqual(len(leftBoundaryFaces), numCellsY_)
		self.assertEqual(len(rightBoundaryFaces), numCellsY_)
		self.assertEqual(len(topBoundaryFaces), numCellsX_)
		self.assertEqual(len(bottomBoundaryFaces), numCellsX_)

		for f in leftBoundaryFaces:
			faceCentroid = f.get_face_center()
			self.assertEqual(faceCentroid[0], 0.0) # Left boundary faces should be at x=0
			self.assertTrue(yRange[0] <= faceCentroid[1] <= yRange[1]) # y should be within domain bounds

		for f in rightBoundaryFaces:
			faceCentroid = f.get_face_center()
			self.assertEqual(faceCentroid[0], 1.0) # Right boundary faces should be at x=1
			self.assertTrue(yRange[0] <= faceCentroid[1] <= yRange[1]) # y should be within domain bounds

		for f in topBoundaryFaces:
			faceCentroid = f.get_face_center()
			self.assertEqual(faceCentroid[1], 1.0) # Top boundary faces should be at y=1
			self.assertTrue(xRange[0] <= faceCentroid[0] <= xRange[1]) # x should be within domain bounds

		for f in bottomBoundaryFaces:
			faceCentroid = f.get_face_center()
			self.assertEqual(faceCentroid[1], 0.0) # Bottom boundary faces should be at y=0
			self.assertTrue(xRange[0] <= faceCentroid[0] <= xRange[1]) # x should be within domain bounds

	def test_get_x_range(self):
		xRange = [-2.5, 3.0]
		cfg = {
			'mesh_parameters': {
				'x_range': xRange,
				'y_range': [0.0, 1.0],
				'num_cells_x': 4,
				'num_cells_y': 4
			}
		}
		path = os.path.join(self.tmpdir, 'mesh_get_x_range.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		self.assertEqual(mesh.get_x_range(), xRange)

	def test_get_y_range(self):
		yRange = [-1.0, 2.0]
		cfg = {
			'mesh_parameters': {
				'x_range': [0.0, 1.0],
				'y_range': yRange,
				'num_cells_x': 4,
				'num_cells_y': 4
			}
		}
		path = os.path.join(self.tmpdir, 'mesh_get_y_range.yaml')
		with open(path, 'w') as f:
			yaml.safe_dump(cfg, f)

		parser = InputConfigParser(path)
		mesh = MeshObject(parser)
		self.assertEqual(mesh.get_y_range(), yRange)



if __name__ == '__main__':
	unittest.main()



