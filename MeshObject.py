import numpy as np
from YamlParser import InputConfigParser
import QuadElement as QE


class MeshObject:
    """
    Creates a structured 2D grid based on mesh configuration parameters.
    Takes an InputConfigParser instance and generates grid coordinates.
    """
    
    def __init__(self, config_parser):
        """
        Initialize MeshObject with configuration parser.
        
        Args:
            config_parser (InputConfigParser): Parser containing mesh parameters
        """
        if not isinstance(config_parser, InputConfigParser):
            raise TypeError("config_parser must be an instance of InputConfigParser")
        
        self.config_parser_ = config_parser
        self.xCoords_ = None
        self.yCoordS_ = None
        self.faces_ = None
        self.internalFaces_ = None
        self.boundaryFaces_ = None
        self.cells_ = None
        self.generate_faces()
        self.generate_grid()
    
    def generate_grid(self):
        """
        Generate a structured 2D grid based on configuration parameters.
        Creates 1D coordinate arrays and 2D mesh grid.
        """
        # Ensure configuration is parsed
        if self.config_parser_.xRange_ is None:
            self.config_parser_.parse_mesh_parameters()
        
        # Validate required parameters
        if any(param is None for param in [
            self.config_parser_.xRange_,
            self.config_parser_.yRange_,
            self.config_parser_.numCellsX_,
            self.config_parser_.numCellsY_
        ]):
            raise ValueError("Missing required mesh parameters: xRange, yRange, numCellsX, or numCellsY")
        
        # Extract parameters
        x_min, x_max = self.config_parser_.xRange_
        y_min, y_max = self.config_parser_.yRange_
        num_cells_x = self.config_parser_.numCellsX_
        num_cells_y = self.config_parser_.numCellsY_
        
        # Generate 1D coordinate arrays
        self.xCoords_ = np.linspace(x_min, x_max, num_cells_x + 1)
        self.yCoordS_ = np.linspace(y_min, y_max, num_cells_y + 1)
        
        # Compute cell centers and create Cell objects
        cellsX_centers = (self.xCoords_[:-1] + self.xCoords_[1:]) / 2
        cellsY_centers = (self.yCoordS_[:-1] + self.yCoordS_[1:]) / 2
        dx = self.xCoords_[1] - self.xCoords_[0]
        dy = self.yCoordS_[1] - self.yCoordS_[0]
        cell_volume = dx * dy
        self.cells_ = []
        for j in range(num_cells_y):
            for i in range(num_cells_x):
                flat_id = j * num_cells_x + i
                centroid = (float(cellsX_centers[i]), float(cellsY_centers[j]))
                cell = QE.Cell(flat_id, cell_volume, indices=(i, j), centroid=centroid)
                self.cells_.append(cell)
        
        print(f"Grid generated: {num_cells_x} x {num_cells_y} cells")
    
    def get_x_coordinates(self):
        """Get 1D array of x node coordinates"""
        if self.xCoords_ is None:
            self.generate_grid()
        return self.xCoords_
    
    def get_y_coordinates(self):
        """Get 1D array of y node coordinates"""
        if self.yCoordS_ is None:
            self.generate_grid()
        return self.yCoordS_
    
    def get_cell_count(self):
        """Get the number of cells in x and y"""
        num_cells_x = self.config_parser_.numCellsX_
        num_cells_y = self.config_parser_.numCellsY_
        return num_cells_x, num_cells_y

    def get_cells(self):
        """Return list of Cell objects (row-major order)"""
        if self.cells_ is None:
            self.generate_grid()
        return self.cells_

    def get_cell_by_flat_id(self, flat_id):
        """Return Cell by flattened id or None if out of range"""
        if self.cells_ is None:
            self.generate_grid()
        if 0 <= flat_id < len(self.cells_):
            return self.cells_[flat_id]
        return None
    
    def generate_faces(self):
        """
        Generate all faces (internal and boundary) for the mesh.
        Internal faces connect two cells, boundary faces have only a left cell.
        """
        if self.cells_ is None:
            self.generate_grid()
        
        num_cells_x = self.config_parser_.numCellsX_
        num_cells_y = self.config_parser_.numCellsY_
        
        # Cell spacing
        dx = (self.config_parser_.xRange_[1] - self.config_parser_.xRange_[0]) / num_cells_x
        dy = (self.config_parser_.yRange_[1] - self.config_parser_.yRange_[0]) / num_cells_y
        
        self.faces_ = []
        self.internalFaces_ = []
        self.boundaryFaces_ = []
        face_id = 0
        
        # Vertical faces (separating cells in x-direction)
        for i in range(num_cells_x + 1):
            for j in range(num_cells_y):
                x_coord = self.xCoords_[i]
                # Face center y: use centroid y of the left cell (or available adjacent cell)
                if 0 < i < num_cells_x:
                    left_flat = j * num_cells_x + (i - 1)
                else:
                    left_flat = j * num_cells_x + 0
                y_coord = self.get_cell_by_flat_id(left_flat).get_centroid()[1]
                
                # Interior vertical faces
                if 0 < i < num_cells_x:
                    # flattened ids: flat = j * num_cells_x + i
                    left_cell = j * num_cells_x + (i - 1)
                    right_cell = j * num_cells_x + i
                    normal = (1.0, 0.0)  # Pointing right
                    face = QE.Face(face_id, left_cell, right_cell, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.internalFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Left boundary faces
                elif i == 0:
                    left_cell = j * num_cells_x + 0
                    normal = (-1.0, 0.0)  # Pointing left (outward)
                    face = QE.Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.boundaryFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Right boundary faces
                elif i == num_cells_x:
                    left_cell = j * num_cells_x + (num_cells_x - 1)
                    normal = (1.0, 0.0)  # Pointing right (outward)
                    face = QE.Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.boundaryFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
        
        # Horizontal faces (separating cells in y-direction)
        for i in range(num_cells_x):
            for j in range(num_cells_y + 1):
                # Face center x: use centroid x averaged between neighboring cells when available
                if 0 < j < num_cells_y:
                    left_flat = (j - 1) * num_cells_x + i
                    right_flat = j * num_cells_x + i
                    x_coord = 0.5 * (self.get_cell_by_flat_id(left_flat).get_centroid()[0]
                                     + self.get_cell_by_flat_id(right_flat).get_centroid()[0])
                else:
                    flat = (0 if j == 0 else (num_cells_y - 1)) * num_cells_x + i
                    x_coord = self.get_cell_by_flat_id(flat).get_centroid()[0]
                y_coord = self.yCoordS_[j]
                
                # Interior horizontal faces
                if 0 < j < num_cells_y:
                    left_cell = (j - 1) * num_cells_x + i
                    right_cell = j * num_cells_x + i
                    normal = (0.0, 1.0)  # Pointing up
                    face = QE.Face(face_id, left_cell, right_cell, (x_coord, y_coord), normal)
                    face.set_area(dx)
                    self.internalFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Bottom boundary faces
                elif j == 0:
                    left_cell = 0 * num_cells_x + i
                    normal = (0.0, -1.0)  # Pointing down (outward)
                    face = QE.Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dx)
                    self.boundaryFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Top boundary faces
                elif j == num_cells_y:
                    left_cell = (num_cells_y - 1) * num_cells_x + i
                    normal = (0.0, 1.0)  # Pointing up (outward)
                    face = QE.Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dx)
                    self.boundaryFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
        
        print(f"Generated {len(self.faces_)} faces: "
              f"{len(self.internalFaces_)} internal, {len(self.boundaryFaces_)} boundary")
    
    def get_faces(self):
        """Get all faces"""
        if self.faces_ is None:
            self.generate_faces()
        return self.faces_
    
    def get_internal_faces(self):
        """Get internal faces only"""
        if self.internalFaces_ is None:
            self.generate_faces()
        return self.internalFaces_
    
    def get_boundary_faces(self):
        """Get boundary faces only"""
        if self.boundaryFaces_ is None:
            self.generate_faces()
        return self.boundaryFaces_
    
    def __repr__(self):
        """String representation of MeshObject"""
        cells_x, cells_y = self.get_cell_count()
        num_faces = len(self.faces_) if self.faces_ is not None else 0
        return (
            f"MeshObject(\n"
            f"  Cells: {cells_x} x {cells_y},\n"
            f"  Faces: {num_faces},\n"
            f"  X Range: {self.config_parser_.xRange_},\n"
            f"  Y Range: {self.config_parser_.yRange_}\n"
            f")"
        )
