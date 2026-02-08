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
        self.cellsX_ = None
        self.cellsY_ = None
        self.faces_ = None
        self.internalFaces_ = None
        self.boundaryFaces_ = None
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
        
        # Generate cell centers
        self.cellsX_ = (self.xCoords_[:-1] + self.xCoords_[1:]) / 2
        self.cellsY_ = (self.yCoordS_[:-1] + self.yCoordS_[1:]) / 2
        
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
    
    def get_cell_centers(self):
        """
        Get the cell center coordinates.
        
        Returns:
            tuple: (cells_x, cells_y) - 1D arrays of cell center coordinates
        """
        if self.cellsX_ is None or self.cellsY_ is None:
            self.generate_grid()
        return self.cellsX_, self.cellsY_
    
    def get_cell_count(self):
        """Get the number of cells in x and y"""
        num_cells_x = self.config_parser_.numCellsX_
        num_cells_y = self.config_parser_.numCellsY_
        return num_cells_x, num_cells_y
    
    def generate_faces(self):
        """
        Generate all faces (internal and boundary) for the mesh.
        Internal faces connect two cells, boundary faces have only a left cell.
        """
        if self.cellsX_ is None or self.cellsY_ is None:
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
                y_coord = (self.cellsY_[j-1] + self.cellsY_[j]) / 2 if j > 0 else self.cellsY_[j]
                
                # Interior vertical faces
                if 0 < i < num_cells_x:
                    left_cell = (i - 1, j)
                    right_cell = (i, j)
                    normal = (1.0, 0.0)  # Pointing right
                    face = QE.Face(face_id, left_cell, right_cell, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.internalFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Left boundary faces
                elif i == 0:
                    left_cell = (0, j)
                    normal = (-1.0, 0.0)  # Pointing left (outward)
                    face = QE.Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.boundaryFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Right boundary faces
                elif i == num_cells_x:
                    left_cell = (num_cells_x - 1, j)
                    normal = (1.0, 0.0)  # Pointing right (outward)
                    face = QE.Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.boundaryFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
        
        # Horizontal faces (separating cells in y-direction)
        for i in range(num_cells_x):
            for j in range(num_cells_y + 1):
                x_coord = (self.cellsX_[i-1] + self.cellsX_[i]) / 2 if i > 0 else self.cellsX_[i]
                y_coord = self.yCoordS_[j]
                
                # Interior horizontal faces
                if 0 < j < num_cells_y:
                    left_cell = (i, j - 1)
                    right_cell = (i, j)
                    normal = (0.0, 1.0)  # Pointing up
                    face = QE.Face(face_id, left_cell, right_cell, (x_coord, y_coord), normal)
                    face.set_area(dx)
                    self.internalFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Bottom boundary faces
                elif j == 0:
                    left_cell = (i, 0)
                    normal = (0.0, -1.0)  # Pointing down (outward)
                    face = QE.Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dx)
                    self.boundaryFaces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Top boundary faces
                elif j == num_cells_y:
                    left_cell = (i, num_cells_y - 1)
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
