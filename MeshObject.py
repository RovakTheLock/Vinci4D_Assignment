import numpy as np
from YamlParser import InputConfigParser


class Face:
    """
    Represents a face in a finite volume mesh.
    Stores references to left and right cell centroids for flux computations.
    """
    
    def __init__(self, face_id, left_cell_idx, right_cell_idx, face_coords, normal_vector=None):
        """
        Initialize a Face.
        
        Args:
            face_id (int): Unique identifier for the face
            left_cell_idx (tuple): Index of left cell (i, j)
            right_cell_idx (tuple): Index of right cell (i, j), None for boundary faces
            face_coords (tuple): Midpoint coordinates of the face (x, y)
            normal_vector (tuple): Outward normal vector (nx, ny), None for internal faces
        """
        self.id_ = face_id
        self.left_cell_idx_ = left_cell_idx
        self.right_cell_idx_ = right_cell_idx
        self.face_coords_ = face_coords
        self.normal_vector_ = normal_vector
        self.area_ = None  # Will be computed based on mesh spacing
        self.is_boundary_ = right_cell_idx is None
    
    def set_area(self, area):
        """Set the face area (length in 2D)"""
        self.area_ = area
    
    def get_left_cell(self):
        """Get left cell index"""
        return self.left_cell_idx_
    
    def get_right_cell(self):
        """Get right cell index"""
        return self.right_cell_idx_
    
    def get_face_center(self):
        """Get face center coordinates"""
        return self.face_coords_
    
    def get_normal_vector(self):
        """Get outward normal vector"""
        return self.normal_vector_
    
    def get_area(self):
        """Get face area"""
        return self.area_
    
    def is_boundary_face(self):
        """Check if this is a boundary face"""
        return self.is_boundary_
    
    def __repr__(self):
        return (
            f"Face(id={self.id_}, "
            f"left={self.left_cell_idx_}, "
            f"right={self.right_cell_idx_}, "
            f"coords={self.face_coords_}, "
            f"boundary={self.is_boundary_})"
        )


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
        self.x_coords_ = None
        self.y_coords_ = None
        self.grid_x_ = None
        self.grid_y_ = None
        self.cells_x_ = None
        self.cells_y_ = None
        self.faces_ = None
        self.internal_faces_ = None
        self.boundary_faces_ = None
    
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
        self.x_coords_ = np.linspace(x_min, x_max, num_cells_x + 1)
        self.y_coords_ = np.linspace(y_min, y_max, num_cells_y + 1)
        
        # Generate 2D mesh grids
        self.grid_x_, self.grid_y_ = np.meshgrid(self.x_coords_, self.y_coords_, indexing='ij')
        
        # Generate cell centers
        self.cells_x_ = (self.x_coords_[:-1] + self.x_coords_[1:]) / 2
        self.cells_y_ = (self.y_coords_[:-1] + self.y_coords_[1:]) / 2
        
        print(f"Grid generated: {num_cells_x} x {num_cells_y} cells")
    
    def get_x_coordinates(self):
        """Get 1D array of x node coordinates"""
        if self.x_coords_ is None:
            self.generate_grid()
        return self.x_coords_
    
    def get_y_coordinates(self):
        """Get 1D array of y node coordinates"""
        if self.y_coords_ is None:
            self.generate_grid()
        return self.y_coords_
    
    def get_grid(self):
        """
        Get the 2D mesh grids.
        
        Returns:
            tuple: (grid_x, grid_y) - 2D arrays of node coordinates
        """
        if self.grid_x_ is None or self.grid_y_ is None:
            self.generate_grid()
        return self.grid_x_, self.grid_y_
    
    def get_cell_centers(self):
        """
        Get the cell center coordinates.
        
        Returns:
            tuple: (cells_x, cells_y) - 1D arrays of cell center coordinates
        """
        if self.cells_x_ is None or self.cells_y_ is None:
            self.generate_grid()
        return self.cells_x_, self.cells_y_
    
    def get_grid_shape(self):
        """Get the shape of the grid (num_nodes_x, num_nodes_y)"""
        if self.grid_x_ is None:
            self.generate_grid()
        return self.grid_x_.shape
    
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
        if self.cells_x_ is None or self.cells_y_ is None:
            self.generate_grid()
        
        num_cells_x = self.config_parser_.numCellsX_
        num_cells_y = self.config_parser_.numCellsY_
        
        # Cell spacing
        dx = (self.config_parser_.xRange_[1] - self.config_parser_.xRange_[0]) / num_cells_x
        dy = (self.config_parser_.yRange_[1] - self.config_parser_.yRange_[0]) / num_cells_y
        
        self.faces_ = []
        self.internal_faces_ = []
        self.boundary_faces_ = []
        face_id = 0
        
        # Vertical faces (separating cells in x-direction)
        for i in range(num_cells_x + 1):
            for j in range(num_cells_y):
                x_coord = self.x_coords_[i]
                y_coord = (self.cells_y_[j-1] + self.cells_y_[j]) / 2 if j > 0 else self.cells_y_[j]
                
                # Interior vertical faces
                if 0 < i < num_cells_x:
                    left_cell = (i - 1, j)
                    right_cell = (i, j)
                    normal = (1.0, 0.0)  # Pointing right
                    face = Face(face_id, left_cell, right_cell, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.internal_faces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Left boundary faces
                elif i == 0:
                    left_cell = (0, j)
                    normal = (-1.0, 0.0)  # Pointing left (outward)
                    face = Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.boundary_faces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Right boundary faces
                elif i == num_cells_x:
                    left_cell = (num_cells_x - 1, j)
                    normal = (1.0, 0.0)  # Pointing right (outward)
                    face = Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dy)
                    self.boundary_faces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
        
        # Horizontal faces (separating cells in y-direction)
        for i in range(num_cells_x):
            for j in range(num_cells_y + 1):
                x_coord = (self.cells_x_[i-1] + self.cells_x_[i]) / 2 if i > 0 else self.cells_x_[i]
                y_coord = self.y_coords_[j]
                
                # Interior horizontal faces
                if 0 < j < num_cells_y:
                    left_cell = (i, j - 1)
                    right_cell = (i, j)
                    normal = (0.0, 1.0)  # Pointing up
                    face = Face(face_id, left_cell, right_cell, (x_coord, y_coord), normal)
                    face.set_area(dx)
                    self.internal_faces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Bottom boundary faces
                elif j == 0:
                    left_cell = (i, 0)
                    normal = (0.0, -1.0)  # Pointing down (outward)
                    face = Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dx)
                    self.boundary_faces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
                
                # Top boundary faces
                elif j == num_cells_y:
                    left_cell = (i, num_cells_y - 1)
                    normal = (0.0, 1.0)  # Pointing up (outward)
                    face = Face(face_id, left_cell, None, (x_coord, y_coord), normal)
                    face.set_area(dx)
                    self.boundary_faces_.append(face)
                    self.faces_.append(face)
                    face_id += 1
        
        print(f"Generated {len(self.faces_)} faces: "
              f"{len(self.internal_faces_)} internal, {len(self.boundary_faces_)} boundary")
    
    def get_faces(self):
        """Get all faces"""
        if self.faces_ is None:
            self.generate_faces()
        return self.faces_
    
    def get_internal_faces(self):
        """Get internal faces only"""
        if self.internal_faces_ is None:
            self.generate_faces()
        return self.internal_faces_
    
    def get_boundary_faces(self):
        """Get boundary faces only"""
        if self.boundary_faces_ is None:
            self.generate_faces()
        return self.boundary_faces_
    
    def __repr__(self):
        """String representation of MeshObject"""
        shape = self.get_grid_shape() if self.grid_x_ is not None else "Not generated"
        cells_x, cells_y = self.get_cell_count()
        num_faces = len(self.faces_) if self.faces_ is not None else 0
        return (
            f"MeshObject(\n"
            f"  Grid Shape: {shape},\n"
            f"  Cells: {cells_x} x {cells_y},\n"
            f"  Faces: {num_faces},\n"
            f"  X Range: {self.config_parser_.xRange_},\n"
            f"  Y Range: {self.config_parser_.yRange_}\n"
            f")"
        )
