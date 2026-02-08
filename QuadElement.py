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
        self.leftCellIdx_ = left_cell_idx
        self.rightCellIdx_ = right_cell_idx
        self.faceCoords_ = face_coords
        self.normalVector_ = normal_vector
        self.area_ = None  # Will be computed based on mesh spacing
        self.isBoundary_ = right_cell_idx is None
    
    def set_area(self, area):
        """Set the face area (length in 2D)"""
        self.area_ = area
    
    def get_left_cell(self):
        """Get left cell index"""
        return self.leftCellIdx_
    
    def get_right_cell(self):
        """Get right cell index"""
        return self.rightCellIdx_
    
    def get_face_center(self):
        """Get face center coordinates"""
        return self.faceCoords_
    
    def get_normal_vector(self):
        """Get outward normal vector"""
        return self.normalVector_
    
    def get_area(self):
        """Get face area"""
        return self.area_
    
    def is_boundary_face(self):
        """Check if this is a boundary face"""
        return self.isBoundary_
    
    def __repr__(self):
        return (
            f"Face(id={self.id_}, "
            f"left={self.leftCellIdx_}, "
            f"right={self.rightCellIdx_}, "
            f"coords={self.faceCoords_}, "
            f"boundary={self.isBoundary_})"
        )


