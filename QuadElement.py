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
        # left_cell_idx and right_cell_idx are expected to be flattened integer IDs or None
        self.leftCellFlatId_ = left_cell_idx
        self.rightCellFlatId_ = right_cell_idx
        self.faceCoords_ = face_coords
        self.normalVector_ = normal_vector
        self.area_ = None  # Will be computed based on mesh spacing
        self.isBoundary_ = right_cell_idx is None
    
    def set_area(self, area):
        """Set the face area (length in 2D)"""
        self.area_ = area
    
    def get_left_cell(self):
        """Get left cell index"""
        return self.leftCellFlatId_
    
    def get_right_cell(self):
        """Get right cell index"""
        return self.rightCellFlatId_
    
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
            f"left={self.leftCellFlatId_}, "
            f"right={self.rightCellFlatId_}, "
            f"coords={self.faceCoords_}, "
            f"normal={self.normalVector_}, "
            f"boundary={self.isBoundary_})"
        )


class Cell:
    """
    Represents a cell in a finite volume mesh.
    Stores a flattened array ID and the cell volume. Optional helpers
    allow storing (i,j) indices and centroid coordinates.
    """

    def __init__(self, flat_id, volume, indices=None, centroid=None):
        """
        Initialize a Cell.

        Args:
            flat_id (int): Flattened cell ID (e.g., row-major index)
            volume (float): Cell volume (area in 2D)
            indices (tuple): Optional (i, j) integer indices
            centroid (tuple): Optional (x, y) centroid coordinates
        """
        self.flatId_ = int(flat_id)
        self.volume_ = float(volume)
        self.indices_ = tuple(indices) if indices is not None else None
        self.centroid_ = tuple(centroid) if centroid is not None else None

    def get_volume(self):
        return self.volume_

    def get_flat_id(self):
        return self.flatId_

    def set_centroid(self, centroid):
        self.centroid_ = tuple(centroid)

    def get_centroid(self):
        return self.centroid_

    def set_indices(self, indices):
        self.indices_ = tuple(indices)

    def get_indices(self):
        return self.indices_

    def __repr__(self):
        return (
            f"Cell(flatId={self.flatId_}, volume={self.volume_}, "
            f"indices={self.indices_}, centroid={self.centroid_})"
        )


