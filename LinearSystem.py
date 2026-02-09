import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve

class LinearSystem:
    def __init__(self, num_dof, name, sparse=False):
        """
        Initialize a linear system with LHS matrix and RHS vector.
        
        Args:
            num_dof (int): Number of degrees of freedom
            name (str): Name of the linear system
            sparse (bool): Use sparse matrix format for large systems
        """
        self.numDof_ = num_dof
        self.name_ = name
        self.sparse_ = sparse
        
        if sparse:
            self.lhs_data = []  # Store (row, col, value) tuples
            self.lhs = None  # Will be built on demand
        else:
            self.lhs = np.zeros((num_dof, num_dof))
        
        self.rhs = np.zeros(num_dof)
    
    def set_lhs(self, row, col, value):
        """Set a value in the LHS matrix."""
        if self.sparse_:
            self.lhs_data.append((row, col, value))
            self.lhs = None  # Invalidate cached matrix
        else:
            self.lhs[row, col] = value
    
    def set_rhs(self, row, value):
        """Set a value in the RHS vector."""
        self.rhs[row] = value
    
    def get_lhs(self):
        """Get LHS matrix in appropriate format."""
        if self.sparse_:
            if self.lhs is None:
                rows, cols, vals = zip(*self.lhs_data) if self.lhs_data else ([], [], [])
                self.lhs = csr_matrix((vals, (rows, cols)), shape=(self.numDof_, self.numDof_))
            return self.lhs
        return self.lhs
    
    def solve(self):
        """Solve the linear system and return solution."""
        lhs = self.get_lhs()
        if self.sparse_:
            return spsolve(lhs, self.rhs)
        else:
            return solve(lhs, self.rhs)
    
    def __repr__(self):
        return f"LinearSystem(name='{self.name_}', dof={self.numDof_})"