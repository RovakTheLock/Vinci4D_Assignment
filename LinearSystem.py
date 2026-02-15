import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve, gmres, bicgstab, LinearOperator, splu

class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, rk):
        self.niter += 1

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
        self.ilu_ = None
        
        if sparse:
            self.lhs_data = []  # Store (row, col, value) tuples
            self.lhs = None  # Will be built on demand
        else:
            self.lhs = np.zeros((num_dof, num_dof))
        
        self.rhs = np.zeros(num_dof)
    def get_name(self):
        """Returns assigned name of linear system"""
        return self.name_
    
    def add_lhs(self, row, col, value):
        """Add a value to the LHS matrix (cumulative sum)."""
        if self.sparse_:
            self.lhs_data.append((row, col, value))
            self.lhs = None  # Invalidate cached matrix
        else:
            self.lhs[row, col] += value
    
    def add_rhs(self, row, value):
        """Add a value to the RHS vector (cumulative sum)."""
        self.rhs[row] += value
    
    def zero(self):
        """Zero out both LHS matrix and RHS vector."""
        if self.sparse_:
            self.lhs_data = []
            self.lhs = None
        else:
            self.lhs.fill(0.0)
        self.rhs.fill(0.0)

    def zero_rhs(self):
        self.rhs.fill(0.0)
    
    def get_lhs(self):
        """Get LHS matrix in appropriate format."""
        if self.sparse_:
            if self.lhs is None:
                rows, cols, vals = zip(*self.lhs_data) if self.lhs_data else ([], [], [])
                self.lhs = csr_matrix((vals, (rows, cols)), shape=(self.numDof_, self.numDof_))
            return self.lhs
        return self.lhs

    def get_rhs(self):
        """Get RHS vector."""
        return self.rhs
    def cache_lu_preconditioner(self):
        self.ilu_ = splu(self.get_lhs())  # Precompute ILU preconditioner for iterative solvers
    
    def solve(self, method='direct', **kwargs):
        """
        Solve the linear system.
        
        Args:
            method (str): 'direct', 'gmres', or 'bicgstab'
            **kwargs: Extra arguments passed to the solver (e.g., tol, restart, M)
        """
        lhs = self.get_lhs()
        
        if method == 'direct':
            if self.sparse_:
                return spsolve(lhs, self.rhs)
            else:
                return solve(lhs, self.rhs)
        
        elif method == 'gmres':
            counter = IterationCounter()
            # gmres returns (x, info); info == 0 means convergence
            if self.ilu_ == None:
                self.ilu_ = splu(self.get_lhs())  # Precompute ILU preconditioner for iterative solvers
            defaultPreconditioner = LinearOperator(self.lhs.shape,self.ilu_.solve)
            defaultArgs = {'rtol': 1e-8, 'restart': None, 'M': defaultPreconditioner}
            if kwargs is not None:
                defaultArgs.update(kwargs)
            x, info = gmres(lhs, self.rhs, **defaultArgs)
            if info != 0:
                print(f"Warning: GMRES did not converge (info={info})")
            return x
            
        elif method == 'bicgstab':
            x, info = bicgstab(lhs, self.rhs,**kwargs)
            return x
            
        else:
            raise ValueError(f"Unknown solver method: {method}")

    
    def __repr__(self):
        return f"LinearSystem(name='{self.name_}', # dof={self.numDof_})\nLHS:\n{self.get_lhs()}\nRHS:\n{self.rhs}"