import numpy as np
import unittest
from LinearSystem import LinearSystem


class TestLinearSystem(unittest.TestCase):
    
    def test_add_lhs_diagonal(self):
        """Test adding 1.0 to each diagonal value in a 3x3 system"""
        num_dof = 3
        system = LinearSystem(num_dof, "test_system", sparse=False)
        
        # Add 1.0 to each diagonal element
        for i in range(num_dof):
            system.add_lhs(i, i, 1.0)
        
        # Verify diagonal values are 1.0
        lhs = system.get_lhs()
        for i in range(num_dof):
            self.assertEqual(lhs[i, i], 1.0)
        
        # Verify off-diagonal values are 0
        for i in range(num_dof):
            for j in range(num_dof):
                if i != j:
                    self.assertEqual(lhs[i, j], 0.0)
    def test_add_lhs_off_diagonal(self):
        """Test adding 2.0 to each off-diagonal value in a 3x3 system"""
        num_dof = 3
        system = LinearSystem(num_dof, "test_system", sparse=False)
        
        # Add 2.0 to each off-diagonal element
        for i in range(num_dof):
            for j in range(num_dof):
                if i != j:
                    system.add_lhs(i, j, 2.0)
        
        # Verify off-diagonal values are 2.0
        lhs = system.get_lhs()
        for i in range(num_dof):
            for j in range(num_dof):
                if i != j:
                    self.assertEqual(lhs[i, j], 2.0)

        # Verify diagonal values are 0.0
        for i in range(num_dof):
            self.assertEqual(lhs[i, i], 0.0)
            
    def test_solve_diagonal_system_direct(self):
        """Create an identity matrix and solve for known values of a 3x3 system."""
        num_dof = 3
        system = LinearSystem(num_dof, "test_system", sparse=False)
        expectedValues = [5.0, -3.0, 2.5]
        
        # Create identity matrix for LHS
        for i in range(num_dof):
            system.add_lhs(i, i, 1.0)
        
        # Set RHS to known values
        for i in range(num_dof):
            system.add_rhs(i, expectedValues[i])
        
        # Solve the system
        solution = system.solve()
        
        # Verify the solution matches the RHS values since LHS is identity
        self.assertAlmostEqual(solution[0], expectedValues[0])
        self.assertAlmostEqual(solution[1], expectedValues[1])
        self.assertAlmostEqual(solution[2], expectedValues[2])
        
    def test_solve_diagonal_system_gmres(self):
        """Create an identity matrix and solve for known values of a 3x3 system using GMRES."""
        num_dof = 3
        system = LinearSystem(num_dof, "test_system", sparse=False)
        expectedValues = [5.0, -3.0, 2.5]
        
        # Create identity matrix for LHS
        for i in range(num_dof):
            system.add_lhs(i, i, 1.0)
        
        # Set RHS to known values
        for i in range(num_dof):
            system.add_rhs(i, expectedValues[i])
        
        # Solve the system
        solution = system.solve(method='gmres')
        
        # Verify the solution matches the RHS values since LHS is identity
        self.assertAlmostEqual(solution[0], expectedValues[0])
        self.assertAlmostEqual(solution[1], expectedValues[1])
        self.assertAlmostEqual(solution[2], expectedValues[2])


if __name__ == '__main__':
    unittest.main()