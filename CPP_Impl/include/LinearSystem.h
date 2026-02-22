#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <petsc.h>
#include <string>
#include <vector>

namespace Vinci4D {

/**
 * @brief Linear system solver using PETSc
 * 
 * Manages LHS matrix and RHS vector for finite volume discretization.
 * Uses PETSc Mat and Vec structures for efficient sparse linear algebra.
 */
class LinearSystem {
public:
    /**
     * @brief Construct a LinearSystem
     * 
     * @param numDof Number of degrees of freedom
     * @param name Name of the linear system
     * @param sparse Use sparse matrix format (always true for PETSc)
     */
    LinearSystem(int numDof, const std::string& name, bool sparse = true, int localDof = PETSC_DECIDE);
    
    /**
     * @brief Destructor - cleans up PETSc objects
     */
    ~LinearSystem();
    
    // Getters
    std::string getName() const { return name_; }
    int getNumDof() const { return numDof_; }
    Mat& getLhs() { return lhs_; }
    Vec& getRhs() { return rhs_; }
    
    /**
     * @brief Add a value to the LHS matrix (cumulative sum)
     */
    void addLhs(int row, int col, double value);
    
    /**
     * @brief Add a value to the RHS vector (cumulative sum)
     */
    void addRhs(int row, double value);
    
    /**
     * @brief Zero out both LHS matrix and RHS vector
     */
    void zero();
    
    /**
     * @brief Zero out only the RHS vector
     */
    void zeroRhs();
    
    /**
     * @brief Finalize matrix assembly (required before solving)
     */
    void assembleMatrix();
    
    /**
     * @brief Solve the linear system
     * 
     * @param method Solver method: "direct", "gmres", "bicgstab", "cg"
     * @param solution Output vector for solution
     * @param rTol Relative tolerance for iterative solvers
     * @param aTol Absolute tolerance for iterative solvers
     * @param maxIter Maximum iterations for iterative solvers
     */
    void solve(const std::string& method, Vec& solution, 
               double rTol = 1e-8, double aTol = 1e-10, int maxIter = 1000);
    
    /**
     * @brief Solve pressure Poisson system with optimized parameters
     * Uses CG with AMG preconditioner and loose tolerances
     * 
     * @param solution Output vector for solution
     * @param rTol Relative tolerance (default 1e-4 for intermediate iterations)
     * @param maxIter Maximum iterations
     */
    void solvePressure(Vec& solution, double rTol = 1e-4, int maxIter = 1000);
    
    /**
     * @brief Get RHS residual norm
     */
    double getRhsNorm();
    
    /**
     * @brief Pin first row to identity to remove nullspace (for pressure systems)
     * Sets row 0: [1 0 0 ...], rhs[0] = 0
     */
    void pinFirstRow();

private:
    int numDof_;
    int localDof_;
    std::string name_;
    Mat lhs_;          // PETSc sparse matrix
    Vec rhs_;          // PETSc RHS vector
    KSP ksp_;          // PETSc linear solver context
    bool assembled_;   // Track if matrix has been assembled
};

} // namespace Vinci4D

#endif // LINEAR_SYSTEM_H
