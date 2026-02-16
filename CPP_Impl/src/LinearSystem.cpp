#include "LinearSystem.h"
#include <iostream>
#include <stdexcept>

namespace Vinci4D {

LinearSystem::LinearSystem(int numDof, const std::string& name, bool sparse)
    : numDof_(numDof), name_(name), assembled_(false) {
    
    // Create PETSc matrix (sparse format)
    MatCreate(PETSC_COMM_WORLD, &lhs_);
    MatSetSizes(lhs_, PETSC_DECIDE, PETSC_DECIDE, numDof_, numDof_);
    MatSetFromOptions(lhs_);
    MatSetUp(lhs_);
    
    // Create PETSc RHS vector
    VecCreate(PETSC_COMM_WORLD, &rhs_);
    VecSetSizes(rhs_, PETSC_DECIDE, numDof_);
    VecSetFromOptions(rhs_);
    
    // Create KSP solver context
    KSPCreate(PETSC_COMM_WORLD, &ksp_);
}

LinearSystem::~LinearSystem() {
    MatDestroy(&lhs_);
    VecDestroy(&rhs_);
    KSPDestroy(&ksp_);
}

void LinearSystem::addLhs(int row, int col, double value) {
    MatSetValue(lhs_, row, col, value, ADD_VALUES);
    assembled_ = false;  // Mark as needing reassembly
}

void LinearSystem::addRhs(int row, double value) {
    VecSetValue(rhs_, row, value, ADD_VALUES);
}

void LinearSystem::zero() {
    // Always destroy and recreate to ensure clean state
    // This avoids issues with pending values in unassembled matrices
    MatDestroy(&lhs_);
    
    // Recreate the matrix with fresh state
    MatCreate(PETSC_COMM_WORLD, &lhs_);
    MatSetSizes(lhs_, PETSC_DECIDE, PETSC_DECIDE, numDof_, numDof_);
    MatSetFromOptions(lhs_);
    MatSetUp(lhs_);
    
    // Initialize all diagonal entries to 0 using ADD_VALUES mode
    // This ensures PETSc's requirement that sparse matrices have all diagonals
    for (int i = 0; i < numDof_; i++) {
        MatSetValue(lhs_, i, i, 0.0, ADD_VALUES);
    }
    
    VecZeroEntries(rhs_);
    assembled_ = false;
}

void LinearSystem::zeroRhs() {
    VecZeroEntries(rhs_);
}

void LinearSystem::assembleMatrix() {
    MatAssemblyBegin(lhs_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(lhs_, MAT_FINAL_ASSEMBLY);
    
    VecAssemblyBegin(rhs_);
    VecAssemblyEnd(rhs_);
    
    assembled_ = true;
}

void LinearSystem::solve(const std::string& method, Vec& solution,
                         double rTol, double aTol, int maxIter) {
    if (!assembled_) {
        assembleMatrix();
    }
    
    // Set solver type based on method
    PC pc;
    if (method == "direct") {
        KSPSetType(ksp_, KSPPREONLY);
        KSPGetPC(ksp_, &pc);
        PCSetType(pc, PCLU);
    } else if (method == "gmres") {
        KSPSetType(ksp_, KSPGMRES);
        KSPGMRESSetRestart(ksp_, 30);
        KSPGetPC(ksp_, &pc);
        PCSetType(pc, PCILU);
        PCFactorSetLevels(pc, 2);
    } else if (method == "bicgstab") {
        KSPSetType(ksp_, KSPBCGS);
        KSPGetPC(ksp_, &pc);
        PCSetType(pc, PCILU);
        PCFactorSetLevels(pc, 2);
    } else if (method == "cg") {
        KSPSetType(ksp_, KSPCG);
        KSPGetPC(ksp_, &pc);
        PCSetType(pc, PCJACOBI);
    } else {
        throw std::runtime_error("Unknown solver method: " + method);
    }
    
    // Set tolerances (both relative and absolute)
    KSPSetTolerances(ksp_, rTol, aTol, PETSC_DEFAULT, maxIter);
    
    // Set operator
    KSPSetOperators(ksp_, lhs_, lhs_);
    KSPSetFromOptions(ksp_);
    
    // Set up the solver (required before solving)
    KSPSetUp(ksp_);
    
    // Solve
    KSPSolve(ksp_, rhs_, solution);
    
    // Check convergence
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp_, &reason);
    
    if (reason < 0) {
        std::cerr << "Warning: KSP did not converge (reason=" << reason << ")" << std::endl;
    }
    
    // Report solver info
    const char* kspType = nullptr;
    KSPGetType(ksp_, &kspType);
    KSPGetPC(ksp_, &pc);
    const char* pcType = nullptr;
    PCGetType(pc, &pcType);
    PetscInt its = 0;
    KSPGetIterationNumber(ksp_, &its);
    PetscReal finalRes = 0.0;
    KSPGetResidualNorm(ksp_, &finalRes);
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "solver: " << (kspType ? kspType : method.c_str()) << std::endl;
    std::cout << "preconditioner: " << (pcType ? pcType : "unknown") << std::endl;
    std::cout << "number of iterations: " << its << std::endl;
    std::cout << "final linear residual: " << finalRes << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

void LinearSystem::solvePressure(Vec& solution, double rTol, int maxIter) {
    if (!assembled_) {
        assembleMatrix();
    }
    
    // Use CG for symmetric pressure Poisson
    KSPSetType(ksp_, KSPCG);
    
    // Use AMG preconditioner (GAMG) optimized for elliptic problems
    PC pc;
    KSPGetPC(ksp_, &pc);
    PCSetType(pc, PCGAMG);
    
    // Set tolerances: use relative tolerance, with loose atol
    KSPSetTolerances(ksp_, rTol, 1e-10, PETSC_DEFAULT, maxIter);
    
    // Set operator
    KSPSetOperators(ksp_, lhs_, lhs_);
    KSPSetFromOptions(ksp_);
    
    // Set up the solver (required before solving)
    KSPSetUp(ksp_);
    
    // Solve
    KSPSolve(ksp_, rhs_, solution);
    
    // Check convergence
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp_, &reason);
    
    if (reason < 0) {
        std::cerr << "Warning: KSP (pressure) did not converge (reason=" << reason << ")" << std::endl;
    }
    
    // Report solver info
    const char* kspType = nullptr;
    KSPGetType(ksp_, &kspType);
    KSPGetPC(ksp_, &pc);
    const char* pcType = nullptr;
    PCGetType(pc, &pcType);
    PetscInt its = 0;
    KSPGetIterationNumber(ksp_, &its);
    PetscReal finalRes = 0.0;
    KSPGetResidualNorm(ksp_, &finalRes);
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "solver: " << (kspType ? kspType : "cg") << std::endl;
    std::cout << "preconditioner: " << (pcType ? pcType : "unknown") << std::endl;
    std::cout << "number of iterations: " << its << std::endl;
    std::cout << "final linear residual: " << finalRes << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

double LinearSystem::getRhsNorm() {
    if (!assembled_) {
        VecAssemblyBegin(rhs_);
        VecAssemblyEnd(rhs_);
    }
    
    PetscReal norm;
    VecNorm(rhs_, NORM_2, &norm);
    return static_cast<double>(norm);
}

void LinearSystem::pinFirstRow() {
    // Pin first row to identity to remove nullspace (for pressure Poisson equation)
    // Set row 0: [1 0 0 ...], rhs[0] = 0
    // This removes the constant pressure nullspace
    
    PetscInt row0 = 0;
    // Clear row 0 except diagonal
    MatZeroRows(lhs_, 1, &row0, 1.0, nullptr, nullptr);
    
    // Set diagonal entry to 1
    MatSetValue(lhs_, 0, 0, 1.0, INSERT_VALUES);
    
    // Set RHS[0] = 0
    VecSetValue(rhs_, 0, 0.0, INSERT_VALUES);
    
    // Assemble changes
    MatAssemblyBegin(lhs_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(lhs_, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(rhs_);
    VecAssemblyEnd(rhs_);
}

} // namespace Vinci4D
