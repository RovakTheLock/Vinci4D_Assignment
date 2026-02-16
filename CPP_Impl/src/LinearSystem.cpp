#include "LinearSystem.h"
#include <iostream>
#include <stdexcept>
#include <mpi.h>

namespace Vinci4D {

namespace {
bool isRootRank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}
}

LinearSystem::LinearSystem(int numDof, const std::string& name, bool sparse, int localDof)
    : numDof_(numDof), localDof_(localDof), name_(name), assembled_(false) {
    
    // Create PETSc matrix for parallel execution
    // PETSc will automatically distribute rows across ranks
    MatCreate(PETSC_COMM_WORLD, &lhs_);
    MatSetSizes(lhs_, localDof_, localDof_, numDof_, numDof_);
    MatSetFromOptions(lhs_);
    MatSetUp(lhs_);
    
    // Preallocate generously for stencil patterns and boundary contributions
    // For 2D: interior cells have ~5-8 neighbors * 2 DOF components = ~10-16 nonzeros
    // Boundary cells and vector systems may have more complex patterns
    // Use 30 as a safe upper bound, allow dynamic allocation if exceeded
    PetscInt prealloc = 30;
    MatSeqAIJSetPreallocation(lhs_, prealloc, nullptr);
    
    // Allow dynamic allocation if preallocation is exceeded
    // This is important for complex assembly patterns
    MatSetOption(lhs_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    
    // Create PETSc RHS vector
    VecCreate(PETSC_COMM_WORLD, &rhs_);
    VecSetSizes(rhs_, localDof_, numDof_);
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
    // Zero out existing matrices without destroying them
    // This preserves PETSc's internal structure and preallocation
    MatZeroEntries(lhs_);
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
        
        // Check if matrix is parallel (mpiaij) or sequential (seqaij)
        PetscInt localSize, globalSize;
        MatGetSize(lhs_, &globalSize, nullptr);
        MatGetLocalSize(lhs_, &localSize, nullptr);
        
        if (localSize < globalSize) {
            // Parallel matrix: use redundant preconditioner (gather to rank 0 and solve)
            PCSetType(pc, PCREDUNDANT);
            // After PCSetUp, configure the redundant solver to use LU
        } else {
            // Sequential matrix: use direct LU
            PCSetType(pc, PCLU);
        }
    } else if (method == "gmres") {
        KSPSetType(ksp_, KSPGMRES);
        KSPGMRESSetRestart(ksp_, 30);
        KSPGetPC(ksp_, &pc);
        PetscInt localSize, globalSize;
        MatGetSize(lhs_, &globalSize, nullptr);
        MatGetLocalSize(lhs_, &localSize, nullptr);
        if (localSize < globalSize) {
            PCSetType(pc, PCJACOBI);
        } else {
            PCSetType(pc, PCILU);
            PCFactorSetLevels(pc, 2);
        }
    } else if (method == "bicgstab") {
        KSPSetType(ksp_, KSPBCGS);
        KSPGetPC(ksp_, &pc);
        PetscInt localSize, globalSize;
        MatGetSize(lhs_, &globalSize, nullptr);
        MatGetLocalSize(lhs_, &localSize, nullptr);
        if (localSize < globalSize) {
            PCSetType(pc, PCJACOBI);
        } else {
            PCSetType(pc, PCILU);
            PCFactorSetLevels(pc, 2);
        }
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
    
    // If using PCREDUNDANT, configure the internal solver to use LU
    if (method == "direct") {
        PetscInt localSize, globalSize;
        MatGetSize(lhs_, &globalSize, nullptr);
        MatGetLocalSize(lhs_, &localSize, nullptr);
        
        if (localSize < globalSize) {
            // Configure redundant solver
            KSP innerksp;
            PC innerpc;
            PCRedundantGetKSP(pc, &innerksp);
            KSPSetType(innerksp, KSPPREONLY);
            KSPGetPC(innerksp, &innerpc);
            PCSetType(innerpc, PCLU);
        }
    }
    
    // Solve
    KSPSolve(ksp_, rhs_, solution);
    
    // Check convergence
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp_, &reason);
    
    if (reason < 0) {
        if (isRootRank()) {
            std::cerr << "Warning: KSP did not converge (reason=" << reason << ")" << std::endl;
        }
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
    if (isRootRank()) {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "solver: " << (kspType ? kspType : method.c_str()) << std::endl;
        std::cout << "preconditioner: " << (pcType ? pcType : "unknown") << std::endl;
        std::cout << "number of iterations: " << its << std::endl;
        std::cout << "final linear residual: " << finalRes << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }
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
        if (isRootRank()) {
            std::cerr << "Warning: KSP (pressure) did not converge (reason=" << reason << ")" << std::endl;
        }
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
    if (isRootRank()) {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "solver: " << (kspType ? kspType : "cg") << std::endl;
        std::cout << "preconditioner: " << (pcType ? pcType : "unknown") << std::endl;
        std::cout << "number of iterations: " << its << std::endl;
        std::cout << "final linear residual: " << finalRes << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
    }
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
