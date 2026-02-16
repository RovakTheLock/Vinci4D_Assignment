#include <petsc.h>
#include <gtest/gtest.h>
#include "../include/LinearSystem.h"
#include <cmath>

using namespace Vinci4D;

class LinearSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // PETSc is initialized in main()
    }
};

TEST_F(LinearSystemTest, Creation) {
    LinearSystem system(10, "TestSystem", true);
    EXPECT_EQ(system.getNumDof(), 10);
    EXPECT_EQ(system.getName(), "TestSystem");
}

TEST_F(LinearSystemTest, AddToLhsAndRhs) {
    LinearSystem system(10, "TestSystem", true);
    system.zero();
    system.addLhs(0, 0, 1.0);
    system.addLhs(1, 1, 2.0);
    system.addRhs(0, 5.0);
    system.addRhs(1, 10.0);
    
    EXPECT_NO_THROW(system.assembleMatrix());
}

TEST_F(LinearSystemTest, SolveSimpleSystem) {
    // Solve: x = 5, 2y = 10
    LinearSystem system(2, "TestSystem", true);  // Only 2 DOF for this simple system
    system.zero();
    system.addLhs(0, 0, 1.0);
    system.addLhs(1, 1, 2.0);
    system.addRhs(0, 5.0);
    system.addRhs(1, 10.0);
    system.assembleMatrix();
    
    Vec solution;
    VecCreate(PETSC_COMM_WORLD, &solution);
    VecSetSizes(solution, PETSC_DECIDE, 2);
    VecSetFromOptions(solution);
    
    EXPECT_NO_THROW(system.solve("direct", solution));
    
    // Check solution
    PetscInt idx0 = 0, idx1 = 1;
    PetscScalar val0, val1;
    VecGetValues(solution, 1, &idx0, &val0);
    VecGetValues(solution, 1, &idx1, &val1);
    
    EXPECT_NEAR(val0, 5.0, 1e-6) << "x[0] should be 5.0";
    EXPECT_NEAR(val1, 5.0, 1e-6) << "x[1] should be 5.0";
    
    VecDestroy(&solution);
}

TEST_F(LinearSystemTest, ZeroSystem) {
    LinearSystem system(10, "TestSystem", true);
    system.addLhs(0, 0, 1.0);
    system.addRhs(0, 1.0);
    system.assembleMatrix();
    
    system.zero();
    double norm = system.getRhsNorm();
    EXPECT_LT(norm, 1e-10) << "RHS norm should be near zero after zeroing";
}

TEST_F(LinearSystemTest, Solve1DPoisson) {
    // Solve -d²u/dx² = 1, with u(0) = u(1) = 0
    int n = 5;
    LinearSystem poissonSystem(n, "Poisson1D", true);
    poissonSystem.zero();
    
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; ++i) {
        poissonSystem.addLhs(i, i, 2.0 / (h * h));
        if (i > 0) poissonSystem.addLhs(i, i - 1, -1.0 / (h * h));
        if (i < n - 1) poissonSystem.addLhs(i, i + 1, -1.0 / (h * h));
        poissonSystem.addRhs(i, 1.0);
    }
    
    poissonSystem.assembleMatrix();
    
    Vec poissonSolution;
    VecCreate(PETSC_COMM_WORLD, &poissonSolution);
    VecSetSizes(poissonSolution, PETSC_DECIDE, n);
    VecSetFromOptions(poissonSolution);
    
    EXPECT_NO_THROW(poissonSystem.solve("gmres", poissonSolution, 1e-10, 1000));
    
    VecDestroy(&poissonSolution);
}

TEST_F(LinearSystemTest, GetRhsNorm) {
    LinearSystem system(10, "TestSystem", true);
    system.zero();
    system.addRhs(0, 3.0);
    system.addRhs(1, 4.0);
    system.assembleMatrix();
    
    double norm = system.getRhsNorm();
    EXPECT_NEAR(norm, 5.0, 1e-6) << "RHS norm should be 5.0 (3-4-5 triangle)";
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
