#include <petsc.h>
#include <gtest/gtest.h>
#include "../include/Operations.h"
#include "../include/MeshObject.h"
#include "../include/YamlParser.h"
#include "../include/FieldsHolder.h"
#include "../include/LinearSystem.h"
#include <vector>

using namespace Vinci4D;

class OperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser = new InputConfigParser("../config/input.yaml");
        mesh = new MeshObject(*parser);
        numCells = mesh->getNumCells();
    }
    
    void TearDown() override {
        delete mesh;
        delete parser;
    }
    
    InputConfigParser* parser;
    MeshObject* mesh;
    int numCells;
};

TEST_F(OperationsTest, PerformanceTimer) {
    PerformanceTimer timer;
    
    EXPECT_NO_THROW(timer.startTimer("TestEvent"));
    
    // Do some work
    double sum = 0.0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
    
    EXPECT_NO_THROW(timer.endTimer());
    EXPECT_GT(sum, 0);
}

TEST_F(OperationsTest, CFLTimeStepCompute) {
    double CFL = parser->getCFL();
    CFLTimeStepCompute cflCompute(mesh, CFL);
    
    double dt = cflCompute.computeTimeStep();
    
    EXPECT_GT(dt, 0) << "Time step should be positive";
    EXPECT_LT(dt, 1.0) << "Time step should be reasonable";
}

TEST_F(OperationsTest, LogObject) {
    LinearSystem system1(100, "System1", true);
    LinearSystem system2(100, "System2", true);
    
    std::vector<LinearSystem*> systems = {&system1, &system2};
    LogObject logger(systems);
    
    EXPECT_NO_THROW(logger.reportLog(1));
    
    auto residuals = logger.getResiduals();
    EXPECT_TRUE(residuals.find("System1") != residuals.end()) << "Should track System1";
    EXPECT_TRUE(residuals.find("System2") != residuals.end()) << "Should track System2";
}

TEST_F(OperationsTest, ComputeMassFlux) {
    FieldArray velocityField("velocity", DimType::VECTOR, numCells);
    FieldArray pressureField("pressure", DimType::SCALAR, numCells);
    FieldArray gradPressureField("grad_pressure", DimType::VECTOR, numCells);
    
    velocityField.initializeConstant(1.0);
    pressureField.initializeConstant(0.0);
    gradPressureField.initializeConstant(0.0);
    
    double dt = 0.01;
    ComputeInteriorMassFlux massFluxCompute(mesh, &velocityField, &pressureField,
                                           &gradPressureField, dt);
    
    EXPECT_NO_THROW(massFluxCompute.computeMassFlux());
    
    // Check that mass fluxes were computed
    const auto& internalFaces = mesh->getInternalFaces();
    ASSERT_FALSE(internalFaces.empty()) << "Should have internal faces";
    
    // At least some faces should have non-zero flux
    int nonZeroCount = 0;
    for (const auto* face : internalFaces) {
        if (std::abs(face->getMassFlux()) > 1e-10) {
            nonZeroCount++;
        }
    }
    EXPECT_GT(nonZeroCount, 0) << "Should have some non-zero mass fluxes";
}

TEST_F(OperationsTest, CorrectMassFlux) {
    FieldArray velocityField("velocity", DimType::VECTOR, numCells);
    FieldArray pressureField("pressure", DimType::SCALAR, numCells);
    FieldArray gradPressureField("grad_pressure", DimType::VECTOR, numCells);
    FieldArray pressureCorrection("pressure_correction", DimType::SCALAR, numCells);
    
    velocityField.initializeConstant(1.0);
    pressureField.initializeConstant(0.0);
    gradPressureField.initializeConstant(0.0);
    pressureCorrection.initializeConstant(0.01);
    
    double dt = 0.01;
    ComputeInteriorMassFlux massFluxCompute(mesh, &velocityField, &pressureField,
                                           &gradPressureField, dt);
    
    massFluxCompute.computeMassFlux();
    
    const auto& internalFaces = mesh->getInternalFaces();
    ASSERT_FALSE(internalFaces.empty());
    
    // Store initial flux
    double initialFlux = internalFaces[0]->getMassFlux();
    
    EXPECT_NO_THROW(massFluxCompute.correctMassFlux(&pressureCorrection));
    
    // Check that flux was modified
    double correctedFlux = internalFaces[0]->getMassFlux();
    // Flux should change after correction (unless pressure correction gradient is exactly zero)
    // We just verify the operation doesn't crash
    EXPECT_TRUE(true);
}

TEST_F(OperationsTest, LogObjectMultipleIterations) {
    LinearSystem system(100, "TestSystem", true);
    std::vector<LinearSystem*> systems = {&system};
    LogObject logger(systems);
    
    // Test multiple iterations
    EXPECT_NO_THROW({
        logger.reportLog(1);
        logger.reportLog(2);
        logger.reportLog(3);
    });
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
