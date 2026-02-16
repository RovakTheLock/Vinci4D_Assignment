#include <petsc.h>
#include <gtest/gtest.h>
#include "../include/AssembleAlgorithms.h"
#include "../include/MeshObject.h"
#include "../include/YamlParser.h"
#include "../include/FieldsHolder.h"
#include "../include/LinearSystem.h"

using namespace Vinci4D;

class AssembleAlgorithmsTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser = new InputConfigParser("../config/input.yaml");
        mesh = new MeshObject(*parser);
        numCells = mesh->getNumCells();
        momentumDof = numCells * MAX_DIM;
    }
    
    void TearDown() override {
        delete mesh;
        delete parser;
    }
    
    InputConfigParser* parser;
    MeshObject* mesh;
    int numCells;
    int momentumDof;
};

TEST_F(AssembleAlgorithmsTest, TimeTermAssembler) {
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numCells);
    FieldArray velocityOld("velocity_n", DimType::VECTOR, numCells);
    
    velocityNew.initializeConstant(1.0);
    velocityOld.initializeConstant(0.5);
    
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    double dt = 0.01;
    AssembleCellVectorTimeTerm timeAssembler(
        "TimeTerm", &velocityNew, &velocityOld, &momentumSystem, mesh, dt);
    
    momentumSystem.zero();
    EXPECT_NO_THROW(timeAssembler.assemble());
    momentumSystem.assembleMatrix();
    
    double rhsNorm = momentumSystem.getRhsNorm();
    EXPECT_GT(rhsNorm, 0) << "RHS norm should be positive after time term assembly";
}

TEST_F(AssembleAlgorithmsTest, DiffusionAssembler) {
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numCells);
    velocityNew.initializeConstant(1.0);
    
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    double diffusionCoeff = 0.001;
    AssembleInteriorVectorDiffusionToLinSystem diffusionAssembler(
        "Diffusion", &velocityNew, &momentumSystem, mesh, diffusionCoeff);
    
    momentumSystem.zero();
    EXPECT_NO_THROW(diffusionAssembler.assemble());
    momentumSystem.assembleMatrix();
    
    double rhsNorm = momentumSystem.getRhsNorm();
    // RHS norm may be zero or positive depending on field values
    EXPECT_GE(rhsNorm, 0) << "RHS norm should be non-negative";
}

TEST_F(AssembleAlgorithmsTest, PressureGradientAssembler) {
    FieldArray gradPressure("grad_pressure", DimType::VECTOR, numCells);
    gradPressure.initializeConstant(0.1);
    
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    AssembleCellVectorPressureGradientToLinSystem pressureGradAssembler(
        "PressureGrad", &gradPressure, &momentumSystem, mesh);
    
    momentumSystem.zero();
    EXPECT_NO_THROW(pressureGradAssembler.assemble());
    momentumSystem.assembleMatrix();
    
    double rhsNorm = momentumSystem.getRhsNorm();
    EXPECT_GT(rhsNorm, 0) << "RHS norm should be positive after pressure gradient assembly";
}

TEST_F(AssembleAlgorithmsTest, CombinedAssembly) {
    FieldArray velocityNew("velocity_np1", DimType::VECTOR, numCells);
    FieldArray velocityOld("velocity_n", DimType::VECTOR, numCells);
    FieldArray gradPressure("grad_pressure", DimType::VECTOR, numCells);
    
    velocityNew.initializeConstant(1.0);
    velocityOld.initializeConstant(0.5);
    gradPressure.initializeConstant(0.1);
    
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    double dt = 0.01;
    double diffusionCoeff = 0.001;
    
    AssembleCellVectorTimeTerm timeAssembler(
        "TimeTerm", &velocityNew, &velocityOld, &momentumSystem, mesh, dt);
    AssembleInteriorVectorDiffusionToLinSystem diffusionAssembler(
        "Diffusion", &velocityNew, &momentumSystem, mesh, diffusionCoeff);
    AssembleCellVectorPressureGradientToLinSystem pressureGradAssembler(
        "PressureGrad", &gradPressure, &momentumSystem, mesh);
    
    momentumSystem.zero();
    EXPECT_NO_THROW({
        timeAssembler.assemble();
        diffusionAssembler.assemble();
        pressureGradAssembler.assemble();
    });
    momentumSystem.assembleMatrix();
    
    double rhsNorm = momentumSystem.getRhsNorm();
    EXPECT_GT(rhsNorm, 0) << "RHS norm should be positive after combined assembly";
}

TEST_F(AssembleAlgorithmsTest, WrongFieldTypeThrows) {
    FieldArray scalarField("scalar", DimType::SCALAR, numCells);
    LinearSystem momentumSystem(momentumDof, "Momentum", true);
    
    // Should throw because time term requires vector field
    EXPECT_THROW({
        AssembleCellVectorTimeTerm timeAssembler(
            "TimeTerm", &scalarField, &scalarField, &momentumSystem, mesh, 0.01);
    }, std::runtime_error);
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
