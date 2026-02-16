#include <petsc.h>
#include <gtest/gtest.h>
#include "../include/FieldsHolder.h"

using namespace Vinci4D;

class FieldsHolderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // PETSc is initialized in main()
    }
};

TEST_F(FieldsHolderTest, CreateScalarField) {
    FieldArray scalarField("pressure", DimType::SCALAR, 100);
    
    EXPECT_EQ(scalarField.getName(), "pressure");
    EXPECT_EQ(scalarField.getType(), DimType::SCALAR);
    EXPECT_EQ(scalarField.getNumComponents(), 1);
    EXPECT_EQ(scalarField.getData().size(), 100);
}

TEST_F(FieldsHolderTest, CreateVectorField) {
    FieldArray vectorField("velocity", DimType::VECTOR, 100);
    
    EXPECT_EQ(vectorField.getName(), "velocity");
    EXPECT_EQ(vectorField.getType(), DimType::VECTOR);
    EXPECT_EQ(vectorField.getNumComponents(), MAX_DIM);
    EXPECT_EQ(vectorField.getData().size(), 100 * MAX_DIM);
}

TEST_F(FieldsHolderTest, InitializeConstant) {
    FieldArray field("test", DimType::SCALAR, 10);
    field.initializeConstant(3.14);
    
    const auto& data = field.getData();
    for (const auto& val : data) {
        EXPECT_DOUBLE_EQ(val, 3.14);
    }
}

TEST_F(FieldsHolderTest, IncrementField) {
    FieldArray field("test", DimType::SCALAR, 10);
    field.initializeConstant(1.0);
    field.increment(2.0, 1.5);  // Add 2.0 * 1.5 = 3.0
    
    const auto& data = field.getData();
    for (const auto& val : data) {
        EXPECT_DOUBLE_EQ(val, 4.0);
    }
}

TEST_F(FieldsHolderTest, CopyToField) {
    FieldArray field1("test1", DimType::SCALAR, 10);
    FieldArray field2("test2", DimType::SCALAR, 10);
    
    field1.initializeConstant(5.0);
    field2.initializeConstant(0.0);
    
    EXPECT_NO_THROW(field1.copyTo(field2));
    
    const auto& data2 = field2.getData();
    for (const auto& val : data2) {
        EXPECT_DOUBLE_EQ(val, 5.0);
    }
}

TEST_F(FieldsHolderTest, SwapFields) {
    FieldArray field1("test1", DimType::SCALAR, 10);
    FieldArray field2("test2", DimType::SCALAR, 10);
    
    field1.initializeConstant(1.0);
    field2.initializeConstant(2.0);
    
    EXPECT_NO_THROW(field1.swapFields(field2));
    
    const auto& data1 = field1.getData();
    const auto& data2 = field2.getData();
    
    for (const auto& val : data1) {
        EXPECT_DOUBLE_EQ(val, 2.0);
    }
    for (const auto& val : data2) {
        EXPECT_DOUBLE_EQ(val, 1.0);
    }
}

TEST_F(FieldsHolderTest, CopyToIncompatibleTypeThrows) {
    FieldArray scalarField("scalar", DimType::SCALAR, 10);
    FieldArray vectorField("vector", DimType::VECTOR, 10);
    
    EXPECT_THROW(scalarField.copyTo(vectorField), std::runtime_error);
}

TEST_F(FieldsHolderTest, CopyToIncompatibleSizeThrows) {
    FieldArray field1("test1", DimType::SCALAR, 10);
    FieldArray field2("test2", DimType::SCALAR, 20);
    
    EXPECT_THROW(field1.copyTo(field2), std::runtime_error);
}

TEST_F(FieldsHolderTest, SwapIncompatibleTypeThrows) {
    FieldArray scalarField("scalar", DimType::SCALAR, 10);
    FieldArray vectorField("vector", DimType::VECTOR, 10);
    
    EXPECT_THROW(scalarField.swapFields(vectorField), std::runtime_error);
}

TEST_F(FieldsHolderTest, FieldNameToString) {
    EXPECT_EQ(fieldNameToString(FieldNames::PRESSURE), "pressure");
    EXPECT_EQ(fieldNameToString(FieldNames::VELOCITY_NEW), "velocity_np1");
    EXPECT_EQ(fieldNameToString(FieldNames::GRAD_PRESSURE), "grad_pressure");
}

TEST_F(FieldsHolderTest, VectorFieldComponents) {
    FieldArray vectorField("velocity", DimType::VECTOR, 5);
    vectorField.initializeConstant(0.0);
    
    auto& data = vectorField.getData();
    
    // Set different values for each component
    for (int i = 0; i < 5; ++i) {
        data[i * MAX_DIM + 0] = static_cast<double>(i);      // x-component
        data[i * MAX_DIM + 1] = static_cast<double>(i) * 2;  // y-component
    }
    
    // Verify values
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(data[i * MAX_DIM + 0], static_cast<double>(i));
        EXPECT_DOUBLE_EQ(data[i * MAX_DIM + 1], static_cast<double>(i) * 2);
    }
}

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
