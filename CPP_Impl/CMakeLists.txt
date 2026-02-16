cmake_minimum_required(VERSION 3.10)
project(Vinci4D_CPP VERSION 1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MPI first (required by PETSc)
find_package(MPI REQUIRED)

# Find PkgConfig for PETSc
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED PETSc)

# Find yaml-cpp
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found via CMake, trying pkg-config")
    pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
    include_directories(${YAML_CPP_INCLUDE_DIRS})
    link_directories(${YAML_CPP_LIBRARY_DIRS})
else()
    message(STATUS "Found yaml-cpp via CMake")
endif()

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${PETSC_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
)

# Link directories
link_directories(
    ${PETSC_LIBRARY_DIRS}
    ${YAML_CPP_LIBRARY_DIRS}
)

# Source files
set(SOURCES
    src/YamlParser.cpp
    src/QuadElement.cpp
    src/MeshObject.cpp
    src/FieldsHolder.cpp
    src/LinearSystem.cpp
    src/AssembleAlgorithms.cpp
    src/Operations.cpp
    src/VtkOutput.cpp
)

# Create library
add_library(vinci4d_lib ${SOURCES})
if(yaml-cpp_FOUND)
    target_link_libraries(vinci4d_lib 
        ${PETSC_LIBRARIES}
        yaml-cpp
        MPI::MPI_CXX
    )
else()
    target_link_libraries(vinci4d_lib 
        ${PETSC_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        MPI::MPI_CXX
    )
endif()

# Main executable
add_executable(vinci4d_main src/main.cpp)
target_link_libraries(vinci4d_main vinci4d_lib)

# Enable testing
enable_testing()

# Test executables
set(TEST_SOURCES
    test_mesh_object
    test_fields_holder
    test_linear_system
    test_assemble_algs
    test_operations
    test_pseudo_sim_setup
)

foreach(test_name ${TEST_SOURCES})
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} 
        vinci4d_lib
        GTest::gtest
        GTest::gtest_main
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Parallel test executable (requires MPI)
add_executable(test_parallel tests/test_parallel.cpp)
target_link_libraries(test_parallel 
    vinci4d_lib
    GTest::gtest
    MPI::MPI_CXX
)

# Add parallel tests with different process counts
find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
if(MPIEXEC_EXECUTABLE)
    add_test(NAME test_parallel_np1 COMMAND ${MPIEXEC_EXECUTABLE} -np 1 ./test_parallel)
    add_test(NAME test_parallel_np2 COMMAND ${MPIEXEC_EXECUTABLE} -np 2 ./test_parallel)
    add_test(NAME test_parallel_np4 COMMAND ${MPIEXEC_EXECUTABLE} -np 4 ./test_parallel)
endif()

# Copy config file to build directory
configure_file(${CMAKE_SOURCE_DIR}/config/input.yaml 
               ${CMAKE_BINARY_DIR}/input.yaml COPYONLY)
