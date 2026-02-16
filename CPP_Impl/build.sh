#!/bin/bash

# Build script for Vinci4D C++ implementation

set -e  # Exit on error

echo "=== Building Vinci4D C++ Implementation ==="

# Check if we should clean
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Try to find MPI compiler
if command -v mpicc &> /dev/null; then
    echo "Found MPI compiler at: $(which mpicc)"
    export CC=mpicc
    export CXX=mpicxx
else
    echo "Warning: MPI compiler not found in PATH"
fi

# Run CMake
echo ""
echo "Running CMake..."
cmake ..

# Build
echo ""
echo "Compiling..."
make -j$(sysctl -n hw.ncpu)

echo ""
echo "=== Build complete! ==="
echo ""
echo "Run main program:    ./vinci4d_main"
echo "Run all tests:       ctest --output-on-failure"
echo "Run specific test:   ./test_mesh_object"
echo ""
