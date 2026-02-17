# Vinci4D C++ Implementation

PETSc-based Finite Volume CFD Solver - C++ port of the Python implementation.

## Dependencies

- **PETSc** (3.24.4+) - Sparse linear system solver
- **yaml-cpp** - YAML configuration file parser  
- **Google Test** - Unit testing framework
- **MPI** - Message Passing Interface (installed with PETSc)

All dependencies can be installed via Homebrew on macOS:
```bash
brew install petsc yaml-cpp googletest
```

## Building

Create a build directory and compile:

```bash
cd CPP_Impl
mkdir build
cd build
cmake ..
make
```

## Running the Main Program

```bash
./vinci4d_main ../config/input.yaml
```

Or from the build directory:
```bash
./vinci4d_main
```
(It will use `input.yaml` copied to the build directory)

## Running Tests

### Run all tests with CTest:
```bash
ctest --output-on-failure
```

### Run individual test executables:
```bash
./test_mesh_object
./test_linear_system
./test_fields_holder
./test_assemble_algs
./test_operations
```

### Run tests with verbose output:
```bash
./test_mesh_object --gtest_verbose
```

### Run specific test cases:
```bash
./test_mesh_object --gtest_filter=MeshObjectTest.MeshDimensions
```

## Project Structure

```
CPP_Impl/
├── CMakeLists.txt          # Build configuration
├── config/
│   └── input.yaml          # Simulation parameters
├── include/                # Header files
│   ├── AssembleAlgorithms.h
│   ├── FieldsHolder.h
│   ├── LinearSystem.h
│   ├── MeshObject.h
│   ├── Operations.h
│   ├── QuadElement.h
│   └── YamlParser.h
├── src/                    # Implementation files
│   ├── AssembleAlgorithms.cpp
│   ├── FieldsHolder.cpp
│   ├── LinearSystem.cpp
│   ├── main.cpp
│   ├── MeshObject.cpp
│   ├── Operations.cpp
│   ├── QuadElement.cpp
│   └── YamlParser.cpp
└── tests/                  # Google Test unit tests
    ├── test_assemble_algs.cpp
    ├── test_fields_holder.cpp
    ├── test_linear_system.cpp
    ├── test_mesh_object.cpp
    └── test_operations.cpp
```

## Key Features

- **PETSc Integration**: Efficient sparse matrix operations and linear solvers (GMRES, BiCGStab, direct)
- **Structured Mesh**: 2D Cartesian grid with cell-centered finite volume discretization
- **Field Management**: Scalar and vector field arrays for pressure, velocity, etc.
- **Assembly Algorithms**: Time derivative, diffusion, and source term assembly
- **Operations**: CFL time step computation, mass flux calculation, performance timing
- **Comprehensive Testing**: Unit tests for all major components using Google Test

## Configuration

Edit `config/input.yaml` to modify simulation parameters:

```yaml
mesh_parameters:
  x_range: [0, 1]
  y_range: [0, 1]
  num_cells_x: 50
  num_cells_y: 50

simulation:
  CFL: 1.0
  Re: 1000
  output_frequency: 40
  continuity_tolerance: 1.0e-6
  momentum_tolerance: 1.0e-6
```

## Linear Solver Options

The LinearSystem class supports multiple PETSc solvers:
- `"direct"` - Direct LU factorization
- `"gmres"` - GMRES iterative solver with ILU preconditioning
- `"bicgstab"` - BiCGStab iterative solver
- `"cg"` - Conjugate Gradient (for symmetric systems)

## Development

### Adding New Tests

1. Create a new test file in `tests/` directory
2. Include gtest headers and use `TEST_F` macros
3. Add the test filename to `TEST_SOURCES` in `CMakeLists.txt`
4. Rebuild and run with `ctest`

### Debugging

Build with debug symbols:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

Run tests with GDB:
```bash
gdb ./test_mesh_object
```

## Notes

- PETSc must be initialized at program start with `PetscInitialize()`
- All tests initialize PETSc in their `main()` functions
- Matrices must be assembled with `assembleMatrix()` before solving
- Field arrays use row-major storage for vector components
