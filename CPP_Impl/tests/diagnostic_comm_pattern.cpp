#include <iostream>
#include <fstream>
#include <mpi.h>
#include "../include/MeshObject.h"
#include "../include/YamlParser.h"

using namespace Vinci4D;

/**
 * Simple diagnostic to print exact neighbor communication pattern
 * No test framework, just raw logging to avoid any MPI Google Test issues
 */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char* yaml_content = R"(
mesh_parameters:
  x_range: [0.0, 1.0]
  y_range: [0.0, 1.0]
  num_cells_x: 10
  num_cells_y: 10
simulation:
  Re: 100
  CFL: 0.5
)";

    std::ofstream tmpYaml("/tmp/ghost_diagnostic.yaml");
    tmpYaml << yaml_content;
    tmpYaml.close();

    InputConfigParser parser("/tmp/ghost_diagnostic.yaml");
    MeshObject mesh(parser);

    // Open file for this rank
    std::string logFile = "/tmp/comm_pattern_" + std::to_string(rank) + ".txt";
    std::ofstream log(logFile);

    log << "=== Rank " << rank << " Communication Pattern ===" << std::endl;
    log << "Position: rankX=" << mesh.getLocalStartX() << "-" << mesh.getLocalEndX()
        << ", rankY=" << mesh.getLocalStartY() << "-" << mesh.getLocalEndY() << std::endl;
    log << "Local cells: " << mesh.getNumLocalCells() << ", Ghost cells: " << mesh.getNumGhostCells() << std::endl;
    log << std::endl;

    // The issue is internal to MeshObject, so we can't directly access ghostCommInfo_
    // But we can at least verify the decomposition is consistent
    log << "Neighbor ranks: ";
    for (auto n : mesh.getNeighborRanks()) {
        log << n << " ";
    }
    log << std::endl;

    log.close();

    // Print from rank 0
    if (rank == 0) {
        std::cout << "Communication pattern logged to /tmp/comm_pattern_*.txt" << std::endl;
        for (int r = 0; r < size; ++r) {
            std::ifstream in("/tmp/comm_pattern_" + std::to_string(r) + ".txt");
            std::string line;
            while (std::getline(in, line)) {
                std::cout << line << std::endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
