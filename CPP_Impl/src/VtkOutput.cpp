#include "VtkOutput.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <mpi.h>

namespace Vinci4D {

VtkOutput::VtkOutput(const std::string& outputDirectory, int outputFrequency)
    : outputDirectory_(outputDirectory), outputFrequency_(outputFrequency) {
    if (outputFrequency_ <= 0) {
        outputFrequency_ = 1;
    }
}

bool VtkOutput::shouldWrite(int step) const {
    if (outputFrequency_ <= 0) {
        return false;
    }
    return (step % outputFrequency_) == 0;
}

void VtkOutput::write(int step, double time, const MeshObject& mesh,
                      const std::vector<FieldArray*>& fields) const {
    int mpiRank = mesh.getMpiRank();
    int mpiSize = mesh.getMpiSize();

    ensureOutputDirectory();

    if (mpiSize == 1) {
        std::ofstream out(buildFileName(step));
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open VTK output file");
        }

        writeHeader(out, mesh);
        out << "FIELD FieldData 1\n";
        out << "TIME 1 1 float\n";
        out << std::fixed << std::setprecision(8) << time << "\n";

        writeCellData(out, mesh, fields);
        if (mpiRank == 0) {
            writePvdEntry(step, time, false);
        }
        return;
    }

    // Parallel output: each rank writes a .vts piece and rank 0 writes a .pvts master
    writeVtsPiece(step, time, mesh, fields);
    
    // Ensure all ranks have finished writing their pieces before master file
    MPI_Barrier(MPI_COMM_WORLD);

    int localExtents[4] = {
        mesh.getLocalStartX(),
        mesh.getLocalEndX() + 1,
        mesh.getLocalStartY(),
        mesh.getLocalEndY() + 1
    };

    std::vector<int> allExtents;
    if (mpiRank == 0) {
        allExtents.resize(mpiSize * 4);
    }

    MPI_Gather(localExtents, 4, MPI_INT,
               mpiRank == 0 ? allExtents.data() : nullptr, 4, MPI_INT,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpiRank == 0) {
        writePvts(step, time, mesh, fields, allExtents);
        writePvdEntry(step, time, true);
    }
}

void VtkOutput::ensureOutputDirectory() const {
    if (outputDirectory_.empty()) {
        return;
    }
    std::filesystem::create_directories(outputDirectory_);
}

std::string VtkOutput::buildFileName(int step) const {
    std::ostringstream name;
    name << outputDirectory_;
    if (!outputDirectory_.empty() && outputDirectory_.back() != '/') {
        name << '/';
    }
    name << "output_" << std::setw(6) << std::setfill('0') << step << ".vtk";
    return name.str();
}

std::string VtkOutput::buildPieceFileName(int step, int rank) const {
    std::ostringstream name;
    name << outputDirectory_;
    if (!outputDirectory_.empty() && outputDirectory_.back() != '/') {
        name << '/';
    }
    name << "output_" << std::setw(6) << std::setfill('0') << step
         << "_rank_" << std::setw(3) << std::setfill('0') << rank << ".vts";
    return name.str();
}

std::string VtkOutput::buildMasterFileName(int step) const {
    std::ostringstream name;
    name << outputDirectory_;
    if (!outputDirectory_.empty() && outputDirectory_.back() != '/') {
        name << '/';
    }
    name << "output_" << std::setw(6) << std::setfill('0') << step << ".pvts";
    return name.str();
}

std::string VtkOutput::buildPvdFileName() const {
    std::ostringstream name;
    name << outputDirectory_;
    if (!outputDirectory_.empty() && outputDirectory_.back() != '/') {
        name << '/';
    }
    name << "output.pvd";
    return name.str();
}

void VtkOutput::writeHeader(std::ofstream& out, const MeshObject& mesh) const {
    auto cellCount = mesh.getCellCount();
    int nx = cellCount[0];
    int ny = cellCount[1];

    const auto& xCoords = mesh.getXCoordinates();
    const auto& yCoords = mesh.getYCoordinates();

    out << "# vtk DataFile Version 3.0\n";
    out << "Vinci4D output\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_GRID\n";
    out << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " 1\n";

    int numPoints = (nx + 1) * (ny + 1);
    out << "POINTS " << numPoints << " float\n";

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            out << std::fixed << std::setprecision(8)
                << xCoords[i] << " " << yCoords[j] << " 0.0\n";
        }
    }
}

void VtkOutput::writeCellData(std::ofstream& out, const MeshObject& mesh,
                              const std::vector<FieldArray*>& fields) const {
    // In parallel mode, only rank 0 writes using local cells
    // TODO: Gather all cell data to rank 0 for complete output
    int numCells = (mesh.getMpiSize() > 1) ? mesh.getNumLocalCells() : mesh.getNumCells();
    out << "\nCELL_DATA " << numCells << "\n";

    for (const auto* field : fields) {
        if (!field) {
            continue;
        }
        const auto& data = field->getData();
        if (field->getType() == DimType::SCALAR) {
            out << "SCALARS " << field->getName() << " float 1\n";
            out << "LOOKUP_TABLE default\n";
            for (int i = 0; i < numCells; ++i) {
                out << std::fixed << std::setprecision(8) << data[i] << "\n";
            }
            out << "\n";
        } else if (field->getType() == DimType::VECTOR) {
            out << "VECTORS " << field->getName() << " float\n";
            for (int i = 0; i < numCells; ++i) {
                int idx = i * MAX_DIM;
                out << std::fixed << std::setprecision(8)
                    << data[idx] << " " << data[idx + 1] << " 0.0\n";
            }
            out << "\n";
        }
    }
}

void VtkOutput::writeVtsPiece(int step, double time, const MeshObject& mesh,
                              const std::vector<FieldArray*>& fields) const {
    int mpiRank = mesh.getMpiRank();
    int mpiSize = mesh.getMpiSize();
    if (mpiSize == 1) {
        return;
    }

    int xs = mesh.getLocalStartX();
    int xe = mesh.getLocalEndX() + 1;
    int ys = mesh.getLocalStartY();
    int ye = mesh.getLocalEndY() + 1;

    std::ofstream out(buildPieceFileName(step, mpiRank));
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open VTS output file");
    }

    auto cellCount = mesh.getCellCount();
    int nx = cellCount[0];
    int ny = cellCount[1];
    const auto& xCoords = mesh.getXCoordinates();
    const auto& yCoords = mesh.getYCoordinates();

    out << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <StructuredGrid WholeExtent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";
    out << "    <Piece Extent=\"" << xs << " " << xe << " " << ys << " " << ye << " 0 0\">\n";
    out << "      <FieldData>\n";
    out << "        <DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">\n";
    out << "          " << std::fixed << std::setprecision(8) << time << "\n";
    out << "        </DataArray>\n";
    out << "      </FieldData>\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int j = ys; j <= ye; ++j) {
        for (int i = xs; i <= xe; ++i) {
            out << std::fixed << std::setprecision(8)
                << xCoords[i] << " " << yCoords[j] << " 0.0\n";
        }
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    out << "      <CellData>\n";
    int numCells = mesh.getNumLocalCells();
    for (const auto* field : fields) {
        if (!field) {
            continue;
        }
        const auto& data = field->getData();
        if (field->getType() == DimType::SCALAR) {
            out << "        <DataArray type=\"Float32\" Name=\"" << field->getName()
                << "\" NumberOfComponents=\"1\" format=\"ascii\">\n";
            for (int i = 0; i < numCells; ++i) {
                out << std::fixed << std::setprecision(8) << data[i] << "\n";
            }
            out << "        </DataArray>\n";
        } else if (field->getType() == DimType::VECTOR) {
            out << "        <DataArray type=\"Float32\" Name=\"" << field->getName()
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for (int i = 0; i < numCells; ++i) {
                int idx = i * MAX_DIM;
                out << std::fixed << std::setprecision(8)
                    << data[idx] << " " << data[idx + 1] << " 0.0\n";
            }
            out << "        </DataArray>\n";
        }
    }
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </StructuredGrid>\n";
    out << "</VTKFile>\n";
}

void VtkOutput::writePvts(int step, double time, const MeshObject& mesh,
                          const std::vector<FieldArray*>& fields,
                          const std::vector<int>& extents) const {
    auto cellCount = mesh.getCellCount();
    int nx = cellCount[0];
    int ny = cellCount[1];
    int mpiSize = mesh.getMpiSize();

    std::ofstream out(buildMasterFileName(step));
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open PVTS output file");
    }

    out << "<VTKFile type=\"PStructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PStructuredGrid WholeExtent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";
    out << "    <PFieldData>\n";
    out << "      <PDataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"/>\n";
    out << "    </PFieldData>\n";
    out << "    <PPoints>\n";
    out << "      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n";
    out << "    </PPoints>\n";
    out << "    <PCellData>\n";
    for (const auto* field : fields) {
        if (!field) {
            continue;
        }
        if (field->getType() == DimType::SCALAR) {
            out << "      <PDataArray type=\"Float32\" Name=\"" << field->getName()
                << "\" NumberOfComponents=\"1\"/>\n";
        } else if (field->getType() == DimType::VECTOR) {
            out << "      <PDataArray type=\"Float32\" Name=\"" << field->getName()
                << "\" NumberOfComponents=\"3\"/>\n";
        }
    }
    out << "    </PCellData>\n";

    for (int rank = 0; rank < mpiSize; ++rank) {
        int idx = rank * 4;
        int xs = extents[idx + 0];
        int xe = extents[idx + 1];
        int ys = extents[idx + 2];
        int ye = extents[idx + 3];
        out << "    <Piece Extent=\"" << xs << " " << xe << " " << ys << " " << ye << " 0 0\" "
            << "Source=\"" << "output_" << std::setw(6) << std::setfill('0') << step
            << "_rank_" << std::setw(3) << std::setfill('0') << rank << ".vts\"/>\n";
    }

    out << "  </PStructuredGrid>\n";
    out << "</VTKFile>\n";
}

void VtkOutput::writePvdEntry(int step, double time, bool isParallel) const {
    // Only rank 0 writes the PVD file to avoid race conditions
    int mpiRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    if (mpiRank != 0) {
        return;  // Non-root ranks skip PVD writing
    }
    
    std::string pvdPath = buildPvdFileName();
    std::string dataFile;
    if (isParallel) {
        std::ostringstream name;
        name << "output_" << std::setw(6) << std::setfill('0') << step << ".pvts";
        dataFile = name.str();
    } else {
        std::ostringstream name;
        name << "output_" << std::setw(6) << std::setfill('0') << step << ".vtk";
        dataFile = name.str();
    }

    std::ostringstream entry;
    entry << "    <DataSet timestep=\"" << std::fixed << std::setprecision(8) << time
          << "\" group=\"\" part=\"0\" file=\"" << dataFile << "\"/>\n";

    std::string content;
    if (std::filesystem::exists(pvdPath)) {
        std::ifstream in(pvdPath);
        std::ostringstream buffer;
        buffer << in.rdbuf();
        content = buffer.str();
    }

    std::ofstream out(pvdPath, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open PVD output file");
    }

    if (content.empty()) {
        out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        out << "  <Collection>\n";
        out << entry.str();
        out << "  </Collection>\n";
        out << "</VTKFile>\n";
        out.close();
        return;
    }

    std::string closing = "  </Collection>\n</VTKFile>\n";
    std::size_t pos = content.rfind(closing);
    if (pos == std::string::npos) {
        out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        out << "  <Collection>\n";
        out << entry.str();
        out << "  </Collection>\n";
        out << "</VTKFile>\n";
        out.close();
        return;
    }

    out << content.substr(0, pos);
    out << entry.str();
    out << closing;
    out.close();
}

} // namespace Vinci4D
