#include "VtkOutput.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
    ensureOutputDirectory();

    std::ofstream out(buildFileName(step));
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open VTK output file");
    }

    writeHeader(out, mesh);
    out << "FIELD FieldData 1\n";
    out << "TIME 1 1 float\n";
    out << std::fixed << std::setprecision(8) << time << "\n";

    writeCellData(out, mesh, fields);
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
    int numCells = mesh.getNumCells();
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

} // namespace Vinci4D
