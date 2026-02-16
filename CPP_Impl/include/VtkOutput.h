#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include "MeshObject.h"
#include "FieldsHolder.h"
#include <fstream>
#include <string>
#include <vector>

namespace Vinci4D {

/**
 * @brief Writes cell-centered fields to legacy VTK files.
 *
 * Output format:
 * - STRUCTURED_GRID with points from mesh coordinates
 * - CELL_DATA for scalar/vector fields
 */
class VtkOutput {
public:
    VtkOutput(const std::string& outputDirectory, int outputFrequency);

    bool shouldWrite(int step) const;

    void write(int step, double time, const MeshObject& mesh,
               const std::vector<FieldArray*>& fields) const;

private:
    std::string outputDirectory_;
    int outputFrequency_;

    void ensureOutputDirectory() const;
    std::string buildFileName(int step) const;
    void writeHeader(std::ofstream& out, const MeshObject& mesh) const;
    void writeCellData(std::ofstream& out, const MeshObject& mesh,
                       const std::vector<FieldArray*>& fields) const;
};

} // namespace Vinci4D

#endif // VTK_OUTPUT_H
