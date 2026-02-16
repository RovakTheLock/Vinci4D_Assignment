#ifndef YAML_PARSER_H
#define YAML_PARSER_H

#include <string>
#include <array>

namespace Vinci4D {

/**
 * @brief Parser for YAML configuration files
 * 
 * Reads mesh parameters and simulation settings from YAML input files.
 */
class InputConfigParser {
public:
    /**
     * @brief Construct a parser and load configuration
     * 
     * @param filePath Path to YAML configuration file
     */
    explicit InputConfigParser(const std::string& filePath);
    
    // Mesh parameters
    std::array<double, 2> getXRange() const { return xRange_; }
    std::array<double, 2> getYRange() const { return yRange_; }
    int getNumCellsX() const { return numCellsX_; }
    int getNumCellsY() const { return numCellsY_; }
    
    // Simulation settings
    double getCFL() const { return CFL_; }
    double getRe() const { return Re_; }
    int getOutputFrequency() const { return outputFrequency_; }
    std::string getOutputDirectory() const { return outputDirectory_; }
    double getContinuityTolerance() const { return continuityTolerance_; }
    double getMomentumTolerance() const { return momentumTolerance_; }
    int getNumNonlinearIterations() const { return numNonlinearIterations_; }
    double getTerminationTime() const { return terminationTime_; }

private:
    void loadConfig();
    void parseMeshParameters();
    void parseSimulationSettings();
    
    std::string filePath_;
    
    // Mesh parameters
    std::array<double, 2> xRange_;
    std::array<double, 2> yRange_;
    int numCellsX_;
    int numCellsY_;
    
    // Simulation settings
    double CFL_;
    double Re_;
    int outputFrequency_;
    std::string outputDirectory_;
    double continuityTolerance_;
    double momentumTolerance_;
    int numNonlinearIterations_;
    double terminationTime_;
};

} // namespace Vinci4D

#endif // YAML_PARSER_H
