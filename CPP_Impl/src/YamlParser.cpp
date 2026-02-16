#include "YamlParser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>

namespace Vinci4D {

InputConfigParser::InputConfigParser(const std::string& filePath)
    : filePath_(filePath),
      numCellsX_(0),
      numCellsY_(0),
      CFL_(1.0),
      Re_(1000.0),
      outputFrequency_(100),
      outputDirectory_("output"),
      continuityTolerance_(1.0e-6),
      momentumTolerance_(1.0e-6),
      numNonlinearIterations_(3),
      terminationTime_(1.0) {
    loadConfig();
    parseMeshParameters();
    parseSimulationSettings();
}

void InputConfigParser::loadConfig() {
    try {
        YAML::Node config = YAML::LoadFile(filePath_);
        std::cout << "Successfully loaded: " << filePath_ << std::endl;
    } catch (const YAML::BadFile& e) {
        throw std::runtime_error("Error: The file '" + filePath_ + "' was not found.");
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing YAML: " + std::string(e.what()));
    }
}

void InputConfigParser::parseMeshParameters() {
    try {
        YAML::Node config = YAML::LoadFile(filePath_);
        
        if (config["mesh_parameters"]) {
            YAML::Node meshParams = config["mesh_parameters"];
            
            if (meshParams["x_range"] && meshParams["x_range"].size() == 2) {
                xRange_[0] = meshParams["x_range"][0].as<double>();
                xRange_[1] = meshParams["x_range"][1].as<double>();
            }
            
            if (meshParams["y_range"] && meshParams["y_range"].size() == 2) {
                yRange_[0] = meshParams["y_range"][0].as<double>();
                yRange_[1] = meshParams["y_range"][1].as<double>();
            }
            
            if (meshParams["num_cells_x"]) {
                numCellsX_ = meshParams["num_cells_x"].as<int>();
            }
            
            if (meshParams["num_cells_y"]) {
                numCellsY_ = meshParams["num_cells_y"].as<int>();
            }
        }
        
        std::cout << "MeshConfigParser successfully loaded mesh parameters" << std::endl;
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing mesh parameters: " + std::string(e.what()));
    }
}

void InputConfigParser::parseSimulationSettings() {
    try {
        YAML::Node config = YAML::LoadFile(filePath_);
        
        if (config["simulation"]) {
            YAML::Node simSettings = config["simulation"];
            
            if (simSettings["CFL"]) {
                CFL_ = simSettings["CFL"].as<double>();
            }
            
            if (simSettings["Re"]) {
                Re_ = simSettings["Re"].as<double>();
            }
            
            if (simSettings["output_frequency"]) {
                outputFrequency_ = simSettings["output_frequency"].as<int>();
            }
            
            if (simSettings["output_directory"]) {
                outputDirectory_ = simSettings["output_directory"].as<std::string>();
            }
            
            if (simSettings["continuity_tolerance"]) {
                continuityTolerance_ = simSettings["continuity_tolerance"].as<double>();
            }
            
            if (simSettings["momentum_tolerance"]) {
                momentumTolerance_ = simSettings["momentum_tolerance"].as<double>();
            }
            
            if (simSettings["num_nonlinear_iterations"]) {
                numNonlinearIterations_ = simSettings["num_nonlinear_iterations"].as<int>();
            }
            
            if (simSettings["termination_time"]) {
                terminationTime_ = simSettings["termination_time"].as<double>();
            }
        }
        
        std::cout << "Successfully loaded simulation settings" << std::endl;
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing simulation settings: " + std::string(e.what()));
    }
}

} // namespace Vinci4D
