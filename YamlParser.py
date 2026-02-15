import yaml


class InputConfigParser:
    def __init__(self, file_path):
        self.filePath_ = file_path
        self.config_ = None
        self.xRange_ = None
        self.yRange_ = None
        self.numCellsY_ = None
        self.numCellsX_ = None
        self.CFL_ = None
        self.Re_ = None
        self.outputFrequency_ = 100
        self.outputDirectory_ = None
        self.continuityTolerance_ = 1.0e-6
        self.momentumTolerance_ = 1.0e-6
        self.numNonlinearIterations_ = 3
        self.terminationTime_ = None
        self.parse_mesh_parameters()
        self.parse_simulation_settings()

    def load_config(self):
        try:
            with open(self.filePath_, 'r') as file:
                self.config_ = yaml.safe_load(file)
                print(f"Successfully loaded: {self.filePath_}")
        except FileNotFoundError:
            print(f"Error: The file '{self.filePath_}' was not found.")
            raise
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")
            raise

    def parse_mesh_parameters(self):
        self.load_config()
        if self.config_ is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        mesh_params = self.config_.get('mesh_parameters', {})
        print(f"MeshConfigParser successfully loaded: {self.filePath_}")
        self.xRange_ = mesh_params.get('x_range')
        self.yRange_ = mesh_params.get('y_range')
        self.numCellsX_ = mesh_params.get('num_cells_x')
        self.numCellsY_ = mesh_params.get('num_cells_y')

    def parse_simulation_settings(self):
        self.load_config()
        if self.config_ is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        simulation_settings = self.config_.get('simulation', {})
        self.CFL_ = simulation_settings.get('CFL')
        self.Re_ = simulation_settings.get('Re')
        self.outputFrequency_ = simulation_settings.get('output_frequency', self.outputFrequency_)
        self.outputDirectory_ = simulation_settings.get('output_directory')
        self.continuityTolerance_ = simulation_settings.get('continuity_tolerance', self.continuityTolerance_)
        self.momentumTolerance_ = simulation_settings.get('momentum_tolerance', self.momentumTolerance_)
        self.numNonlinearIterations_ = simulation_settings.get('num_nonlinear_iterations', self.numNonlinearIterations_)
        self.terminationTime_ = simulation_settings.get('termination_time', self.terminationTime_)

    def __repr__(self):
        return (
            f"MeshConfigParser(\n"
            f"  filePath_='{self.filePath_}',\n"
            f"  xRange_={self.xRange_},\n"
            f"  yRange_={self.yRange_},\n"
            f"  numCellsX_={self.numCellsX_},\n"
            f"  numCellsY_={self.numCellsY_},\n"
            f"  CFL_={self.CFL_},\n"
            f"  Re_={self.Re_},\n"
            f"  outputFrequency_={self.outputFrequency_},\n"
            f"  outputDirectory_='{self.outputDirectory_}',\n"
            f"  continuityTolerance_={self.continuityTolerance_},\n"
            f"  momentumTolerance_={self.momentumTolerance_},\n"
            f")"
        )