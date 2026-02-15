import yaml


class InputConfigParser:
    def __init__(self, file_path):
        self.filePath_ = file_path
        self.config_ = None
        self.xRange_ = None
        self.yRange_ = None
        self.numCellsY_ = None
        self.numCellsX_ = None
        self.parse_mesh_parameters()

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


    def __repr__(self):
        return (
            f"MeshConfigParser(\n"
            f"  filePath_='{self.filePath_}',\n"
            f"  xRange_={self.xRange_},\n"
            f"  yRange_={self.yRange_},\n"
            f"  numCellsX_={self.numCellsX_},\n"
            f"  numCellsY_={self.numCellsY_},\n"
            f")"
        )