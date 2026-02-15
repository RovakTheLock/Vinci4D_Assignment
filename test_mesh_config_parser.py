import unittest
import tempfile
import os
import yaml
from YamlParser import InputConfigParser


class TestMeshConfigParser(unittest.TestCase):
    """Unit tests for MeshConfigParser class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.test_dir = tempfile.mkdtemp()
        self.valid_config = {
            'mesh_parameters': {
                'x_range': [0, 1],
                'y_range': [0, 1],
                'num_cells_x': 50,
                'num_cells_y': 50
            },
            'simulation': {
                'CFL': 1.0,
                'Re': 1000,
                'output_frequency': 40,
                'continuity_tolerance': 1.0e-6,
                'momentum_tolerance': 1.0e-6
            }
        }

    def tearDown(self):
        """Clean up after each test method"""
        # Remove all test files
        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            os.remove(file_path)
        os.rmdir(self.test_dir)

    def _create_yaml_file(self, filename, config):
        """Helper method to create a YAML file with given config"""
        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, 'w') as f:
            yaml.dump(config, f)
        return file_path

    # load_config tests
    def test_load_config_valid_file(self):
        """Test that load_config successfully loads a valid YAML file"""
        file_path = self._create_yaml_file("valid_config.yaml", self.valid_config)
        parser = InputConfigParser(file_path)
        parser.load_config()
        self.assertIsNotNone(parser.config_)
        self.assertEqual(parser.config_, self.valid_config)

    def test_load_config_file_not_found(self):
        """Test that load_config raises FileNotFoundError for non-existent file"""
        with self.assertRaises(FileNotFoundError):
            parser = InputConfigParser("nonexistent_file.yaml")
            parser.load_config()

    def test_load_config_invalid_yaml(self):
        """Test that load_config raises YAMLError for malformed YAML"""
        file_path = os.path.join(self.test_dir, "invalid.yaml")
        with open(file_path, 'w') as f:
            f.write("invalid: yaml: content: [")  # Malformed YAML
        
        with self.assertRaises(yaml.YAMLError):
            parser = InputConfigParser(file_path)

    # parse_mesh_parameters tests
    def test_parse_mesh_parameters_valid_config(self):
        """Test that parse_mesh_parameters correctly parses valid config"""
        file_path = self._create_yaml_file("valid_config.yaml", self.valid_config)
        parser = InputConfigParser(file_path)
        parser.parse_mesh_parameters()
        
        self.assertEqual(parser.xRange_, [0, 1])
        self.assertEqual(parser.yRange_, [0, 1])
        self.assertEqual(parser.numCellsX_, 50)
        self.assertEqual(parser.numCellsY_, 50)

    def test_parse_mesh_parameters_without_load_config(self):
        """Test that parse_mesh_parameters calls load_config internally"""
        file_path = self._create_yaml_file("valid_config.yaml", self.valid_config)
        parser = InputConfigParser(file_path)
        # Should not raise an error, load_config is called internally
        parser.parse_mesh_parameters()
        self.assertIsNotNone(parser.config_)

    def test_parse_mesh_parameters_missing_mesh_parameters_key(self):
        """Test parsing when 'mesh_parameters' key is missing"""
        empty_config = {}
        file_path = self._create_yaml_file("empty_config.yaml", empty_config)
        parser = InputConfigParser(file_path)
        parser.parse_mesh_parameters()
        
        self.assertIsNone(parser.xRange_)
        self.assertIsNone(parser.yRange_)
        self.assertIsNone(parser.numCellsX_)
        self.assertIsNone(parser.numCellsY_)

    def test_parse_mesh_parameters_partial_config(self):
        """Test parsing with only some mesh parameters present"""
        partial_config = {
            'mesh_parameters': {
                'x_range': [0, 2],
                'num_cells_x': 100
            }
        }
        file_path = self._create_yaml_file("partial_config.yaml", partial_config)
        parser = InputConfigParser(file_path)
        parser.parse_mesh_parameters()
        
        self.assertEqual(parser.xRange_, [0, 2])
        self.assertEqual(parser.numCellsX_, 100)
        self.assertIsNone(parser.yRange_)
        self.assertIsNone(parser.numCellsY_)

    def test_parse_mesh_parameters_different_values(self):
        """Test parsing with different mesh parameter values"""
        custom_config = {
            'mesh_parameters': {
                'x_range': [-5, 10],
                'y_range': [0, 20],
                'num_cells_x': 200,
                'num_cells_y': 100
            }
        }
        file_path = self._create_yaml_file("custom_config.yaml", custom_config)
        parser = InputConfigParser(file_path)
        parser.parse_mesh_parameters()
        
        self.assertEqual(parser.xRange_, [-5, 10])
        self.assertEqual(parser.yRange_, [0, 20])
        self.assertEqual(parser.numCellsX_, 200)
        self.assertEqual(parser.numCellsY_, 100)

    def test_parse_mesh_parameters_nonexistent_file(self):
        """Test that parse_mesh_parameters raises error for nonexistent file"""
        with self.assertRaises(FileNotFoundError):
            parser = InputConfigParser("nonexistent_file.yaml")

    def test_parse_mesh_parameters_with_extra_fields(self):
        """Test parsing config with extra fields (should be ignored)"""
        config_with_extra = {
            'mesh_parameters': {
                'x_range': [0, 1],
                'y_range': [0, 1],
                'num_cells_x': 50,
                'num_cells_y': 50,
                'extra_field': 'should_be_ignored'
            },
            'other_section': {'key': 'value'}
        }
        file_path = self._create_yaml_file("extra_config.yaml", config_with_extra)
        parser = InputConfigParser(file_path)
        parser.parse_mesh_parameters()
        
        self.assertEqual(parser.xRange_, [0, 1])
        self.assertEqual(parser.yRange_, [0, 1])
        self.assertEqual(parser.numCellsX_, 50)
        self.assertEqual(parser.numCellsY_, 50)

    # parse_simulation_settings tests
    def test_parse_simulation_settings_valid_config(self):
        """Test that parse_simulation_settings correctly parses simulation config"""
        file_path = self._create_yaml_file("valid_sim_config.yaml", self.valid_config)
        parser = InputConfigParser(file_path)

        self.assertEqual(parser.CFL_, 1.0)
        self.assertEqual(parser.Re_, 1000)
        self.assertEqual(parser.outputFrequency_, 40)
        self.assertEqual(parser.continuityTolerance_, 1.0e-6)
        self.assertEqual(parser.momentumTolerance_, 1.0e-6)

    def test_parse_simulation_settings_missing_simulation_key(self):
        """Test parsing when 'simulation' key is missing"""
        config_without_sim = {
            'mesh_parameters': {
                'x_range': [0, 1],
                'y_range': [0, 1],
                'num_cells_x': 50,
                'num_cells_y': 50
            }
        }
        file_path = self._create_yaml_file("no_sim_config.yaml", config_without_sim)
        parser = InputConfigParser(file_path)

        self.assertIsNone(parser.CFL_)
        self.assertIsNone(parser.Re_)
        self.assertIsNone(parser.outputFrequency_)
        self.assertIsNone(parser.continuityTolerance_)
        self.assertIsNone(parser.momentumTolerance_)


if __name__ == '__main__':
    unittest.main()
