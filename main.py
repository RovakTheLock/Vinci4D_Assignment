import argparse
import yaml
import sys
from YamlParser import MeshConfigParser

def main():
    parser = argparse.ArgumentParser(description="Process a mesh configuration file.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input YAML file")
    args = parser.parse_args()
    try:
        mesh_parser = MeshConfigParser(args.input)
        mesh_parser.parse_mesh_parameters()
    except Exception as e:
        print(f"An error occurred while trying to parse mesh parameters: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()