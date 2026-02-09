import argparse
import yaml
import sys
from YamlParser import InputConfigParser
from MeshObject import MeshObject
from FieldsHolder import FieldNames
from FieldsHolder import FieldArray
from FieldsHolder import DimType

def main():
    parser = argparse.ArgumentParser(description="Process a mesh configuration file.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input YAML file")
    args = parser.parse_args()
    try:
        mesh_parser = InputConfigParser(args.input)
        mesh_parser.parse_mesh_parameters()
    except Exception as e:
        print(f"An error occurred while trying to parse mesh parameters: {e}")
        sys.exit(1)
    mesh = MeshObject(mesh_parser)
    fieldDict = {}
    fieldDict[FieldNames.PRESSURE.value] = FieldArray(FieldNames.PRESSURE.value, DimType.SCALAR, mesh.get_num_cells())
    fieldDict[FieldNames.VELOCITY.value] = FieldArray(FieldNames.VELOCITY.value, DimType.VECTOR, mesh.get_num_cells())
    fieldDict[FieldNames.MASS_FLUX_FACE.value] = FieldArray(FieldNames.MASS_FLUX_FACE.value, DimType.SCALAR, mesh.get_num_faces())

if __name__ == "__main__":
    main()