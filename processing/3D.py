from processing.image import get_door_rect
from processing.polygon import extrude_and_save_multipolygon, \
    convert_obj_to_gltf


def generate_3D_models(data, output_name):
    walls = data['wall']
    inner_poly = data['inner']
    door = data['door']
    # bedrooms = data['bedroom']
    # bathrooms = data['bathroom']
    # kitchen = data['kitchen']
    # living = data['living']

    output_name = output_name.split(".")[0]
    obj_file_path = f"./outputs/objs/{output_name}.obj"
    gltf_file_path = f"./outputs/gltf/{output_name}.gltf"

    door_poly = get_door_rect(door.centroid, inner_poly)

    extrude_and_save_multipolygon(walls, inner_poly, door_poly, 27,
                                  output_file=obj_file_path,
                                  file_type="obj")

    convert_obj_to_gltf(obj_file_path=obj_file_path,
                        gltf_file_path=gltf_file_path)
