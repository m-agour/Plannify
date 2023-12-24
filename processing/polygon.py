import ntpath
import os
import subprocess

import geopandas as gpd
import numpy as np
import trimesh
from PIL import Image
from matplotlib import pyplot as plt
from shapely import MultiPolygon, affinity, Polygon, box
from shapely.ops import unary_union

from processing.geometry import get_door_rect

materials_templates = {
    "wall": '''newmtl material_idx
Ka 0.20000000 0.20000000 0.20000000
Kd 0.10000000 0.10000000 0.10000000
Ks 0.20000000 0.20000000 0.20000000
Ns 0.20000000
map_Kd Bricks17_col.jpg
map_Ns Bricks17_rgh.jpg
map_Bump Bricks17_nrm.jpg
''',
    "floor": '''newmtl material_idx
Ka 0.20000000 0.20000000 0.20000000
Kd 0.10000000 0.10000000 0.10000000
Ks 0.20000000 0.20000000 0.20000000
Ns 0.20000000
map_Kd Concrete12_col.jpg
map_Ns Concrete12_rgh.jpg
map_Bump Concrete12_nrm.jpg
''',
    "grass": '''newmtl material_idx

map_Kd Ground_ScrubGrassField_col.png
map_Bump Ground_ScrubGrassField_norm.png
'''
}


def convert_obj_to_gltf(obj_file_path, gltf_file_path):
    command = f'obj2gltf -i "{obj_file_path}" -o "{gltf_file_path}"'
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    # print error
    for line in process.stdout:
        print(line.decode("utf-8").replace("\n", ""))


def get_mesh(polygon, texture, height=5., translate=0.):
    # Extrude the polygon normally along the Z-axis
    mesh = trimesh.creation.extrude_polygon(polygon, height=-height)
    # Rotate the mesh to align the extrusion along the Y-axis
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2,
                                                              [1, 0, 0])
    translation_matrix = trimesh.transformations.translation_matrix(
        [0, 0, translate])
    mesh.apply_transform(translation_matrix)
    mesh.apply_transform(rotation_matrix)
    material = trimesh.visual.material.SimpleMaterial(image=texture)
    mesh.visual.material = material
    return mesh


def extract_mesh_data(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    uv_coords = mesh.visual.uv

    # Convert face indices to tuples of vertex and UV indices
    face_tuples = []
    for face in faces:
        face_tuples.append([(v_idx, v_idx) for v_idx in face])

    return vertices, face_tuples, uv_coords


def merge_trimesh_meshes(meshes, output_path, scale=1.0):
    vertex_offset = 1
    uv_offset = 1
    materials = ""
    file_name = os.path.splitext(ntpath.basename(output_path))[0]
    dir_name = os.path.dirname(output_path)

    with open(output_path, 'w') as file:
        mtl_name = file_name + '.mtl'
        mtl_path = os.path.join(dir_name, mtl_name)
        with open(mtl_path, 'w') as mtl_file:
            file.write(f'mtllib {mtl_name}\n')
            for i, (mesh, typ) in enumerate(meshes):
                file.write(f"usemtl material_{i}\n")
                vertices, faces, uv_coords = extract_mesh_data(mesh)
                if typ in materials_templates:
                    materials += materials_templates[typ].replace(
                        "material_idx", f"material_{i}")
                for vertex in vertices:
                    file.write(f'v {" ".join(map(str, vertex))}\n')

                for uv_coord in uv_coords:
                    uv_coord = uv_coord / scale
                    file.write(f'vt {" ".join(map(str, uv_coord))}\n')

                for face in faces:
                    face_str = ' '.join(
                        [f'{v_idx + vertex_offset}/{uv_idx + vertex_offset}'
                         for v_idx, uv_idx in face])
                    file.write(f'f {face_str}\n')

                vertex_offset += len(vertices)
                uv_offset += len(uv_coords)

            mtl_file.write(materials)


def load_texture_image(image_path: str):
    try:
        image = Image.open(image_path)
        return image
    except IOError:
        print("Unable to load image")
        return None


def create_multipolygon_from_exterior_points(multi_poly):
    polygons = []
    for poly in multi_poly.geoms:
        ext = poly.exterior.coords[:-1]
        polygon = Polygon(ext)
        polygons.append(polygon)

    new_multi_poly = MultiPolygon(polygons)

    return new_multi_poly


def extrude_and_save_multipolygon(multi_poly: MultiPolygon, floor, door,
                                  height: float, output_file: str,
                                  file_type: str = 'obj'):
    wall_meshes = []
    cement_meshes = []

    x1, y1, x2, y2 = multi_poly.bounds
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    multi_poly = affinity.translate(multi_poly, -cx, -cy)
    door = affinity.translate(door, -cx, -cy)
    floor = affinity.translate(floor, -cx, -cy)

    gpd.GeoSeries([multi_poly, door, floor]).plot()
    plt.show()

    wall_texture = load_texture_image("assets/textures/Bricks17_col.jpg")
    floor_texture = load_texture_image("assets/textures/wall_2.jpg")
    if not isinstance(multi_poly, MultiPolygon):
        multi_poly = MultiPolygon([multi_poly])

    floor_poly = create_multipolygon_from_exterior_points(multi_poly).buffer(
        -1, cap_style=2, join_style=2)
    floor_poly = unary_union(floor_poly)

    floor_mesh = get_mesh(floor_poly, floor_texture, height=2)
    floor_mesh.visual.uv = np.array(
        [[vertex[0] / 30, vertex[2] / 30] for vertex in floor_mesh.vertices])
    # meshes.append(mesh)

    house_bounds = np.array(floor_poly.bounds) + np.array(
        [-300, -300, 300, 300])
    grass_poly = box(*house_bounds)
    grass_poly = affinity.translate(grass_poly, 0, 0, 0)
    grass_mesh = get_mesh(grass_poly, floor_texture, height=0.1)
    grass_mesh = grass_mesh.subdivide().subdivide().subdivide().subdivide().subdivide()

    # noise = np.random.normal(scale=0.5, size=grass_mesh.vertices.shape)
    # grass_mesh.vertices += noise
    # grass_mesh.vertex_normals = grass_mesh.vertex_normals
    #
    grass_mesh.visual.uv = np.array(
        [[vertex[0] / 30, vertex[2] / 30] for vertex in grass_mesh.vertices])

    for polygon in multi_poly.geoms:
        polygon = polygon.buffer(-0.6).buffer(0.6 / 2)

        mesh = get_mesh(polygon, wall_texture, height=height,
                        translate=-0.0001)
        mesh.visual.uv = np.array(
            [[(vertex[0] + vertex[2]) / 20, vertex[1] / 20] for vertex in
             mesh.vertices])
        wall_meshes.append(mesh)

        mesh = get_mesh(polygon, wall_texture, height=0.0001,
                        translate=-height)
        mesh.visual.uv = np.array(
            [[vertex[0] / 10, vertex[2] / 10] for vertex in mesh.vertices])
        cement_meshes.append(mesh)

    door_mesh = get_mesh(door, wall_texture, height=height * 0.8)
    door_mesh.visual.uv = np.array(
        [[(vertex[0] + vertex[2]) / 30, vertex[1] / 30] for vertex in
         door_mesh.vertices])

    # scale meshes
    scale = 1 / 5
    for i in range(len(wall_meshes)):
        wall_meshes[i].vertices *= scale
        wall_meshes[i].visual.uv *= scale
    for i in range(len(cement_meshes)):
        cement_meshes[i].vertices *= scale
        cement_meshes[i].visual.uv *= scale
    floor_mesh.vertices *= scale
    floor_mesh.visual.uv *= scale
    grass_mesh.vertices *= scale
    grass_mesh.visual.uv *= scale
    door_mesh.vertices *= scale
    door_mesh.visual.uv *= scale

    merge_trimesh_meshes([[w, 'wall'] for w in wall_meshes] +
                         [[c, 'floor'] for c in cement_meshes] +
                         [[floor_mesh, 'floor'], [grass_mesh, 'grass']],
                         output_file, scale)


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
