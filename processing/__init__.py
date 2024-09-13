import shapely
from processing.image import generate_textured_image, pre_process_web_data
from processing.polygon import generate_3D_models
from model.utils import process_centroid_generator_model, \
    process_input_boundaries_model, get_rooms_counts

from model import rooms_counts_model, regressor_model, boundaries_model


def generate_floor_plan(data, output_name):
    scaled_data = pre_process_web_data(data)

    inner_poly, door_poly = scaled_data["mask"], scaled_data["door_pos"]
    area = data["area"]

    bed_count, bath_count = get_rooms_counts(rooms_counts_model,
                                             inner_poly,
                                             door_poly.centroid,
                                             area)
    bed_count, bath_count = min(bed_count, 3), min(bath_count, 2)
    rooms_centroids = process_centroid_generator_model(regressor_model,
                                                       inner_poly,
                                                       door_poly,
                                                       no_bedrooms=bed_count,
                                                       no_bathrooms=bath_count,
                                                       area=area,
                                                       neighbours_poly=None)

    bedrooms, bathrooms, kitchen = (rooms_centroids["bedroom"],
                                    rooms_centroids["bathroom"],
                                    rooms_centroids["kitchen"])

    rooms_polygons = process_input_boundaries_model(boundaries_model,
                                                    inner_poly,
                                                    door_poly,
                                                    bedrooms, bathrooms,
                                                    kitchen)

    generate_textured_image(rooms_polygons, area=area, output_name=output_name)

    generate_3D_models(rooms_polygons, output_name=output_name)
