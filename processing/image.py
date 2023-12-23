import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from shapely import Polygon, box, MultiPolygon, GeometryCollection
from shapely.affinity import translate, scale
from shapely.geometry import Point
from PIL import ImageFont, Image, ImageDraw
from processing.geometry import get_mask
from processing.polygon import extrude_and_save_multipolygon, \
    convert_obj_to_gltf


def get_rects(image, centroids, door, get_max=0):
    bounds = get_mask(centroids, point_s=8).astype(np.uint8)
    d = ~get_mask(door.centroid, point_s=30).astype(np.uint8)
    img = image.astype(np.uint8)
    img[np.where(bounds == 0)] = 0

    kernel = np.ones((1, 1))
    convolved = convolve2d(image.reshape((256, 256)), kernel, mode='same',
                           boundary='fill', fillvalue=0)
    # plt.imshow(image)
    # plt.show()
    y, x = np.unravel_index(np.argmax(convolved), convolved.shape)
    return [Point(x, y)]


def imfy(img):
    """Convert image to uint8 format"""
    im = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return im


def get_door_rect(door_point, inner, scale=1):
    width = inner.area ** 0.5 * 0.04 * scale
    door = inner.exterior.intersection(
        door_point.buffer(width, join_style=2)).buffer(width / 2,
                                                       join_style=2).bounds
    door_box = box(*door)
    door = door_box - door_box.intersection(inner)
    return door


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def load_tex(path, as_rgb=False, brightness_inc=0):
    tex = cv2.imread(path)
    if as_rgb:
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
    if brightness_inc:
        tex = increase_brightness(tex, brightness_inc)
    return tex


doortex = load_tex("assets/door.png")
outex = load_tex("assets/outer1.jpg")
intex = load_tex("assets/inner1.jpg")
kitex = load_tex("assets/kitchen.png", as_rgb=True, brightness_inc=0)
batex = load_tex("assets/kitchen-Copy1.png", as_rgb=True, brightness_inc=0)
betex = load_tex("assets/bedroom.png", as_rgb=True, brightness_inc=0)
livtex = load_tex("assets/living.png", as_rgb=True, brightness_inc=30)
waltex = load_tex("assets/wall.png", as_rgb=True, brightness_inc=250)


def fix_mask_3d_pad(mask, padding=64):
    mask = cv2.merge(3 * [mask])
    mask = cv2.copyMakeBorder(mask.copy(), padding, padding, padding, padding,
                              cv2.BORDER_CONSTANT)
    return mask


def fix_fit_poly(poly, size, pad=False):
    area = poly.area
    # poly = scale(poly, 1, -1, 1)
    #################################################################
    # scale to fit width
    # Get polygon bounding box
    minx, miny, maxx, maxy = poly.bounds
    bbox_width = maxx - minx
    bbox_height = maxy - miny

    aspect_ratio = bbox_width / bbox_height

    if aspect_ratio > size[0] / size[1]:
        new_width = size[0]
        new_height = size[0] / aspect_ratio
    else:
        new_width = size[1] * aspect_ratio
        new_height = size[1]

    scale_x = new_width / bbox_width
    scale_y = new_height / bbox_height

    # spower = width / max(w, h)
    poly_scaled = scale(poly, scale_y, scale_x, 1, (minx, miny))

    x1, y1, x2, y2 = poly_scaled.bounds
    w, h = x2 - x1, y2 - y1

    if pad:
        poly_fit = translate(poly_scaled, -x1 + ((size[0] - w) / 2),
                             -y1 + ((size[1] - h) / 2))

    else:
        poly_fit = translate(poly_scaled, -x1, -y1)

    # poly_padded = scale(poly_fit, 1, 1, 1)
    #

    if isinstance(poly_fit, (MultiPolygon, GeometryCollection)):
        return max(poly_fit.geoms, key=lambda x: x.area), min(poly_fit.geoms,
                                                              key=lambda
                                                                  x: x.area)

    return poly_fit, None


def draw_display_picture(mask, door, bedrooms, bathrooms, kitchen, living,
                         wall, icon=False, padding=64):
    mask = fix_mask_3d_pad(mask, padding=padding)
    door = fix_mask_3d_pad(door, padding=padding)
    # get masks of other enitites
    bedrooms = fix_mask_3d_pad(bedrooms, padding=padding)
    bathrooms = fix_mask_3d_pad(bathrooms, padding=padding)
    kitchen = fix_mask_3d_pad(kitchen, padding=padding)
    wall = fix_mask_3d_pad(wall, padding=padding)
    living = fix_mask_3d_pad(living, padding=padding)

    if not icon:
        out = outex[:mask.shape[0], :mask.shape[1]].copy()
        inner = intex[:mask.shape[0], :mask.shape[1]].copy()
    # else:
    #     out = outex_icon[:mask.shape[0], :mask.shape[1]].copy()
    #     inner = intex_icon[:mask.shape[0], :mask.shape[1]].copy()

    # out = out[:mask.shape[0], :mask.shape[1]]
    out[~np.logical_not(mask)] = 0
    out[~np.logical_not(mask)] = 0

    if icon:
        mask_blur = cv2.GaussianBlur(mask, (101, 101), 10) & ~mask
    else:
        mask_blur = cv2.GaussianBlur(mask, (71, 71), 10) & ~mask
    out = out - (np.array([1, 1, 1]) * mask_blur) // 4

    # inner = inner[:mask.shape[0], :mask.shape[1]]
    inner[np.logical_not(mask)] = 0
    inner[np.logical_not(mask)] = 0

    # plt.imshow((inner + out)[:, :, ::-1])

    final = (out + inner)

    final[final < 0] = 0

    final = cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_BGR2RGB)
    final[np.where((door != [0, 0, 0]).all(axis=2))] = [153, 102, 255]
    # final[np.where((living != [0, 0, 0]).all(axis=2))] = livtex[np.where((living != [0, 0, 0]).all(axis=2))]
    final[np.where((bathrooms != [0, 0, 0]).all(axis=2))] = batex[
        np.where((bathrooms != [0, 0, 0]).all(axis=2))]
    final[np.where((kitchen != [0, 0, 0]).all(axis=2))] = kitex[
        np.where((kitchen != [0, 0, 0]).all(axis=2))]
    final[np.where((bedrooms != [0, 0, 0]).all(axis=2))] = betex[
        np.where((bedrooms != [0, 0, 0]).all(axis=2))]
    final[np.where((wall != [0, 0, 0]).all(axis=2))] = waltex[
        np.where((wall != [0, 0, 0]).all(axis=2))]

    return final


def to_disp_mask(in_poly):
    if isinstance(in_poly, list):
        in_poly = MultiPolygon(in_poly)
    p = scale(in_poly, xfact=4, yfact=4, origin=(0, 0))
    return get_mask(p, [4 * 256] * 2)


def post_processing(data):
    inner = data.get('mask', [])
    door = data.get('door_pos', [])
    area = data.get('area', 0)

    poly = Polygon(inner)
    door = Point(door)
    door = get_door_rect(door, poly)

    compound_poly = MultiPolygon([poly, door])

    ai_width = (256, 256)

    poly_ai, door_ai = fix_fit_poly(compound_poly, ai_width, pad=True)

    scaled_ai_poly = scale(poly_ai, xfact=0.8, yfact=0.8,
                           origin=(ai_width[0] / 2, ai_width[1] / 2))
    scaled_ai_door = scale(door_ai, xfact=0.8, yfact=0.8,
                           origin=(ai_width[0] / 2, ai_width[1] / 2))

    scaled_ai_door = get_door_rect(scaled_ai_door.centroid, scaled_ai_poly,
                                   0.8)

    poly_disp_mask = to_disp_mask(scaled_ai_poly)
    door_disp_mask = to_disp_mask(scaled_ai_door)

    ai_channel = get_mask(scaled_ai_poly, ai_width)
    door_channel = scaled_ai_door.buffer(3).intersection(
        scaled_ai_poly).centroid


def generate_textured_image(data, area=140):
    walls = data['wall']
    inner_poly = data['inner']
    door_poly = data['door']
    bedrooms = data['bedroom']
    bathrooms = data['bathroom']
    kitchen = data['kitchen']
    living = data['living']

    # extrude_and_save_multipolygon(walls, inner_poly, door_poly, 27,
    #                               f"objs/output.obj", file_type="obj")

    # convert_obj_to_gltf(f"objs/output.obj",
    #                     f"C:/Demon Home/PlanifyDraw/build/output.gltf")

    walls = walls.buffer(-0.0001)
    bedroom_disp_mask = to_disp_mask(bedrooms)
    bathroom_disp_mask = to_disp_mask(bathrooms)
    kitchen_disp_mask = to_disp_mask(kitchen)
    living_disp_mask = to_disp_mask(living)
    walls_disp_mask = to_disp_mask(walls)
    poly_disp_mask = to_disp_mask(inner_poly)
    door_disp_mask = to_disp_mask(door_poly)

    disp = draw_display_picture(poly_disp_mask, door_disp_mask,
                                bedroom_disp_mask, bathroom_disp_mask,
                                kitchen_disp_mask, living=living_disp_mask,
                                wall=walls_disp_mask, padding=0)

    # write room name on each room using cv2 or pillow and make it center
    # aligned
    pil_image = Image.fromarray(disp)
    draw = ImageDraw.Draw(pil_image)

    def add_room_text(room_poly, message1, message2, room_font, drawer,
                      v_spacing=12,
                      color=(255, 255, 255), stroke_color=(40, 40, 40),
                      text_padding=64):
        if not room_poly.centroid.coords:
            return
        x, y = room_poly.centroid.coords[0]

        _, _, w, h = drawer.textbbox((0, 0), message1, font=room_font)
        if stroke_color is not None:
            stroke_width = 1
        else:
            stroke_width = 0
        drawer.text(
            ((x * 4 + text_padding - w / 2),
             (y * 4 + text_padding - h / 2 - v_spacing)),
            message1, font=room_font, fill=color, stroke_width=stroke_width,
            stroke_fill=stroke_color)
        _, _, w, h = drawer.textbbox((0, 0), message2, font=room_font)
        drawer.text(
            ((x * 4 + text_padding - w / 2),
             (y * 4 + text_padding - h / 2 + v_spacing)),
            message2, font=room_font, fill=color, stroke_width=stroke_width,
            stroke_fill=stroke_color)

    spacing = 12
    padding = 0

    for i, room in enumerate(bedrooms):
        font = ImageFont.truetype("assets/Lato-Medium.ttf", 21)
        add_room_text(room, "Bedroom",
                      f"{int(room.area * area / inner_poly.area)} m²", font,
                      draw, spacing,
                      (255, 255, 255), (0, 0, 0), padding)

    for i, room in enumerate(bathrooms):
        font = ImageFont.truetype("assets/Lato-Heavy.ttf", 17)
        add_room_text(room, "Bathroom",
                      f"{int(room.area * area / inner_poly.area)} m²", font,
                      draw, spacing,
                      (181, 230, 29), (0, 0, 0), padding)

    for i, room in enumerate(kitchen):
        font = ImageFont.truetype("assets/Lato-Medium.ttf", 20)
        add_room_text(room, "Kitchen",
                      f"{int(room.area * area / inner_poly.area)} m²", font,
                      draw, spacing,
                      (70, 40, 70), (200, 150, 100), padding)

    font = ImageFont.truetype("assets/Lato-Medium.ttf", 21)
    add_room_text(living, "Living Area",
                  f"{int(living.area * area / inner_poly.area)} m²", font,
                  draw, spacing,
                  (20, 20, 26), (200, 200, 200), padding)

    disp = np.array(pil_image)
    # save image
    plt.imsave('outputs/output.png', disp)
    return data
