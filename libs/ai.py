import time
import uuid
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely import from_wkt, GeometryCollection
from dims_optimizer.algorithm import optimize
from dims_optimizer.geomtry import get_final_layout, fix_floor_plan, plot_series, extrude_and_save_multipolygon
from dims_optimizer.room import get_rooms
import torch
from typing import Iterable
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import cv2
import numpy as np
from pandas import Series
from shapely.geometry import MultiPolygon, Polygon, Point, box
import matplotlib.pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import io
from model import *


def get_mask(poly, shape=(256, 256), point_s=15, line_s=0):
    try:
        img = np.zeros(shape, dtype=np.uint8)
        if isinstance(poly, Polygon):
            if poly.is_empty:
                return img

            if line_s:
                points = np.array(poly.exterior.coords[:], dtype=np.int32)
                img = cv2.polylines(img, [points], True, (0, 255, 0), thickness=2)
            else:
                img = cv2.drawContours(img, np.int32([poly.exterior.coords]), -1, 255, -1)

        elif isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                img = cv2.drawContours(img, np.int32([p.exterior.coords]), -1, 255, -1)

        elif isinstance(poly, Series):
            polys = [p for p in poly.tolist() if p]
            img = get_mask(polys, shape, point_s)

        elif isinstance(poly, Iterable):
            for p in poly:
                img = (img != 0) | (get_mask(p, shape, point_s) != 0)
            img = img.astype(np.uint8) * 255
        elif isinstance(poly, Point):
            p = poly.coords[0]
            img = cv2.circle(img, (int(p[0]), int(p[1])), point_s, 255, -1)
        return img.astype(np.uint8)
    except:
        return img


def get_rooms_number(area):
    # bd, bt = rooms_config_dt_regressor.predict([[area * 0.7]])[0]
    bd, bt = rooms_config_dt_regressor.predict([[area * 0.7]])[0]
    return min(3, int(bd)), min(2, int(bt))


onehot = {}
for i in [1, 2, 3, 4]:
    for j in [1, 2, 3, 4]:
        img = np.zeros((256, 256, 8))
        img[:, :, i - 1] = np.ones((256, 256))
        img[:, :, j + 4 - 1] = 1
        onehot[(i, j)] = img[:, :, :]


def norm_area(area):
    return (area - 30) / 220

def norm_area(area):
    return (area - 52) / 120

def restore_area(area):
    return area * 80 + 52


def get_input(inner, door, bedn, bathn, channels=None, n=None, draw=False, aug=False, aug_bath=False, aug_draw=False,
              unet_1=False, area=120):
    if channels is None:
        channels = []

    d = np.zeros((256, 256, 3))

    d[:, :, 0] = door
    d[:, :, 1] = inner


    input_image = torch.from_numpy(d).permute(2, 0, 1) / 255

    n = len(channels) + 3 if not n else n

    if unet_1:
        oh = torch.zeros((4, 8, 8))
        oh[bedn - 1] = 1.0
        ar = torch.tensor(norm_area(area))
        input_image = input_image[[1, 0], :, :].bool().float(), ar.float(), oh.float()
        return input_image
    if aug_draw:
        first = torch.from_numpy(channels[0]).view(1, 256, 256).bool().float()
        second = torch.from_numpy(channels[1]).view(1, 256, 256).bool().float()
        nu = onehot[(bedn, bathn)]
        input_image = torch.cat((input_image[[1, 0], :, :].bool().float(),
                                 torch.from_numpy(nu).permute(2, 0, 1).bool().float(), first, second), dim=0)
        return input_image
    if aug_bath:
        first = torch.from_numpy(channels[0]).view(1, 256, 256).bool().float()
        nu = onehot[(bedn, bathn)]
        input_image = torch.cat(
            (input_image[[1, 0], :, :].bool().float(), torch.from_numpy(nu).permute(2, 0, 1).bool().float(), first),
            dim=0)
        return input_image
    if aug:
        nu = onehot[(bedn, bathn)]
        input_image = torch.cat(
            (input_image[[1, 0], :, :].bool().float(), torch.from_numpy(nu).permute(2, 0, 1).bool().float()), dim=0)
        return input_image
    # if new_bath:
    #     first = torch.from_numpy(channels[0]).view(1, 256, 256).bool().float()
    #     input_image = torch.cat((input_image[[1, 0], :, :].bool().float(), area_input.float(), first.float()), dim=0)
    #     return input_image
    # if new:
    #     input_image = torch.cat((input_image[[1, 0], :, :].bool().float(), area_input.float()), dim=0)
    #     return input_image

    if draw:
        first = torch.from_numpy(channels[0]).view(1, 256, 256).bool().float()
        second = torch.from_numpy(channels[1]).view(1, 256, 256).bool().float()
        input_image = torch.cat((input_image[[1, 0], :, :].bool().float(), first.float(), second.float()), dim=0)
        return input_image

    d1 = []
    # if not skip_area:
    # d1 = [area_input.float()]
    if n == 4:
        first = torch.from_numpy(channels[0]).view(1, 256, 256).bool().float()
        data = d1 + [first.float()]
    elif n == 3:
        data = d1
    elif n == 5:
        first = torch.from_numpy(channels[0]).view(1, 256, 256).bool().float()
        second = torch.from_numpy(channels[1]).view(1, 256, 256).bool().float()
        data = d1 + [first.float(), second.float()]
    input_image = torch.cat((input_image[[1, 0], :, :].bool().float(), *data), dim=0)
    return input_image


def imfy(img):
    im = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    # plt.imshow(im)
    # plt.show()
    return im


def get_rects(imgr, centroids):
    bounds = get_mask(centroids, point_s=8).astype(np.uint8)
    img = (imgr).astype(np.uint8)
    #     img[bounds>0] = 0

    img[np.where(bounds == 0)] = 0

    shapes = []
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(cnts[0]):
        M = cv2.moments(c)
        box = cv2.boundingRect(c);
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h

        ii = imgr[y1:y2, x1:x2]

        intensity = np.mean(ii)

        if not M["m00"]:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shapes.append([cx, cy, intensity])

    return [Point(*i[:2]) for i in sorted(shapes, key=lambda x: x[2], reverse=True)]


def get_rects_2(channel, gcenters=True):
    a = channel.copy()

    a[a < 0.9] = 0
    a[a > 1] = 1

    a = (255 * a).astype(np.uint8)

    _, thresh = cv2.threshold(a, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_c = []
    centers = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        shapely_box = box(x, y, x + w, y + h)
        final_c.append(shapely_box.buffer(5, join_style=2, cap_style=2))

        mask = np.zeros((256, 256))
        mask = cv2.drawContours(mask, [c], -1, 255, -1)

        rms = channel.copy()
        rms[mask == 0] = 0

        M = cv2.moments(rms)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append(Point(cX, cY))

    if gcenters:
        return centers
    return MultiPolygon(final_c).buffer(0)


def get_other_tensor(first=None, second=None):
    arr = np.zeros((2, 256, 256))
    if first is not None:
        arr[0] = first
    if second is not None:
        arr[1] = second
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    return tensor


def perturb_polygon(polygon, x_range=(-2, 2), y_range=(-2, 2)):
    """
    Apply random perturbation to the coordinates of a shapely polygon.

    Args:
        polygon (Polygon): The original shapely polygon.
        x_range (tuple): The range for random perturbation in x-axis. Default is (-0.1, 0.1).
        y_range (tuple): The range for random perturbation in y-axis. Default is (-0.1, 0.1).

    Returns:
        Polygon: The perturbed shapely polygon.
    """
    # Get the coordinates of the original polygon
    original_coords = np.array(polygon.exterior.coords)

    # Iterate through each vertex of the original polygon
    perturbed_coords = []
    for x, y in original_coords:
        # Generate a random perturbation for x and y coordinates within the specified range
        perturbation_x = np.random.uniform(x_range[0], x_range[1])
        perturbation_y = np.random.uniform(y_range[0], y_range[1])

        # Apply the perturbation to the original coordinates
        perturbed_x = x + perturbation_x
        perturbed_y = y + perturbation_y

        perturbed_coords.append((perturbed_x, perturbed_y))

    # Create a new Polygon object with the perturbed coordinates
    perturbed_polygon = Polygon(perturbed_coords)

    return perturbed_polygon


nbedd = {f"bed{i + 1}": i for i in range(3)}
nbathd = {f"bath{i + 1 - 3}": i for i in range(3, 6)}
nd = {**nbedd, **nbathd}


def get_input_latest(inner, door, bedn, bathn, task, area, bedrooms=[], bathrooms=[]):
    s = 0.8

    perturb_range = 0

    # inner = perturb_polygon(inner, x_range=(-perturb_range, perturb_range),
    #                         y_range=(-perturb_range, perturb_range)).buffer(0)

    noif = 2

    no_bedrooms = bedn
    no_bathrooms = bathn

    x = np.zeros((256, 256, 20))

    front = door.buffer(3).intersection(inner).centroid

    bedroomsc = bedrooms
    bathroomsc = bathrooms

    x[:, :, 0] = get_mask(front, (256, 256), point_s=7) > 0
    x[:, :, 1] = get_mask(inner, (256, 256), point_s=10) > 0

    if 'kit' not in task:
        x[:, :, nd[task] + 2] = 1.0  # 2, 3, 4, 5 ,6, 7

    bed_start = 9  # 9 10 11
    bath_start = 12  # 12 13 14
    num_start = 15  # 15 16 17
    bed_m, bath_m = 18, 19

    if 'bed' in task:
        bnn = int(task[3])
        x[:, :, bnn + bed_start - 1] = 1.0
        x[:, :, num_start + no_bedrooms - 1][:bnn] = 1.0
        x[:, :, bed_m] = get_mask(bedroomsc[:bnn - 1], (256, 256), point_s=6) > 0

    elif 'bath' in task:
        btn = int(task[4])
        x[:, :, btn + bath_start - 1] = 1.0
        x[:, :, num_start + no_bathrooms - 1] = 1.0
        x[:, :, bath_m] = get_mask(bathroomsc[:btn - 1], (256, 256), point_s=6) > 0
        x[:, :, bed_m] = get_mask(bedroomsc, (256, 256), point_s=6) > 0

    elif 'kit' in task:
        x[:, :, bath_m] = get_mask(bathroomsc, (256, 256), point_s=6) > 0
        x[:, :, bed_m] = get_mask(bedroomsc, (256, 256), point_s=6) > 0

    #     area += random.randint(-10, 10)

    x[:, :, 8] = norm_area(area)

    return torch.from_numpy(x).permute(2, 0, 1).float()


import numpy as np
from scipy.signal import convolve2d


def get_rects(imgr, centroids, door, get_max=True):
    bounds = get_mask(centroids, point_s=8).astype(np.uint8)
    d = ~get_mask(door.centroid, point_s=30).astype(np.uint8)
    #     imgr[:, :, 0] = (imgr[:, :, 0] - d).astype(np.uint8)
    #     plt.imshow(imgr)
    #     plt.gca().invert_yaxis()
    #     plt.show()
    img = (imgr).astype(np.uint8)
    img[np.where(bounds == 0)] = 0

    if get_max:
        kernel = np.ones((5, 5))
        convolved = convolve2d(imgr.reshape((256, 256)), kernel, mode='valid')
        # plt.imshow(imgr)
        # plt.show()
        y, x = np.unravel_index(np.argmax(convolved), convolved.shape)
        return [Point(x, y)]

    shapes = []
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(cnts[0]):
        M = cv2.moments(c)
        box = cv2.boundingRect(c);
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h

        ii = img[y1:y2, x1:x2]

        ii = ii * (ii >= np.median(imgr))
        intensity = np.sum(ii)

        if not M["m00"]:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shapes.append([cx, cy, intensity])

    return [Point(*i[:2]) for i in sorted(shapes, key=lambda x: x[2], reverse=True)]


i1 = None


def get_centroid(inner, door, bedn, bathn, task, area, bedrooms=[], bathrooms=[], plot=True, model=None):
    global i1
    x1 = get_input_latest(inner=inner, door=door, bedn=bedn, bathn=bathn, task=task, area=area,
                          bedrooms=bedrooms, bathrooms=bathrooms).unsqueeze(0)
    o1 = model(x1.to(device)).cpu() * x1[:, 1, :, :]
    inp3 = o1[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, :]
    if task == 'bed1':
        i1 = inp3
    im = inp3
    img = cv2.GaussianBlur(im, (15, 15), 5)
    blobs_log = blob_log(img, min_sigma=4, max_sigma=30, threshold=0.01)
    bed_centroids = [Point(i[1], i[0]) for i in blobs_log]
    bed_centroids = get_rects(im, bed_centroids, door)[0]

    if plot:
        inp1 = x1[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, [19, 1, 18]]
        inp1[:, :, 1] = inp1[:, :, 1] - inp1[:, :, 0]
        inp1[:, :, 1] = inp1[:, :, 1] - inp1[:, :, 2]
        inp1[:, :, [0, 2]] += x1[0].permute(1, 2, 0).cpu().detach().numpy()[:, :, [0, 0]]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # display each image on each subplot
        axs[0].imshow(inp1, cmap='inferno')
        axs[0].set_title("Input")

        axs[1].imshow(inp3, cmap='inferno')
        axs[1].set_title("Output")

        bed_centroids_in = get_mask(bed_centroids, point_s=5)
        axs[2].imshow(bed_centroids_in, cmap='inferno')
        axs[2].set_title("Segmented output")

        for a in axs:
            a.invert_yaxis()

        fig.suptitle(task, fontsize=16)  # set title for the whole figure
        plt.show()
    #
    return bed_centroids


def get_polys(data):
    bedrooms = data['bedrooms']
    bathrooms = data['bathrooms']
    kitchen = data['kitchen']
    door_poly = data['door']
    inner_poly = data['inner']

    bedrooms_list = get_rooms(bedrooms)
    bathrooms_list = get_rooms(bathrooms + kitchen)
    for b in bathrooms_list[:-1]:
        b.kitchen = False
    bathrooms_list[-1].kitchen = True
    bedrooms, bathrooms = optimize(inner_poly, bedrooms_list, bathrooms_list, door_poly, step=0.1)

    polys = fix_floor_plan(inner_poly, bedrooms, bathrooms, door_poly, plot=0.1)

    series = gpd.GeoSeries(polys)

    bedrooms, bathrooms = optimize(inner_poly, bedrooms_list, bathrooms_list, door_poly, plot=0.1)
    polys = fix_floor_plan(inner_poly, bedrooms, bathrooms, door_poly, plot=0.0, as_series=True)

    walls = polys.loc["walls"]
    door = polys.loc["door"]
    bedrooms = polys.loc["bedrooms"]
    bathrooms = polys.loc["bathrooms"]
    kitchen = polys.loc["kitchen"]

    series = gpd.GeoSeries(polys)
    # series.plot(cmap="tab20")

    extrude_and_save_multipolygon(walls, inner_poly, door, 27, f"objs/output.obj", file_type="obj")

    import subprocess

    def convert_obj_to_gltf(obj_file_path, gltf_file_path):
        command = f'obj2gltf -i "{obj_file_path}" -o "{gltf_file_path}"'
        print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        # print error
        for line in process.stdout:
            print(line.decode("utf-8").replace("\n", ""))

    convert_obj_to_gltf(f"objs/output.obj", f"C:/Demon Home/PlanifyDraw/build/output.gltf")

    p_dict = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "kitchen": kitchen,
        "walls": walls,
    }
    return p_dict


def get_input_wh(current, inner, door, bedc, otherc):
    x = np.zeros((256, 256, 5))
    front = door.centroid
    x[:, :, 0] = get_mask(front, (256, 256), point_s=7) > 0
    x[:, :, 1] = get_mask(inner, (256, 256), point_s=10) > 0
    x[:, :, 2] = get_mask(bedc, (256, 256), point_s=6) > 0
    x[:, :, 3] = get_mask(otherc, (256, 256), point_s=6) > 0
    x[:, :, 4] = get_mask(current, (256, 256), point_s=6) > 0

    return torch.from_numpy(x).permute(2, 0, 1).float()


class Room:
    def __init__(self, minx, miny, maxx, maxy, ccw=True, typ=None):
        coords = [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
        if not ccw:
            coords = coords[::-1]
        self.polygon = Polygon(coords)
        self.typ = typ

    def intersection(self, other):
        if isinstance(other, Room):
            self.polygon = self.polygon.intersection(other.polygon)
        elif isinstance(other, Polygon):
            self.polygon = self.polygon.intersection(other)
        return self

    def difference(self, other):
        if isinstance(other, Room):
            self.polygon = self.polygon.difference(other.polygon)
        elif isinstance(other, Polygon):
            self.polygon = self.polygon.difference(other)
        return self

    def union(self, other):
        if isinstance(other, Room):
            self.polygon = self.polygon.union(other.polygon)
        elif isinstance(other, Polygon):
            self.polygon = self.polygon.union(other)
        return self

    def buffer(self, distance):
        new = Room(0, 0, 0, 0)
        new.polygon = self.polygon.buffer(distance, join_style=2, cap_style=2)
        new.typ = self.typ
        return new


import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import geopandas as gpd

from libs.ai import Room
from shapely.ops import unary_union
from shapely import MultiPolygon, Polygon


def fit_p(addition, room):
    inter = room.buffer(2.5, cap_style=2, join_style=2).intersection(addition.buffer(2.5, cap_style=2, join_style=2))
    room = room.union(inter).buffer(-5, cap_style=2, join_style=2).buffer(9, cap_style=2, join_style=2).buffer(-4,
                                                                                                               cap_style=2,
                                                                                                               join_style=2)
    return room


def buffer(polygon, distance):
    return polygon.buffer(distance, cap_style=2, join_style=2)


def buffer_up_down(polygon, distance):
    return polygon.buffer(distance, cap_style=2, join_style=2).buffer(-distance, cap_style=2, join_style=2)


def buffer_down_up(polygon, distance):
    p = polygon.buffer(-distance, cap_style=2, join_style=2).buffer(distance, cap_style=2, join_style=2)
    if isinstance(p, MultiPolygon):
        p = max(p, key=lambda x: x.area)
    return p


def buffer_up_down_up(polygon, distance):
    return polygon.buffer(distance, cap_style=2, join_style=2).buffer(-distance * 2, cap_style=2, join_style=2).buffer(
        distance, cap_style=2, join_style=2)


def buffer_down_up_down(polygon, distance):
    return polygon.buffer(-distance, cap_style=2, join_style=2).buffer(distance * 2, cap_style=2, join_style=2).buffer(
        -distance, cap_style=2, join_style=2)


def get_wh_polys(data):
    bedrooms = data['bedrooms']
    bathrooms = data['bathrooms']
    kitchen = data['kitchen']
    door_poly = data['door']
    inner_poly = data['inner']

    bedc = bedrooms
    otherc = bathrooms + kitchen
    alless = bedc + otherc

    boxes = []

    wall_width = 3
    room_wall_width = 3

    def get_box(centroid, inner_poly, door_poly, typ):
        x = get_input_wh(centroid, inner_poly, door_poly, bedc, otherc).unsqueeze(0)
        y = xxyy_model(x.to(device)).detach().cpu().numpy()[0]
        x1, y1, x2, y2 = y * 256
        b = Room(x1, y1, x2, y2, typ=typ)
        return b

    for i in bedrooms:
        boxes.append(get_box(i, inner_poly, door_poly, typ="bedroom"))

    for i in bathrooms:
        boxes.append(get_box(i, inner_poly, door_poly, typ="bathroom"))

    for i in kitchen:
        boxes.append(get_box(i, inner_poly, door_poly, typ="kitchen"))

    boxes = [i.intersection(inner_poly) for i in boxes]

    boxes = sorted(boxes, key=lambda x: x.polygon.area, reverse=True)

    # plot

    for i in range(len(boxes) - 1):
        new_b = boxes[i]
        for j in range(i + 1, len(boxes)):
            new_b = new_b.difference(boxes[j])
        boxes[i] = new_b.buffer(-5).buffer(5)
        boxes[j] = boxes[j].buffer(-5).buffer(5)

    bedrooms = [i.polygon for i in boxes if i.typ == "bedroom"]
    bathrooms = [i.polygon for i in boxes if i.typ == "bathroom"]
    kitchen = [i.polygon for i in boxes if i.typ == "kitchen"]

    inner_actual = inner_poly.buffer(-wall_width, cap_style=2, join_style=2)

    for i in range(3):
        inner_small = inner_poly.buffer(-wall_width, cap_style=2, join_style=2)
        fits = []
        for p in bedrooms + bathrooms + kitchen:
            inner_small = inner_small.difference(p)
        inner_small_shaved = inner_small.buffer(-8, cap_style=2, join_style=2)
        if isinstance(inner_small_shaved, Polygon):
            inner_small_shaved = MultiPolygon([inner_small_shaved])

        inner_small_shaved = max(inner_small_shaved.geoms, key=lambda x: x.area).buffer(9, cap_style=2, join_style=2)

        pieces = inner_small.difference(inner_small_shaved)

        if isinstance(pieces, Polygon):
            pieces = MultiPolygon([pieces])

        for lo in pieces.geoms:
            x1, y1, x2, y2 = lo.bounds
            w, h = x2 - x1, y2 - y1
            x1, y1, x2, y2 = lo.bounds
            area = (x2 - x1) * (y2 - y1)
            score = lo.area / area
            if (max(w, h) / min(w, h) < 3 and score > 0.8) or (min(w, h) > 10 and lo.area > 10 * 4000):
                continue
            best_fit = None
            best_fit_area_barea = float("inf")
            for p in bedrooms + bathrooms + kitchen:
                inter = lo.union(p.buffer(1, cap_style=2, join_style=2))
                area = inter.area
                x1, y1, x2, y2 = inter.bounds
                barea = (x2 - x1) * (y2 - y1)
                if (barea - area) / area < best_fit_area_barea:
                    best_fit = p
                    best_fit_area_barea = (barea - area) / area
            fits.append([best_fit, lo])

        for i in range(len(fits)):
            for j in range(len(bedrooms)):
                if fits[i][0] == bedrooms[j]:
                    bedrooms[j] = fit_p(fits[i][1], bedrooms[j])
            for j in range(len(bathrooms)):
                if fits[i][0] == bathrooms[j]:
                    bathrooms[j] = fit_p(fits[i][1], bathrooms[j])
            for j in range(len(kitchen)):
                if fits[i][0] == kitchen[j]:
                    kitchen[j] = fit_p(fits[i][1], kitchen[j])
    alles = []
    #

    all_rooms = unary_union(bedrooms + bathrooms + kitchen)
    living = inner_actual.difference(all_rooms)
    if isinstance(living, MultiPolygon):
        living = max(living.geoms, key=lambda x: x.area)

    for p in bedrooms:
        room = Room(0, 0, 0, 0, typ="bedroom")
        room.polygon = p
        alles.append(room)
    for p in bathrooms:
        room = Room(0, 0, 0, 0, typ="bathroom")
        room.polygon = p
        alles.append(room)
    for p in kitchen:
        room = Room(0, 0, 0, 0, typ="kitchen")
        room.polygon = p
        alles.append(room)

    alles = sorted(alles, key=lambda x: x.polygon.area, reverse=True)

    for i in range(len(alles) - 1):
        new_b = alles[i]
        for j in range(i + 1, len(alles)):
            new_b = new_b.difference(alles[j].buffer(room_wall_width))
        alles[i] = new_b
        alles[j] = alles[j]

    outer_wall = inner_poly.difference(inner_actual)
    bedrooms = [buffer_down_up(buffer_up_down(i.polygon, 5), 5).difference(outer_wall) for i in alles if i.typ == "bedroom"]
    bathrooms = [buffer_down_up(buffer_up_down(i.polygon, 5), 5).difference(outer_wall) for i in alles if i.typ == "bathroom"]
    kitchen = [buffer_down_up(buffer_up_down(i.polygon, 5), 5).difference(outer_wall) for i in alles if i.typ == "kitchen"]

    all_space = sum([bedrooms, bathrooms, kitchen], [])
    walls = unary_union([i.buffer(room_wall_width, cap_style=2, join_style=2).difference(i) for i in all_space] + [
        outer_wall]).intersection(inner_poly)

    living = inner_poly.difference(unary_union(all_space + [walls])).difference(outer_wall)

    if isinstance(living, MultiPolygon):
        l = max(living.geoms, key=lambda x: x.area)
        others = unary_union([i for i in living.geoms if i != l])
        walls = walls.union(others)
        living = l

    all_space_buffered = [inner_poly, door_poly] + [i for i in all_space]
    # gpd.GeoSeries(all_space_buffered).plot(cmap="tab10")
    try:
        gpd.GeoSeries([GeometryCollection(i) for i in [[inner_poly], [door_poly], bedrooms, bathrooms, kitchen]]).plot(cmap="tab10")
        plt.show()
    except:
        ...
    data = {
        "inner": inner_poly,
        "living": living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "kitchen": kitchen,
        "walls": walls,
        "door": door_poly,
        # "series": series
    }
    return data


def get_design_single(inner, door, area, inner_poly, door_poly):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    bed_n, bath_n = get_rooms_number(area)

    print(area, bed_n, bath_n)

    bedrooms_predicted_centroids = []
    bathrooms_predicted_centroids = []
    kitchen_predicted_centroids = []
    # bed_n = 2
    # bath_n = 2
    area = 110
    p=0
    door_centroid = door_poly.centroid
    with torch.no_grad():
        for i in range(bed_n):
            task = f"bed{i + 1}"
            c = get_centroid(inner_poly, door_centroid, model=model, bedn=bed_n, bathn=bath_n, task=task, area=area,
                             bedrooms=bedrooms_predicted_centroids,
                             bathrooms=bathrooms_predicted_centroids, plot=p)
            bedrooms_predicted_centroids.append(c)

        for i in range(bath_n):
            task = f"bath{i + 1}"
            c = get_centroid(inner_poly, door_centroid, model=model, bedn=bed_n, bathn=bath_n, task=task, area=area,
                             bedrooms=bedrooms_predicted_centroids,
                             bathrooms=bathrooms_predicted_centroids, plot=p)
            bathrooms_predicted_centroids.append(c)

        task = f"kit"
        c = get_centroid(inner_poly, door_centroid, model=model, bedn=bed_n, bathn=bath_n, task=task, area=area,
                         bedrooms=bedrooms_predicted_centroids,
                         bathrooms=bathrooms_predicted_centroids, plot=p)
        kitchen_predicted_centroids.append(c)
    # task = f"kit"
    # c = get_centroid(inner_poly, door_centroid, model=model, bedn=bed_n, bathn=bath_n, task=task, area=area,
    #                  bedrooms=bedrooms_predicted_centroids,
    #                  bathrooms=bathrooms_predicted_centroids + kitchen_predicted_centroids, plot=1)

    # series = gpd.GeoSeries([GeometryCollection(i) for i in [[inner_poly], [door_poly], bedrooms_predicted_centroids, bathrooms_predicted_centroids, kitchen_predicted_centroids]]).plot(cmap="tab10")
    # series.plot(cmap="Dark ")
    # plt.show()

    return {
        "bedrooms": bedrooms_predicted_centroids,
        "bathrooms": bathrooms_predicted_centroids,
        "kitchen": kitchen_predicted_centroids,
        "door": door_poly,
        "inner": inner_poly,
    }

    #
    # other = get_other_tensor()
    # input_image, area_tens, onehot_tens = get_input(inner, door, bedn, bathn, unet_1=True)
    # encoder_features = aio_model.get_encoder_features(input_image.unsqueeze(0))
    # x = aio_model(input_image.unsqueeze(0), area_tens.unsqueeze(0), onehot_tens.unsqueeze(0), other, encoder_features=encoder_features, decoder_n=0)[0].permute(1, 2, 0).detach().numpy()
    # im = imfy(x[:, :, 0])
    # img = cv2.GaussianBlur(im, (15, 15), 5)
    # blobs_log = blob_log(img, min_sigma=4, max_sigma=30, threshold=0.04)
    # bed_centroids = [Point(i[1], i[0]) for i in blobs_log]
    # bed_centroid_m = get_rects(im, bed_centroids)[0]
    # bed_centroids_in = get_mask(bed_centroids, point_s=8)
    #
    # if bedn >= 2:
    #     other = get_other_tensor(bed_centroids_in)
    #     x = aio_model(input_image.unsqueeze(0), area_tens.unsqueeze(0), onehot_tens.unsqueeze(0), other, encoder_features=encoder_features, decoder_n=1)[0].permute(1, 2, 0).detach().numpy()
    #     im = imfy(x[:, :, 0])
    #     img = cv2.GaussianBlur(im, (15, 15), 5)
    #     blobs_log = blob_log(img, min_sigma=4, max_sigma=30, threshold=0.01)
    #     bed_centroids = [Point(i[1], i[0]) for i in blobs_log]
    #     bed_centroids = [get_rects(im, bed_centroids)[0], bed_centroid_m]
    #     bed_centroids_in = get_mask(bed_centroids, point_s=8)
    #
    # if bedn >= 3:
    #     other = get_other_tensor(bed_centroids_in)
    #     x = aio_model(input_image.unsqueeze(0), area_tens.unsqueeze(0), onehot_tens.unsqueeze(0), other, encoder_features=encoder_features, decoder_n=2)[0].permute(1, 2, 0).detach().numpy()
    #     im = imfy(x[:, :, 0])
    #     img = cv2.GaussianBlur(im, (15, 15), 5)
    #     blobs_log = blob_log(img, min_sigma=4, max_sigma=30, threshold=0.01)
    #     bed_centroids_l = [Point(i[1], i[0]) for i in blobs_log]
    #     bed_centroids = [get_rects(im, bed_centroids_l)[0]] + bed_centroids
    #     bed_centroids_in = get_mask(bed_centroids, point_s=8)
    #
    # print(bedn, len(bed_centroids))
    # other = get_other_tensor(bed_centroids_in)
    # x = aio_model(input_image.unsqueeze(0), area_tens.unsqueeze(0), onehot_tens.unsqueeze(0), other, encoder_features=encoder_features, decoder_n=3)[0].permute(1, 2, 0).detach().numpy()
    # im = imfy(x[:, :, 0])
    # img = cv2.GaussianBlur(im, (15, 15), 5)
    # blobs_log = blob_log(img, min_sigma=4, max_sigma=30, threshold=0.01)
    # bath_centroids = [Point(i[1], i[0]) for i in blobs_log]
    # bath_centroids = bath_centroid_1 = get_rects(im, bath_centroids)[0]
    # bath_centroids_in = get_mask(bath_centroids, point_s=6)
    #
    #
    # if bathn >= 2:
    #     other = get_other_tensor(bed_centroids_in, bath_centroids_in)
    #     x = aio_model(input_image.unsqueeze(0), area_tens.unsqueeze(0), onehot_tens.unsqueeze(0), other,
    #                   encoder_features=encoder_features, decoder_n=4)[0].permute(1, 2, 0).detach().numpy()
    #     im = imfy(x[:, :, 0])
    #     img = cv2.GaussianBlur(im, (15, 15), 5)
    #     blobs_log = blob_log(img, min_sigma=4, max_sigma=30, threshold=0.01)
    #     bath_centroids = [Point(i[1], i[0]) for i in blobs_log]
    #     bath_centroids = [get_rects(im, bath_centroids)[0] , bath_centroid_1]
    #     bath_centroids_in = get_mask(bath_centroids, point_s=6)
    #
    #
    # other = get_other_tensor(bed_centroids_in, bath_centroids_in)
    # x = aio_model(input_image.unsqueeze(0), area_tens.unsqueeze(0), onehot_tens.unsqueeze(0), other,
    #               encoder_features=encoder_features, decoder_n=5)[0].permute(1, 2, 0).detach().numpy()
    # # x = bed_b_model(input_image.reshape(1, 2, 256, 256))[0].permute(1, 2, 0).detach().numpy()
    # im = imfy(x[:, :, 0])
    # img = cv2.GaussianBlur(im, (15, 15), 5)
    # blobs_log = blob_log(img, min_sigma=4, max_sigma=30, threshold=0.01)
    # kit_centroids = [Point(i[1], i[0]) for i in blobs_log]
    # kit_centroids = [get_rects(im, kit_centroids)[0]]
    # kit_centroids_in = get_mask(kit_centroids, point_s=8)
    #
    # bed_centroids_in = get_mask(bed_centroids, point_s=8) > 0
    # bath_centroids_in = get_mask(bath_centroids, point_s=5) > 0
    # input_image = get_input(inner, door, bedn, bathn, channels=[bed_centroids_in, bath_centroids_in], aug_draw=True)
    # x = draw_model(input_image.reshape(1, 12, 256, 256))
    # max_channels = torch.max(x, dim=1)[1]
    #
    # x[:, 4] *= 3
    # x[:, 0][max_channels != 0] = 0
    # x[:, 1][max_channels != 1] = 0
    # x[:, 2][max_channels != 2] = 0
    # x[:, 3][max_channels != 3] = 0
    # x[:, 4][max_channels != 4] = 0
    # x[:, 2] += x[:, 3]
    # x[:, 0] += x[:, 4]
    # x[:, 1] += x[:, 4]
    # x[:, 2] += x[:, 4]
    #
    # x = (x > 0.5).bool().float()
    # x = x[0].permute(1, 2, 0).detach().numpy()
    # x[x < 0] = 0
    # x[x > 1] = 1
    #
    #
    # print(bed_centroids)
    # print(bath_centroids)
    # print(inner_poly)
    # print(door_poly)
    # import geopandas as gpd
    # door_poly, inner_poly = inner_poly, door_poly
    # bedrooms_list = get_rooms(bed_centroids)
    # bathrooms_list = get_rooms(bath_centroids)
    # print(inner_poly)
    # print(door_poly)
    # print(bed_centroids)
    # print(bath_centroids)
    # print(kit_centroids)
    #
    # bedrooms, bathrooms = optimize(inner_poly, bedrooms_list, bathrooms_list, door_poly, step=0.1)
    # polys = fix_floor_plan(inner_poly, bedrooms, bathrooms, door_poly, plot=0.1)
    # series = gpd.GeoSeries(polys)
    #
    #
    # bedrooms, bathrooms = optimize(inner_poly, bedrooms_list, bathrooms_list, door_poly, plot=0.1)
    # polys = fix_floor_plan(inner_poly, bedrooms, bathrooms, door_poly, plot=0.0, as_series=True)
    #
    # walls = polys.loc["walls"]
    # door = polys.loc["door"]
    # print(walls)
    # series = gpd.GeoSeries(polys)
    #
    # random_name = f"C:\Demon Home\PlanifyDraw\public\out_1 - Copy.gltf"
    # extrude_and_save_multipolygon(walls, inner_poly, door, 27, random_name, file_type="gltf")
    #
    #
    # # plot_series(series, sleep=3.0)
    # series.plot(cmap="Dark2")
    # plt.show()
    #
    # print("Done")
    # time.sleep(10)
    #
    # x[:, :, 0] = x[:, :, 0] - kit_centroids_in[:, :]
    # x[:, :, 1] = x[:, :, 1] - kit_centroids_in[:, :]
    # x[:, :, 2] = x[:, :, 2] - kit_centroids_in[:, :]
    #
    # plt.imshow((x[:, :, :3] * 255).astype(np.uint8))
    # plt.show()
    # return (x[:, :, :3] * 255).astype(np.uint8)
    #
    # # stop here
    # img = x.copy()
    # for _ in range(1):
    #     bed_rects = get_rects_2(img[:, :, 0])
    #     bath_rects = get_rects_2(img[:, :, 1])
    #     bed_centroids_in = get_mask([i for i in bed_rects], point_s=8) > 0
    #     bath_centroids_in = get_mask([i for i in bath_rects], point_s=5) > 0
    #     input_image = get_input(inner, door, bedn, bathn, channels=[bed_centroids_in, bath_centroids_in], aug_draw=1)
    #     img = draw_model(input_image.reshape(1, 12, 256, 256))
    #
    #     max_channels = torch.max(img, dim=1)[1]
    #     img[:, 0][max_channels != 0] = 0
    #     img[:, 1][max_channels != 1] = 0
    #     img[:, 2][max_channels != 2] = 0
    #     img[img > 1] = 1
    #     img[img < 0] = 0
    #     img = img[0].permute(1, 2, 0).detach().numpy()
    #
    # bed_centroids_in = get_mask(bed_centroids, point_s=8) > 0
    # bath_centroids_in = get_mask(bath_centroids, point_s=5) > 0
    # input_image = get_input(inner, door, bedn, bathn, channels=[bed_centroids_in, bath_centroids_in], aug_draw=1)
    #
    # img = draw_model(input_image.reshape(1, 12, 256, 256))[0].permute(1, 2, 0).detach().numpy()
    #
    # bed_rects = get_rects_2(img[:, :, 0], 0)
    # bath_rects = get_rects_2(img[:, :, 1], 0)
    #
    # bed_rects = bed_rects - bath_rects.buffer(2, join_style=2, cap_style=2)
    # rr = 0.7 * get_mask(bed_rects) + 0.5 * get_mask(bath_rects) + 0.25 * input_image[1, :, :].numpy() * 255 + 0.35 * input_image[0, :, :].numpy() * 255
    #
    # plt.imshow(rr)
    # plt.show()
aio_model = None

def get_design_enc(inner, door, area, inner_poly, door_poly):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    bed_n, bath_n = get_rooms_number(area)
    bedn, bathn = bed_n, bath_n


    print(area, bed_n, bath_n)

    def get_next_centroid(input_image, encoder_features, area_tens, oh_tensor, bed_centroids=[],
                          bath_centroids=[], decoder_n=0):

        bed_centroids_in = get_mask(bed_centroids, point_s=8) > 0
        bath_centroids_in = get_mask(bath_centroids, point_s=5) > 0


        other = get_other_tensor(bed_centroids_in, bath_centroids_in)

        x = aio_model(input_image.unsqueeze(0), area_tens.unsqueeze(0), oh_tensor.unsqueeze(0), other,
                       decoder_n=decoder_n)[0].permute(1, 2, 0).detach().cpu().numpy()


        im = imfy(x[:, :, 0])
        img = cv2.GaussianBlur(im, (15, 15), 5)
        blobs_log = blob_log(img, min_sigma=4, max_sigma=30, threshold=0.04)
        cent = [Point(i[1], i[0]) for i in blobs_log]
        if cent:
            cent =  get_rects(im, bed_centroids, door_poly)
            return cent[0] if cent else None
        return None

    input_image, area_tens, onehot_tens = get_input(inner, door, bedn, bathn, unet_1=True)

    # encoder_features = aio_model.get_encoder_features(input_image.unsqueeze(0))
    encoder_features = None


    bed_centroids = []
    bath_centroids = []
    kit_centroids = []

    for i in range(min(bedn, 3)):
        bed_centroid = get_next_centroid(input_image, encoder_features,area_tens, onehot_tens,
                                         bed_centroids, bath_centroids, decoder_n=i)
        if bed_centroid:
            bed_centroids.append(bed_centroid)
        else:
            break

    for i in range(min(bathn, 2)):
        bath_centroid = get_next_centroid(input_image, encoder_features,  area_tens, onehot_tens,
                                          bed_centroids, bath_centroids, decoder_n=i+3)
        if bath_centroid:
            bath_centroids.append(bath_centroid)
        else:
            break

    kit_centroid = get_next_centroid(input_image, encoder_features, area_tens, onehot_tens,
                                     bed_centroids, bath_centroids, decoder_n=5)
    if kit_centroid:
        kit_centroids.append(kit_centroid)

    series = [GeometryCollection(i) for i in [[door_poly], [inner_poly], bed_centroids, bath_centroids, kit_centroids]]
    series = gpd.GeoSeries(series)
    # series.plot(cmap='tab10')
    # plt.show()
    #
    return {
        "bedrooms": bed_centroids,
        "bathrooms": bath_centroids,
        "kitchen": kit_centroids,
        "door": door_poly,
        "inner": inner_poly
    }
