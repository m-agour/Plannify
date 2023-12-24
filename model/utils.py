import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from shapely import Polygon, LineString, MultiPolygon, affinity
from shapely.geometry import box, Point
from shapely.ops import unary_union
from skimage.feature import blob_dog, blob_log, blob_doh
from processing.geometry import get_mask, cluster_boxes, b_area
from processing.image import get_rects
import geopandas as gpd


def process_centroid_generator_model(model, inner_poly, door_poly,
                                     no_bedrooms=3, no_bathrooms=3, area=140,
                                     neighbours_poly=None):
    bedrooms = []
    bathrooms = []
    kitchen = []

    area = min(area / 400, 1)

    x = np.zeros((1, 256, 256, 12))

    x[0, :, :, 0] = get_mask(door_poly.centroid, (256, 256), point_s=5) > 0
    x[0, :, :, 1] = get_mask(inner_poly, (256, 256), point_s=5) > 0

    length_box = 256 * area / 2
    x[0, :, :, 2] = get_mask(
        box(128 - length_box, 128 - length_box, 128 + length_box,
            128 + length_box), (256, 256), point_s=10) > 0
    x[0, :, :, 3] = get_mask([neighbours_poly], (256, 256), point_s=5) > 0
    x[0, :, :, 3 + no_bedrooms] = 1.0
    x[0, :, :, 7 + no_bathrooms] = 1.0

    with torch.no_grad():
        x = torch.from_numpy(x).permute((0, 3, 1, 2)).float()

        bottle_neck = model.forward_encoder_main(x)

        for i in range(no_bedrooms + no_bathrooms + 1):

            x_task = np.zeros((1, 256, 256, 3))

            x_task[0, :, :, 0] = get_mask(bedrooms, (256, 256), point_s=5) > 0
            x_task[0, :, :, 1] = get_mask(bathrooms, (256, 256), point_s=5) > 0
            x_task[0, :, :, 2] = get_mask(kitchen, (256, 256), point_s=5) > 0

            if len(bedrooms) < no_bedrooms:
                state = 1
            elif len(bathrooms) < no_bathrooms:
                state = 2
            else:
                state = 3

            x_task = torch.from_numpy(x_task).permute((0, 3, 1, 2)).float()

            other_features = model.forward_encoder_mini(x_task)

            out = model.forward_decoder(bottle_neck, other_features,
                                        state=state)

            out = out.detach().cpu()[0][0]

            im = out.numpy()
            current = get_mask(bedrooms + bathrooms, point_s=10)
            im[current != 0] = 0
            img = cv2.GaussianBlur(im, (15, 15), 5)

            blobs_log = blob_log(img, min_sigma=4, max_sigma=30,
                                 threshold=0.001)
            bed_centroids = [Point(i[1], i[0]) for i in blobs_log]

            centroids = get_rects(img, bed_centroids, door_poly, get_max=0)

            if len(bedrooms) < no_bedrooms:
                bedrooms.extend(centroids)
            elif len(bathrooms) < no_bathrooms:
                bathrooms.extend(centroids)
            else:
                kitchen = centroids

    return {
        "bedroom": bedrooms,
        "bathroom": bathrooms,
        "kitchen": kitchen
    }


def process_input_boundaries_model(model, inner_poly, door_poly, bedrooms,
                                   bathrooms, kitchen):
    centroids = bedrooms + bathrooms + kitchen
    front = door_poly.centroid
    inner = inner_poly

    bed_centroids = [i for i in bedrooms]
    bath_centroids = [i for i in bathrooms]
    kit_centroids = [i for i in kitchen]

    boxes = []
    for i in range(1):
        boxes = []
        for i, centroid in enumerate(centroids):
            x = np.zeros((1, 5, 256, 256))
            in_boxes = unary_union(boxes)
            x[0, 0] = get_mask(front, (256, 256), point_s=7) > 0
            x[0, 1] = get_mask(inner, (256, 256), point_s=10) > 0
            x[0, 2] = get_mask(centroids, (256, 256), point_s=6) > 0
            x[0, 3] = get_mask(centroid, (256, 256), point_s=6) > 0
            x[0, 4] = get_mask(in_boxes, (256, 256), point_s=6) > 0
            input_tensor = torch.from_numpy(x).float()
            output = model(input_tensor)
            output = output.detach().numpy()[0] * 256
            boxes.append(box(*output))
        # gpd.GeoSeries(
        #     [inner, front] + [i.intersection(inner.buffer(-2, join_style=2))
        #                       for i
        #                       in boxes]).plot(cmap='tab20')
        # plt.show()

        centroids = [i.centroid for i in boxes]

    rooms = sorted(boxes, key=lambda x: x.area, reverse=1)

    buff_amount = 2
    no_wall_inner = inner.buffer(-buff_amount, join_style=2)

    new_boxes = cluster_boxes(boxes)

    new_boxes = sorted(new_boxes, key=lambda x: x.area, reverse=True)

    left_living = no_wall_inner.buffer(0)

    for i, b_curr in enumerate(new_boxes):
        b_curr = b_curr.buffer(buff_amount / 2, join_style=2, cap_style=2)
        for b_next in new_boxes[i + 1:]:
            b_curr = b_curr.difference(
                b_next.buffer(buff_amount / 2, join_style=2, cap_style=2))

        left_living = left_living.difference(b_curr)
        b_curr = b_curr.buffer(-buff_amount / 2, join_style=2,
                               cap_style=2).intersection(no_wall_inner)
        new_boxes[i] = b_curr
    left_living = left_living.buffer(buff_amount / 2, join_style=2,
                                     cap_style=2).intersection(no_wall_inner)

    for i in range(1):
        leftovers = no_wall_inner.difference(
            unary_union(new_boxes + [left_living]))
        other_leftovers = left_living.difference(
            left_living.buffer(-buff_amount * 2, join_style=2,
                               cap_style=2).buffer(
                buff_amount * 2, join_style=2, cap_style=2))
        leftovers = unary_union([leftovers, other_leftovers])
        if isinstance(leftovers, Polygon):
            leftovers = [leftovers]
        else:
            leftovers = list(leftovers.geoms)
        new_leftovers = []
        for i, l_curr in enumerate(leftovers):
            x1, y1, x2, y2 = l_curr.bounds
            area_b = (x2 - x1) * (y2 - y1)
            area_a = l_curr.area
            if area_a / area_b < 0.99:
                xs, ys = l_curr.exterior.coords.xy
                x_cutters = []
                y_cutters = []
                new_pieces = []
                for j in range(len(xs)):
                    x_cutters.append(
                        LineString([(xs[j], -20), (xs[j], 300)]).buffer(
                            0.00001,
                            join_style=2,
                            cap_style=2))
                for j in range(len(ys)):
                    y_cutters.append(
                        LineString([(-20, ys[j]), (300, ys[j])]).buffer(
                            0.00001,
                            join_style=2,
                            cap_style=2))

                for j, x_cutter in enumerate(x_cutters):
                    for k, y_cutter in enumerate(y_cutters):
                        _p = l_curr.difference(x_cutter).difference(y_cutter)
                        if isinstance(_p, Polygon):
                            _p = MultiPolygon([_p])
                        _p = [p for p in _p.geoms if
                              1 or p.area / b_area(p) > 0.99]
                        new_pieces.extend(_p)
                new_pieces = sorted(new_pieces, key=lambda x: x.area,
                                    reverse=False)
                for p in new_pieces:
                    inters = p.intersection(l_curr)
                    if inters.area and inters.area / p.area > 0.9 and b_area(
                            p.intersection(l_curr)) / b_area(p) > 0.99:
                        new_leftovers.append(p)
                        l_curr = l_curr.difference(p)
            else:
                new_leftovers.append(l_curr)

        leftovers = new_leftovers
        if isinstance(left_living, MultiPolygon):
            left_living_single = max(left_living.geoms, key=lambda x: x.area)
            leftovers += [i for i in left_living.geoms if
                          i != left_living_single]
            left_living = left_living_single

        for l_curr in leftovers:
            best_fit_idx = None
            best_fit_ratio = float('inf')
            best_fit = None
            for b_i, b_curr in enumerate(new_boxes):
                b_area_old = b_area(b_curr)
                b_curr_new = b_curr.union(l_curr)
                b_area_new = b_area(b_curr_new)

                ratio = b_area_new / b_area_old
                if ratio < best_fit_ratio:
                    best_fit_ratio = ratio
                    best_fit = b_curr_new.buffer(buff_amount, join_style=2,
                                                 cap_style=2).buffer(
                        -buff_amount,
                        join_style=2,
                        cap_style=2)
                    best_fit_idx = b_i
            new_boxes[best_fit_idx] = best_fit.buffer(buff_amount / 2,
                                                      join_style=2,
                                                      cap_style=2).buffer(
                -buff_amount / 2, join_style=2, cap_style=2).intersection(
                inner)
            new_boxes[best_fit_idx] = new_boxes[best_fit_idx].buffer(
                -buff_amount * 2, join_style=2, cap_style=2).buffer(
                buff_amount * 2, join_style=2, cap_style=2)

    living = no_wall_inner.buffer(0)
    boxes = sorted(new_boxes, key=lambda x: x.area, reverse=True)
    new_boxes = [box(*i.bounds) for i in boxes]

    for i, box_1 in enumerate(new_boxes):
        for j, box_2 in enumerate(new_boxes):

            max_box = box_1
            min_box = box_2

            if min_box.intersection(max_box).area / min_box.area < 0.16:
                min_box = min_box.difference(max_box)
                new_boxes[j] = min_box.intersection(inner)
                new_boxes[i] = max_box.intersection(inner)

        left_living = left_living.difference(new_boxes[i])

    lines = []
    for b in new_boxes + [left_living]:
        exterior_line = b.buffer(buff_amount / 2, join_style=2,
                                 cap_style=2).difference(b)
        lines.append(exterior_line)

    lines.append(inner.difference(no_wall_inner))
    lines = unary_union(lines)

    new_boxes = sorted(new_boxes, key=lambda x: x.area, reverse=True)

    living = no_wall_inner.buffer(0)

    for i, b_curr in enumerate(new_boxes):
        for b_next in new_boxes[i + 1:]:
            b_curr = b_curr.difference(
                b_next.buffer(buff_amount / 2, join_style=2, cap_style=2))

        left_living = left_living.difference(b_curr)
        b_curr = b_curr.intersection(inner)
        new_boxes[i] = b_curr

    left_living = left_living.buffer(buff_amount / 2, join_style=2,
                                     cap_style=2).intersection(
        no_wall_inner).difference(lines)

    gpd.GeoSeries([inner, front] + new_boxes + [lines, left_living]).plot(
        cmap='tab20')
    plt.show()

    out_bedrooms = []
    out_bathrooms = []
    out_kitchen = []

    for bx in new_boxes:
        bxc = bx.centroid
        min_dist = float('inf')
        min_type = None
        for p in bed_centroids:
            if p.distance(bxc) < min_dist:
                min_dist = p.distance(bxc)
                min_type = "bedroom"
        for p in bath_centroids:
            if p.distance(bxc) < min_dist:
                min_dist = p.distance(bxc)
                min_type = "bathroom"
        for p in kit_centroids:
            if p.distance(bxc) < min_dist:
                min_dist = p.distance(bxc)
                min_type = "kitchen"
        if min_type == "bedroom":
            out_bedrooms.append(bx)
        elif min_type == "bathroom":
            out_bathrooms.append(bx)
        else:
            out_kitchen.append(bx)

    return {
        "bedroom": out_bedrooms,
        "bathroom": out_bathrooms,
        "kitchen": out_kitchen,
        "living": left_living,
        "wall": lines,
        "inner": inner,
        "door": front
    }


def get_rooms_counts(model, inner_poly, door_pos, area):
    x = np.zeros((1, 256, 256, 2))

    area_s = (area - 60) / 250

    area_s /= 0.8

    inner_poly = affinity.scale(inner_poly, xfact=area_s, yfact=area_s,
                                origin=(256 / 2, 256 / 2))
    door_pos = affinity.scale(door_pos.buffer(1), xfact=area_s, yfact=area_s,
                              origin=(256 / 2, 256 / 2)).centroid

    x[0, :, :, 0] = get_mask(get_mask(inner_poly), (256, 256), point_s=5) > 0
    x[0, :, :, 1] = get_mask(get_mask(door_pos), (256, 256), point_s=5) > 0

    x = torch.from_numpy(x).permute((0, 3, 1, 2)).float()

    output = model(x).detach().numpy().round()[0].astype(int)

    return output
