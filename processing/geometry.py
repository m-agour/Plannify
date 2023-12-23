import re

import cv2
import numpy as np
from pandas import Series
from shapely.geometry import Point, MultiPolygon, Polygon
from typing import Iterable

_points_pattern = r'<POINT \((\d+(?:\.\d+)?)\s(\d+(?:\.\d+)?)\)>'


def parse_points(input_str):
    coords_str = re.findall(_points_pattern, input_str)
    points = [Point(float(x), float(y)) for x, y in coords_str]
    return points


def get_mask(poly, shape=(256, 256), point_s=5, line_s=0):
    """ Return image contains multiploygon as a numpy array mask

    Parameters
    ----------
    poly: Polygon or MultiPolygon or Iterable[Polygon or MultiPolygon]
        The Polygon/s to get mask for
    shape: tuple
        The shape of the canvas to draw polygon/s on

    Returns
    -------
    ndarray
        Mask array of the input polygon/s
        :param line_s:
        :param poly:
        :param shape:
        :param point_s:

    """

    img = np.zeros(shape, dtype=np.uint8)
    if isinstance(poly, Polygon):
        if poly.is_empty:
            return img
        if line_s:
            points = np.array(poly.exterior.coords[:], dtype=np.int32)
            img = cv2.polylines(img, [points], True, 255, thickness=line_s)
        else:
            img = cv2.drawContours(img, np.int32([poly.exterior.coords]), -1,
                                   255, -1)

        for interior in poly.interiors:

            points = np.array(interior.coords[:], dtype=np.int32)
            if line_s:
                img = cv2.polylines(img, [points], True, 0, thickness=line_s)
            else:
                img = cv2.drawContours(img, np.int32([interior.coords]), -1, 0,
                                       -1)

    elif isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            if line_s:
                points = np.array(p.exterior.coords[:], dtype=np.int32)
                img = cv2.polylines(img, [points], True, 255, thickness=line_s)
            else:
                img = cv2.drawContours(img, np.int32([p.exterior.coords]), -1,
                                       255, -1)

    elif isinstance(poly, Series):
        polys = [p for p in poly.tolist() if p]
        img = get_mask(polys, shape, point_s, line_s)

    elif isinstance(poly, Iterable):
        for p in poly:
            img = (img != 0) | (get_mask(p, shape, point_s, line_s) != 0)
        img = img.astype(np.uint8) * 255
    elif isinstance(poly, Point):
        p = poly.coords[0]
        img = cv2.circle(img, (int(p[0]), int(p[1])), point_s, 255, -1)
    return img.astype(np.uint8)


def cluster_points(in_points, eps=8, as_dict=False, median=True):
    anc = []
    clusters = []
    if as_dict:
        out_dict = {}
    points_sorted = sorted(in_points)

    curr_point = points_sorted[0]

    curr_cluster = [curr_point]

    for point in points_sorted[1:]:
        if point <= curr_point + eps:
            curr_cluster.append(point)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
            curr_point = point
    clusters.append(curr_cluster)

    for c in clusters:
        if median:
            mn = int(np.median(c))
        else:
            mn = int(np.mean(c))
        anc.append(mn)
        if as_dict:
            out_dict = {**out_dict, **{i: mn for i in c}}
    if as_dict:
        return out_dict

    return anc


def cluster_boxes(boxes):
    new_boxes = []

    xs_lst = []
    ys_lst = []

    for b in boxes:
        xs, ys = b.exterior.coords.xy
        xs_lst.extend(xs)
        ys_lst.extend(ys)

    xs_lst_ = cluster_points(xs_lst, as_dict=True, eps=5)
    ys_lst_ = cluster_points(ys_lst, as_dict=True, eps=5)

    for _box in boxes:
        vertices = np.array(_box.exterior.coords)
        vertices[:, 0] = [xs_lst_[i] for i in vertices[:, 0]]
        vertices[:, 1] = [ys_lst_[i] for i in vertices[:, 1]]
        new_box = Polygon(vertices)
        new_boxes.append(new_box)

    return new_boxes


def b_area(p):
    x1, y1, x2, y2 = p.bounds
    return (x2 - x1) * (y2 - y1)

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