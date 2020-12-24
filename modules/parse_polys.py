'''
Reference code from the author Wovchena
https://github.com/Wovchena/text-detection-fots.pytorch
'''

import cv2
import numpy as np
from shapely.geometry import Polygon
import torch


def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres=0.3):
    if 0 == len(S):
        return np.array([])
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]


def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


def parse_polys(img_name, img, output_dir, cls, distances, angle, confidence_threshold=0.5, intersection_threshold=0.3):
    cls = torch.sigmoid(cls).squeeze().data.cpu().numpy()
    distances = distances.squeeze().data.cpu().numpy()
    angle = angle.squeeze().data.cpu().numpy()
    polys = []
    height, width = cls.shape
    IN_OUT_RATIO = 4
    for y in range(height):
        for x in range(width):
            if cls[y, x] < confidence_threshold:
                continue

            poly_height = distances[0, y, x] + distances[2, y, x]
            poly_width = distances[1, y, x] + distances[3, y, x]
            poly_angle = angle[y, x] - np.pi / 4
            x_rot = x * np.cos(-poly_angle) + y * np.sin(-poly_angle)
            y_rot = -x * np.sin(-poly_angle) + y * np.cos(-poly_angle)
            poly_y_center = y_rot * IN_OUT_RATIO + (poly_height / 2 - distances[0, y, x])
            poly_x_center = x_rot * IN_OUT_RATIO - (poly_width / 2 - distances[1, y, x])
            poly = [
                int(((poly_x_center - poly_width / 2) * np.cos(poly_angle) + (poly_y_center - poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center - poly_width / 2) * np.sin(poly_angle) + (poly_y_center - poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center + poly_width / 2) * np.cos(poly_angle) + (poly_y_center - poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center + poly_width / 2) * np.sin(poly_angle) + (poly_y_center - poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center + poly_width / 2) * np.cos(poly_angle) + (poly_y_center + poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center + poly_width / 2) * np.sin(poly_angle) + (poly_y_center + poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center - poly_width / 2) * np.cos(poly_angle) + (poly_y_center + poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center - poly_width / 2) * np.sin(poly_angle) + (poly_y_center + poly_height / 2) * np.cos(poly_angle))),
                cls[y, x]
            ]
            polys.append(poly)

    polys = nms_locality(np.array(polys), intersection_threshold)
    if img is not None:
        for poly in polys:
            pts = np.array(poly[:8]).reshape((4, 2)).astype(np.int32)
            cv2.line(img, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), color=(0, 255, 0))
            cv2.line(img, (pts[1, 0], pts[1, 1]), (pts[2, 0], pts[2, 1]), color=(0, 255, 0))
            cv2.line(img, (pts[2, 0], pts[2, 1]), (pts[3, 0], pts[3, 1]), color=(0, 255, 0))
            cv2.line(img, (pts[3, 0], pts[3, 1]), (pts[0, 0], pts[0, 1]), color=(0, 255, 0))
        cv2.imshow('polys', img)
        if output_dir is not None:
            # save the images to output dir
            filename = output_dir + '/' + img_name + '.jpg'
            cv2.imwrite(filename, img)
        cv2.waitKey()
    return polys


def generate_bboxes(img_name, img, score_maps, geo_maps, angle_maps, confidence_threshold=0.5, intersection_threshold=0.3):
    score_maps = torch.sigmoid(score_maps).squeeze().data.cpu().numpy()
    geo_maps = geo_maps.squeeze().data.cpu().numpy()
    angle_maps = angle_maps.squeeze().data.cpu().numpy()
    polys = []
    height, width = score_maps.shape[1], score_maps.shape[2]
    mask = (score_maps > confidence_threshold).astype(score_maps.dtype)

    IN_OUT_RATIO = 4
    for y in range(height):
        for x in range(width):
            if score_maps[y, x] < confidence_threshold:
                continue

            poly_height = geo_maps[0, y, x] + geo_maps[2, y, x]
            poly_width = geo_maps[1, y, x] + geo_maps[3, y, x]
            poly_angle = angle_maps[y, x] - np.pi / 4
            x_rot = x * np.cos(-poly_angle) + y * np.sin(-poly_angle)
            y_rot = -x * np.sin(-poly_angle) + y * np.cos(-poly_angle)
            poly_y_center = y_rot * IN_OUT_RATIO + (poly_height / 2 - geo_maps[0, y, x])
            poly_x_center = x_rot * IN_OUT_RATIO - (poly_width / 2 - geo_maps[1, y, x])
            poly = [
                int(((poly_x_center - poly_width / 2) * np.cos(poly_angle) + (poly_y_center - poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center - poly_width / 2) * np.sin(poly_angle) + (poly_y_center - poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center + poly_width / 2) * np.cos(poly_angle) + (poly_y_center - poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center + poly_width / 2) * np.sin(poly_angle) + (poly_y_center - poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center + poly_width / 2) * np.cos(poly_angle) + (poly_y_center + poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center + poly_width / 2) * np.sin(poly_angle) + (poly_y_center + poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center - poly_width / 2) * np.cos(poly_angle) + (poly_y_center + poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center - poly_width / 2) * np.sin(poly_angle) + (poly_y_center + poly_height / 2) * np.cos(poly_angle))),
                score_maps[y, x]
            ]
            polys.append(poly)

    polys = nms_locality(np.array(polys), intersection_threshold)
    return polys