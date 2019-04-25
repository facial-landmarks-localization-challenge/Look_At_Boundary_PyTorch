from .dataset_info import *
from .args import args
from .pdb import pdb
from .visual import *

import cv2
import time
import random
import numpy as np
from scipy.interpolate import splprep, splev


def get_annotations_list(dataset, split, ispdb=False):
    annotations = []
    annotation_file = open(dataset_route[dataset] + dataset + '_' + split + '_annos.txt')

    for line in range(dataset_size[dataset][split]):
        annotations.append(annotation_file.readline().rstrip().split())
    annotation_file.close()

    if ispdb:
        annos = []
        allshapes = np.zeros((2 * kp_num[dataset], len(annotations)))
        for line_index, line in enumerate(annotations):
            coord_x = np.array(list(map(float, line[:2*kp_num[dataset]:2])))
            coord_y = np.array(list(map(float, line[1:2*kp_num[dataset]:2])))
            position_before = np.float32([[int(line[-7]), int(line[-6])],
                                          [int(line[-7]), int(line[-4])],
                                          [int(line[-5]), int(line[-4])]])
            position_after = np.float32([[0, 0],
                                         [0, args.crop_size - 1],
                                         [args.crop_size - 1, args.crop_size - 1]])
            crop_matrix = cv2.getAffineTransform(position_before, position_after)
            coord_x_after_crop = crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2]
            coord_y_after_crop = crop_matrix[1][0] * coord_x + crop_matrix[1][1] * coord_y + crop_matrix[1][2]
            allshapes[0:kp_num[dataset], line_index] = list(coord_x_after_crop)
            allshapes[kp_num[dataset]:2*kp_num[dataset], line_index] = list(coord_y_after_crop)
        newidx = pdb(dataset, allshapes, dataset_pdb_numbins[dataset])
        for id_index in newidx:
            annos.append(annotations[int(id_index)])
        return annos

    return annotations


def convert_img_to_gray(img):
    if img.shape[2] == 1:
        return img
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return gray
    elif img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    else:
        raise Exception("img shape wrong!\n")


def get_random_transform_param(split, bbox):
    translation, trans_dir, rotation, scaling, flip, gaussian_blur = 0, 0, 0, 1., 0, 0
    if split in ['train']:
        random.seed(time.time())
        translate_param = int(args.trans_ratio * abs(bbox[2] - bbox[0]))
        translation = random.randint(-translate_param, translate_param)
        trans_dir = random.randint(0, 3)  # LU:0 RU:1 LL:2 RL:3
        rotation = random.uniform(-args.rotate_limit, args.rotate_limit)
        scaling = random.uniform(1-args.scale_ratio, 1+args.scale_ratio)
        flip = random.randint(0, 1)
        gaussian_blur = random.randint(0, 1)
    return translation, trans_dir, rotation, scaling, flip, gaussian_blur


def further_transform(pic, bbox, flip, gaussian_blur):
    if flip == 1:
        pic = cv2.flip(pic, 1)
    if abs(bbox[2] - bbox[0]) < 120 or gaussian_blur == 0:
        return pic
    else:
        return cv2.GaussianBlur(pic, (5, 5), 1)


def get_affine_matrix(crop_size, rotation, scaling):
    center = (crop_size / 2.0, crop_size / 2.0)
    return cv2.getRotationMatrix2D(center, rotation, scaling)


def pic_normalize(pic):  # for accelerate, now support gray pic only
    pic = np.float32(pic)
    mean, std = cv2.meanStdDev(pic)
    pic_channel = 1 if len(pic.shape) == 2 else 3
    for channel in range(0, pic_channel):
        if std[channel][0] < 1e-6:
            std[channel][0] = 1
    pic = (pic - mean) / std
    return np.float32(pic)


def get_cropped_coords(dataset, crop_matrix, coord_x, coord_y, flip=0):
    coord_x, coord_y = np.array(coord_x), np.array(coord_y)
    temp_x = crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2] if flip == 0 else \
        float(args.crop_size) - (crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2]) - 1
    temp_y = crop_matrix[1][0] * coord_x + crop_matrix[1][1] * coord_y + crop_matrix[1][2]
    if flip:
        temp_x = temp_x[np.array(flip_relation[dataset])[:, 1]]
        temp_y = temp_y[np.array(flip_relation[dataset])[:, 1]]
    return temp_x, temp_y


def get_gt_coords(dataset, affine_matrix, coord_x, coord_y):
    out = np.zeros(2*kp_num[dataset])
    out[:2*kp_num[dataset]:2] = affine_matrix[0][0] * coord_x + affine_matrix[0][1] * coord_y + affine_matrix[0][2]
    out[1:2*kp_num[dataset]:2] = affine_matrix[1][0] * coord_x + affine_matrix[1][1] * coord_y + affine_matrix[1][2]
    return np.array(np.float32(out))


def get_gt_heatmap(dataset, split, gt_coords):
    dataset = '300W' if dataset == 'COFW' and split == 'test68' else dataset
    coord_x, coord_y, gt_heatmap = [], [], []
    for index in range(boundary_num):
        gt_heatmap.append(np.ones((64, 64)))
        gt_heatmap[index].tolist()
    boundary_x = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
                  'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}
    boundary_y = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
                  'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}
    points = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
              'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}
    resize_matrix = cv2.getAffineTransform(np.float32([[0, 0], [0, args.crop_size-1],
                                                       [args.crop_size-1, args.crop_size-1]]),
                                           np.float32([[0, 0], [0, heatmap_size-1],
                                                       [heatmap_size-1, heatmap_size-1]]))
    for kp_index in range(kp_num[dataset]):
        coord_x.append(
            resize_matrix[0][0] * gt_coords[2 * kp_index] +
            resize_matrix[0][1] * gt_coords[2 * kp_index + 1] +
            resize_matrix[0][2] + random.uniform(-0.2, 0.2)
        )
        coord_y.append(
            resize_matrix[1][0] * gt_coords[2 * kp_index] +
            resize_matrix[1][1] * gt_coords[2 * kp_index + 1] +
            resize_matrix[1][2] + random.uniform(-0.2, 0.2)
        )
    for boundary_index in range(boundary_num):
        for kp_index in range(
                point_range[dataset][boundary_index][0],
                point_range[dataset][boundary_index][1]
        ):
            boundary_x[boundary_keys[boundary_index]].append(coord_x[kp_index])
            boundary_y[boundary_keys[boundary_index]].append(coord_y[kp_index])
        if boundary_keys[boundary_index] in boundary_special.keys() and\
                dataset in boundary_special[boundary_keys[boundary_index]]:
            boundary_x[boundary_keys[boundary_index]].append(
                coord_x[duplicate_point[dataset][boundary_keys[boundary_index]]])
            boundary_y[boundary_keys[boundary_index]].append(
                coord_y[duplicate_point[dataset][boundary_keys[boundary_index]]])
    for k_index, k in enumerate(boundary_keys):
        if point_num_per_boundary[dataset][k_index] >= 2.:
            if len(boundary_x[k]) == len(set(boundary_x[k])) or len(boundary_y[k]) == len(set(boundary_y[k])):
                points[k].append(boundary_x[k])
                points[k].append(boundary_y[k])
                res = splprep(points[k], s=0.0, k=1)
                u_new = np.linspace(res[1].min(), res[1].max(), interp_points_num[k])
                boundary_x[k], boundary_y[k] = splev(u_new, res[0], der=0)
    for index, k in enumerate(boundary_keys):
        if point_num_per_boundary[dataset][index] >= 2.:
            for i in range(len(boundary_x[k]) - 1):
                cv2.line(gt_heatmap[index], (int(boundary_x[k][i]), int(boundary_y[k][i])),
                         (int(boundary_x[k][i+1]), int(boundary_y[k][i+1])), 0)
        else:
            cv2.circle(gt_heatmap[index], (int(boundary_x[k][0]), int(boundary_y[k][0])), 2, 0, -1)
        gt_heatmap[index] = np.uint8(gt_heatmap[index])
        gt_heatmap[index] = cv2.distanceTransform(gt_heatmap[index], cv2.DIST_L2, 5)
        gt_heatmap[index] = np.float32(np.array(gt_heatmap[index]))
        gt_heatmap[index] = gt_heatmap[index].reshape(64*64)
        (gt_heatmap[index])[(gt_heatmap[index]) < 3. * args.sigma] = \
            np.exp(-(gt_heatmap[index])[(gt_heatmap[index]) < 3 * args.sigma] *
                   (gt_heatmap[index])[(gt_heatmap[index]) < 3 * args.sigma] / 2. * args.sigma * args.sigma)
        (gt_heatmap[index])[(gt_heatmap[index]) >= 3. * args.sigma] = 0.
        gt_heatmap[index] = gt_heatmap[index].reshape([64, 64])
    return np.array(gt_heatmap)


def get_item_from(dataset, split, annotation):
    pic = cv2.imread(dataset_route[dataset]+annotation[-1])
    pic = convert_img_to_gray(pic) if not args.RGB else pic
    dataset = '300W' if dataset == 'COFW' and split == 'test68' else dataset
    coord_x = list(map(float, annotation[:2*kp_num[dataset]:2]))
    coord_y = list(map(float, annotation[1:2*kp_num[dataset]:2]))
    coord_xy = np.array(np.float32(list(map(float, annotation[:2*kp_num[dataset]]))))
    bbox = np.array(list(map(int, annotation[-7:-3])))

    translation, trans_dir, rotation, scaling, flip, gaussian_blur = get_random_transform_param(split, bbox)

    position_before = np.float32([[int(bbox[0]) + pow(-1, trans_dir+1)*translation,
                                   int(bbox[1]) + pow(-1, trans_dir//2+1)*translation],
                                  [int(bbox[0]) + pow(-1, trans_dir+1)*translation,
                                   int(bbox[3]) + pow(-1, trans_dir//2+1)*translation],
                                  [int(bbox[2]) + pow(-1, trans_dir+1)*translation,
                                   int(bbox[3]) + pow(-1, trans_dir//2+1)*translation]])
    position_after = np.float32([[0, 0],
                                 [0, args.crop_size - 1],
                                 [args.crop_size - 1, args.crop_size - 1]])
    crop_matrix = cv2.getAffineTransform(position_before, position_after)
    pic_crop = cv2.warpAffine(pic, crop_matrix, (args.crop_size, args.crop_size))
    pic_crop = further_transform(pic_crop, bbox, flip, gaussian_blur) if args.split in ['train'] else pic_crop
    affine_matrix = get_affine_matrix(args.crop_size, rotation, scaling)
    pic_affine = cv2.warpAffine(pic_crop, affine_matrix, (args.crop_size, args.crop_size))
    pic_affine = pic_normalize(pic_affine) if not args.RGB else pic_affine

    coord_x_cropped, coord_y_cropped = get_cropped_coords(dataset, crop_matrix, coord_x, coord_y, flip=flip)
    gt_coords_xy = get_gt_coords(dataset, affine_matrix, coord_x_cropped, coord_y_cropped)

    gt_heatmap = get_gt_heatmap(dataset, split, gt_coords_xy)

    return pic_affine, gt_coords_xy, gt_heatmap, coord_xy, bbox, annotation[-1]
