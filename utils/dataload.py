from .dataset_info import *
from .args import args
from .pdb import pdb
from .visual import watch_pic_kp_xy

import cv2
import time
import random
import numpy as np
from scipy.interpolate import splprep, splev


def get_annotations_list(dataset, split, ispdb=False):
    annotation_file = None
    annotations = []
    if dataset == 'JD':
        if split == 'train':
            annotation_file = open(dataset_route[dataset] + 'Training_data/landmark.txt')
    elif dataset == 'WFLW':
        if split == 'train':
            annotation_file = open(dataset_route[dataset] + 'WFLW_annotations/list_98pt_rect_attr_train_test/'
                                   'list_98pt_rect_attr_train.txt')
        elif split == 'test':
            annotation_file = open(dataset_route[dataset] + 'WFLW_annotations/list_98pt_rect_attr_train_test/'
                                   'list_98pt_rect_attr_test.txt')
        elif split == 'blur':
            annotation_file = open(dataset_route[dataset] + 'WFLW_annotations/list_98pt_test/'
                                   'list_98pt_test_blur.txt')
        elif split == 'expression':
            annotation_file = open(dataset_route[dataset] + 'WFLW_annotations/list_98pt_test/'
                                   'list_98pt_test_expression.txt')
        elif split == 'illumination':
            annotation_file = open(dataset_route[dataset] + 'WFLW_annotations/list_98pt_test/'
                                   'list_98pt_test_illumination.txt')
        elif split == 'largepose':
            annotation_file = open(dataset_route[dataset] + 'WFLW_annotations/list_98pt_test/'
                                   'list_98pt_test_largepose.txt')
        elif split == 'makeup':
            annotation_file = open(dataset_route[dataset] + 'WFLW_annotations/list_98pt_test/'
                                   'list_98pt_test_makeup.txt')
        elif split == 'occlusion':
            annotation_file = open(dataset_route[dataset] + 'WFLW_annotations/list_98pt_test/'
                                   'list_98pt_test_occlusion.txt')
    elif dataset == '300W':
        if split == 'train':
            annotation_file = open(dataset_route[dataset] + 'landmark_trainset.txt')
        elif split == 'common_subset':
            annotation_file = open(dataset_route[dataset] + 'landmark_common_subset.txt')
        elif split == 'challenge_subset':
            annotation_file = open(dataset_route[dataset] + 'landmark_challenge_subset.txt')
        elif split == 'fullset':
            annotation_file = open(dataset_route[dataset] + 'landmark_fullset.txt')
        elif split == 'testset':
            annotation_file = open(dataset_route[dataset] + 'landmark_testset.txt')

    elif dataset == 'AFLW':
        if split == 'train':
            annotation_file = open(dataset_route[dataset] + 'landmark_trainset.txt')
        elif split == 'fullset':
            annotation_file = open(dataset_route[dataset] + 'landmark_fullset.txt')
        elif split == 'frontalset':
            annotation_file = open(dataset_route[dataset] + 'landmark_frontalset.txt')

    for line in range(dataset_size[dataset][split]):
        annotations.append(annotation_file.readline().rstrip().split())
    annotation_file.close()

    if ispdb:
        annos = []
        allshapes = np.zeros((2 * dataset_kp_num[dataset], len(annotations)))
        for line_index, line in enumerate(annotations):
            coord_x, coord_y = [], []
            if dataset == 'WFLW':
                for kp_index in range(dataset_kp_num[dataset]):
                    coord_x.append(float(line[2 * kp_index]))
                    coord_y.append(float(line[2 * kp_index + 1]))
                position_before = np.float32([[int(line[-11]), int(line[-10])],
                                              [int(line[-11]), int(line[-8])],
                                              [int(line[-9]), int(line[-8])]])
            elif dataset == '300W':
                for kp_index in range(dataset_kp_num[dataset]):
                    coord_x.append(float(line[2 * kp_index]))
                    coord_y.append(float(line[2 * kp_index + 1]))
                position_before = np.float32([[int(line[-5]), int(line[-4])],
                                              [int(line[-5]), int(line[-2])],
                                              [int(line[-3]), int(line[-2])]])
            elif dataset == 'AFLW':
                for kp_index in range(dataset_kp_num[dataset]):
                    coord_x.append(float(line[2 * kp_index]))
                    coord_y.append(float(line[2 * kp_index + 1]))
                position_before = np.float32([[int(line[-5]), int(line[-3])],
                                              [int(line[-5]), int(line[-2])],
                                              [int(line[-4]), int(line[-2])]])
            else:
                raise Exception('This dataset is not supported yet!')
            position_after = np.float32([[0, 0],
                                         [0, args.crop_size - 1],
                                         [args.crop_size - 1, args.crop_size - 1]])
            crop_matrix = cv2.getAffineTransform(position_before, position_after)
            coord_x_after, coord_y_after = cropped_pic_kp(dataset, crop_matrix, coord_x, coord_y)
            for data_index in range(dataset_kp_num[dataset]):
                allshapes[data_index][line_index] = float(coord_x_after[data_index])
                allshapes[data_index + dataset_kp_num[dataset]][line_index] = float(coord_y_after[data_index])
        newidx = pdb(dataset, allshapes, dataset_numbins[dataset])
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


def cropped_pic_kp(dataset, crop_matrix, coord_x, coord_y, flip=0):
    temp_x, temp_y, coord_x_after_crop, coord_y_after_crop = [], [], [], []
    for kp_index in range(dataset_kp_num[dataset]):
        if flip:
            temp_x.append(
                float(args.crop_size) -
                (crop_matrix[0][0] * coord_x[kp_index] +
                 crop_matrix[0][1] * coord_y[kp_index] +
                 crop_matrix[0][2]) -
                1)
        else:
            temp_x.append(
                crop_matrix[0][0] * coord_x[kp_index] +
                crop_matrix[0][1] * coord_y[kp_index] +
                crop_matrix[0][2])
        temp_y.append(
            crop_matrix[1][0] * coord_x[kp_index] +
            crop_matrix[1][1] * coord_y[kp_index] +
            crop_matrix[1][2])
    if flip:
        for kp_index in range(dataset_kp_num[dataset]):
            coord_x_after_crop.append(temp_x[flip_relation[dataset][kp_index][1]])
            coord_y_after_crop.append(temp_y[flip_relation[dataset][kp_index][1]])
        return coord_x_after_crop, coord_y_after_crop
    else:
        return temp_x, temp_y


def get_random_transform_param(split, bbox):
    translation, trans_dir, rotation, scaling, flip = 0, 0, 0, 1., 0
    if split in ['train']:
        random.seed(time.time())
        translate_param = int(args.trans_ratio * abs(bbox[2] - bbox[0]))
        translation = random.randint(-translate_param, translate_param)
        # LU:0 RU:1 LL:2 RL:3
        trans_dir = random.randint(0, 3)
        rotation = random.uniform(-args.rotate_limit, args.rotate_limit)
        scaling = random.uniform(1-args.scale_ratio, 1+args.scale_ratio)
        flip = random.randint(0, 1)
    return translation, trans_dir, rotation, scaling, flip


def get_affine_matrix(crop_size, rotation, scaling):
    center = (crop_size / 2.0, crop_size / 2.0)
    affine_mat = cv2.getRotationMatrix2D(center, rotation, scaling)
    return affine_mat


def pic_normalize(pic):
    pic = np.float32(pic)
    mean, std = cv2.meanStdDev(pic)
    pic_channel = 1 if len(pic.shape) == 2 else 3
    for channel in range(0, pic_channel):
        if std[channel][0] < 1e-6:
            std[channel][0] = 1
    for channel in range(0, pic_channel):
        for w in range(pic.shape[1]):
            for h in range(pic.shape[0]):
                if pic_channel == 1:
                    pic[w][h] = (pic[w][h] - mean[channel][0]) / std[channel][0]
                else:
                    pic[w][h][channel] = (pic[w][h][channel] - mean[channel][0]) / std[channel][0]
    return pic


def get_gt_coords(dataset, coord_x, coord_y, affine_mat):
    out = []
    for kp_index in range(dataset_kp_num[dataset]):
        out.append(
            affine_mat[0][0] * coord_x[kp_index] +
            affine_mat[0][1] * coord_y[kp_index] +
            affine_mat[0][2])
        out.append(
            affine_mat[1][0] * coord_x[kp_index] +
            affine_mat[1][1] * coord_y[kp_index] +
            affine_mat[1][2])
    out = np.array(np.float32(out))
    return out


def get_gt_heatmap(dataset, gt_coords, watch_heatmap=0):
    coord_x, coord_y, gt_heatmap = [], [], []
    for index in range(boundary_num):
        gt_heatmap.append(np.ones((64, 64)))
        gt_heatmap[index].tolist()
    boundary_x = {'chin': [], 'leb': [], 'reb': [], 'bon': [], 'breath': [],
                  'lue': [], 'lle': [], 'rue': [], 'rle': [],
                  'usul': [], 'lsul': [], 'usll': [], 'lsll': []}
    boundary_y = {'chin': [], 'leb': [], 'reb': [], 'bon': [], 'breath': [],
                  'lue': [], 'lle': [], 'rue': [], 'rle': [],
                  'usul': [], 'lsul': [], 'usll': [], 'lsll': []}
    points = {'chin': [], 'leb': [], 'reb': [], 'bon': [], 'breath': [],
              'lue': [], 'lle': [], 'rue': [], 'rle': [],
              'usul': [], 'lsul': [], 'usll': [], 'lsll': []}
    resize_matrix = cv2.getAffineTransform(np.float32([[0, 0], [0, args.crop_size-1],
                                                       [args.crop_size-1, args.crop_size-1]]),
                                           np.float32([[0, 0], [0, heatmap_size-1],
                                                       [heatmap_size-1, heatmap_size-1]]))
    for kp_index in range(dataset_kp_num[dataset]):
        # 此处引入random变量，防止无法成功插值，另外，由于仅用于产生热图，因此也不会影响真实坐标
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
                boundary_index_range[dataset][boundary_index][0],
                boundary_index_range[dataset][boundary_index][1]
        ):
            boundary_x[boundary_keys[boundary_index]].append(coord_x[kp_index])
            boundary_y[boundary_keys[boundary_index]].append(coord_y[kp_index])
        if boundary_keys[boundary_index] in boundary_special.keys() and\
                dataset in boundary_special[boundary_keys[boundary_index]]:
            boundary_x[boundary_keys[boundary_index]].append(
                coord_x[fixed_point[dataset][boundary_keys[boundary_index]]])
            boundary_y[boundary_keys[boundary_index]].append(
                coord_y[fixed_point[dataset][boundary_keys[boundary_index]]])
    for k_index, k in enumerate(boundary_keys):
        if points_per_boundary[dataset][k_index] >= 2.:
            if len(boundary_x[k]) == len(set(boundary_x[k])) or len(boundary_y[k]) == len(set(boundary_y[k])):
                points[k].append(boundary_x[k])
                points[k].append(boundary_y[k])
                res = splprep(points[k], s=0.0, k=1)
                u_new = np.linspace(res[1].min(), res[1].max(), interp_points_num[k])
                boundary_x[k], boundary_y[k] = splev(u_new, res[0], der=0)
    for index, k in enumerate(boundary_keys):
        for i in range(len(boundary_x[k])):
            cv2.line(gt_heatmap[index], (int(boundary_x[k][i]), int(boundary_y[k][i])),
                     (int(boundary_x[k][i]), int(boundary_y[k][i])), 0)
        gt_heatmap[index] = np.uint8(gt_heatmap[index])
        gt_heatmap[index] = cv2.distanceTransform(gt_heatmap[index], cv2.DIST_L2, 5)
        gt_heatmap[index] = np.float32(gt_heatmap[index])
        for h in range(heatmap_size):
            for w in range(heatmap_size):
                if gt_heatmap[index][h][w] < 3.0:
                    gt_heatmap[index][h][w] = np.exp(-gt_heatmap[index][h][w] * gt_heatmap[index][h][w] / 2)
                else:
                    gt_heatmap[index][h][w] = 0
    gt_heatmap = np.float32(np.array(gt_heatmap))
    if watch_heatmap != 0:
        heatmap_sum = gt_heatmap[0]
        for index in range(boundary_num - 1):
            heatmap_sum += gt_heatmap[index + 1]
        cv2.imshow('heatmap_sum', heatmap_sum)
        cv2.moveWindow('heatmap_sum', 0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return gt_heatmap


def getitem_from(dataset, split, annotation, eval_flag=0):
    flip, rotation, scaling = 0, 0., 1.
    pic, pic_gray, coord_x, coord_y, coord_xy, position_before, position_after, bbox = \
        None, None, [], [], [], None, None, np.zeros(4)
    if dataset == 'JD':
        pic = cv2.imread(dataset_route[dataset]+'Training_data/' +
                         annotation[0].split('_')[0]+'/picture/'+annotation[0])
        pic_gray = convert_img_to_gray(pic) if not args.RGB else pic
        for kp_index in range(dataset_kp_num[dataset]):
            coord_x.append(float(annotation[2 * kp_index + 1]))
            coord_y.append(float(annotation[2 * kp_index + 2]))
            coord_xy.append(float(annotation[2 * kp_index + 1]))
            coord_xy.append(float(annotation[2 * kp_index + 2]))

        # 该段根据bounding box裁剪出人脸并resize为256*256大小
        bounding_box_file = open(dataset_route[dataset]+'training_dataset_face_detection_bounding_box/' +
                                 annotation[0]+'.rect')
        bounding_box = bounding_box_file.readline().rstrip().split()
        bounding_box_file.close()
        for i in range(4):
            bbox[i] = int(bounding_box[i])
        
        translation, trans_dir, rotation, scaling, flip = get_random_transform_param(split, bbox)

        position_before = np.float32([[int(bounding_box[0]) + pow(-1, trans_dir+1)*translation,
                                       int(bounding_box[1]) + pow(-1, trans_dir//2+1)*translation],
                                      [int(bounding_box[0]) + pow(-1, trans_dir+1)*translation,
                                       int(bounding_box[3]) + pow(-1, trans_dir//2+1)*translation],
                                      [int(bounding_box[2]) + pow(-1, trans_dir+1)*translation,
                                       int(bounding_box[3]) + pow(-1, trans_dir//2+1)*translation]])
    elif dataset == 'WFLW':
        pic = cv2.imread(dataset_route[dataset] + 'WFLW_images/' + annotation[-1])
        pic_gray = convert_img_to_gray(pic) if not args.RGB else pic
        for kp_index in range(dataset_kp_num[dataset]):
            coord_x.append(float(annotation[2 * kp_index]))
            coord_y.append(float(annotation[2 * kp_index + 1]))
            coord_xy.append(float(annotation[2 * kp_index]))
            coord_xy.append(float(annotation[2 * kp_index + 1]))
        for i in range(4):
            bbox[i] = int(annotation[-11+i])

        translation, trans_dir, rotation, scaling, flip = get_random_transform_param(split, bbox)

        # 该段根据bounding box裁剪出人脸并resize为256*256大小
        position_before = np.float32([[int(annotation[-11]) + pow(-1, trans_dir+1)*translation,
                                       int(annotation[-10]) + pow(-1, trans_dir//2+1)*translation],
                                      [int(annotation[-11]) + pow(-1, trans_dir+1)*translation,
                                       int(annotation[-8]) + pow(-1, trans_dir//2+1)*translation],
                                      [int(annotation[-9]) + pow(-1, trans_dir+1)*translation,
                                       int(annotation[-8]) + pow(-1, trans_dir//2+1)*translation]])
    elif dataset == '300W':
        pic = cv2.imread(dataset_route[dataset] + annotation[-1])
        pic_gray = convert_img_to_gray(pic) if not args.RGB else pic
        for kp_index in range(dataset_kp_num[dataset]):
            coord_x.append(float(annotation[2 * kp_index]))
            coord_y.append(float(annotation[2 * kp_index + 1]))
            coord_xy.append(float(annotation[2 * kp_index]))
            coord_xy.append(float(annotation[2 * kp_index + 1]))
        for i in range(4):
            bbox[i] = int(annotation[-5+i])

        translation, trans_dir, rotation, scaling, flip = get_random_transform_param(split, bbox)

        # 该段根据bounding box裁剪出人脸并resize为256*256大小
        position_before = np.float32([[int(annotation[-5]) + pow(-1, trans_dir + 1) * translation,
                                       int(annotation[-4]) + pow(-1, trans_dir // 2 + 1) * translation],
                                      [int(annotation[-5]) + pow(-1, trans_dir + 1) * translation,
                                       int(annotation[-2]) + pow(-1, trans_dir // 2 + 1) * translation],
                                      [int(annotation[-3]) + pow(-1, trans_dir + 1) * translation,
                                       int(annotation[-2]) + pow(-1, trans_dir // 2 + 1) * translation]])
    elif dataset == 'AFLW':
        pic = cv2.imread(dataset_route[dataset] + annotation[-1])
        pic_gray = convert_img_to_gray(pic) if not args.RGB else pic
        for kp_index in range(dataset_kp_num[dataset]):
            coord_x.append(float(annotation[2 * kp_index]))
            coord_y.append(float(annotation[2 * kp_index + 1]))
            coord_xy.append(float(annotation[2 * kp_index]))
            coord_xy.append(float(annotation[2 * kp_index + 1]))
        bbox[0] = int(annotation[-5])
        bbox[1] = int(annotation[-3])
        bbox[2] = int(annotation[-4])
        bbox[3] = int(annotation[-2])

        translation, trans_dir, rotation, scaling, flip = get_random_transform_param(split, bbox)

        # 该段根据bounding box裁剪出人脸并resize为256*256大小
        position_before = np.float32([[int(annotation[-5]) + pow(-1, trans_dir + 1) * translation,
                                       int(annotation[-3]) + pow(-1, trans_dir // 2 + 1) * translation],
                                      [int(annotation[-5]) + pow(-1, trans_dir + 1) * translation,
                                       int(annotation[-2]) + pow(-1, trans_dir // 2 + 1) * translation],
                                      [int(annotation[-4]) + pow(-1, trans_dir + 1) * translation,
                                       int(annotation[-2]) + pow(-1, trans_dir // 2 + 1) * translation]])
    position_after = np.float32([[0, 0],
                                 [0, args.crop_size - 1],
                                 [args.crop_size - 1, args.crop_size - 1]])
    crop_matrix = cv2.getAffineTransform(position_before, position_after)
    pic_crop = cv2.warpAffine(pic_gray, crop_matrix, (args.crop_size, args.crop_size))
    if flip == 1:
        pic_crop = cv2.flip(pic_crop, 1)
    affine_mat = get_affine_matrix(args.crop_size, rotation, scaling)
    pic_affine = cv2.warpAffine(pic_crop, affine_mat, (args.crop_size, args.crop_size))
    pic_affine = pic_normalize(pic_affine)

    # 该段将原始关键点坐标转化为裁剪后的图像上的坐标
    coord_x_after_crop, coord_y_after_crop = cropped_pic_kp(dataset, crop_matrix, coord_x, coord_y, flip=flip)

    # 该段将原始关键点坐标转化为仿射变换后的图像上的坐标
    gt_keypoints = get_gt_coords(dataset, coord_x_after_crop, coord_y_after_crop, affine_mat)

    # 该段根据生成的新的坐标点，将坐标转换为64*64大小图片上的坐标并绘制boundary热图
    # 若在函数最后设置watch_heatmap=1，则显示热图(需要使用pycharm的science mode)
    gt_heatmap = get_gt_heatmap(dataset, gt_keypoints, watch_heatmap=0)

    # watch_pic_kp_xy(dataset, pic_crop, coord_x_after_crop, coord_y_after_crop)

    if eval_flag == 0:
        return pic_affine, gt_keypoints, gt_heatmap
    else:
        return pic_affine, np.array(np.float32(coord_xy)), gt_heatmap, bbox, str(annotation[-1])


if __name__ == '__main__':
    from visual import watch_pic_kp
    anno = get_annotations_list(args.dataset, args.split, ispdb=args.PDB)
    for iii, jjj in enumerate(anno):
        pic_affine_t, gt_keypoints_t, gt_heatmap_t = getitem_from(args.dataset, args.split, jjj)
        watch_pic_kp(args.dataset, pic_affine_t, gt_keypoints_t)
        if iii > 10:
            break
