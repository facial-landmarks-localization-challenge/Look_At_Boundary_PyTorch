import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import numpy as np
from sklearn.metrics import auc
from utils import *


def calc_d_fake(dataset, pred_coords, gt_coords, bcsize, bcsize_set):
    error_regressor = (pred_coords - gt_coords) ** 2
    dist_regressor = torch.zeros(bcsize, dataset_kp_num[dataset])
    for batch in range(bcsize):
        for point in range(dataset_kp_num[dataset]):
            dist_regressor[batch][point] = \
                (error_regressor[batch][2 * point] + error_regressor[batch][2 * point + 1]) < 2.25  # 1.5*1.5
    dfake = torch.zeros(bcsize_set, boundary_num)
    for batch_index in range(bcsize):
        for boundary_index in range(boundary_num):
            for kp_index in range(
                    boundary_index_range[dataset][boundary_index][0],
                    boundary_index_range[dataset][boundary_index][1]
            ):
                if dist_regressor[batch_index][kp_index] == 1:
                    dfake[batch_index][boundary_index] += 1
            if boundary_keys[boundary_index] in boundary_special.keys() and \
                    dataset in boundary_special[boundary_keys[boundary_index]] and \
                    dist_regressor[batch_index][fixed_point[dataset][boundary_keys[boundary_index]]] == 1:
                dfake[batch_index][boundary_index] += 1
        for boundary_index in range(boundary_num):
            if dfake[batch_index][boundary_index] / points_per_boundary[dataset][boundary_index] < 0.8:
                dfake[batch_index][boundary_index] = 0.
            else:
                dfake[batch_index][boundary_index] = 1.
    if bcsize < bcsize_set:
        for batch_index in range(bcsize, bcsize_set):
            dfake[batch_index] = dfake[batch_index - bcsize]
    return dfake


def get_devices_list(args):
    devices_list = [torch.device('cpu')]
    if args.cuda:
        devices_list = []
        for dev in args.gpu_id.split(','):
            devices_list.append(torch.device('cuda:'+dev))
        cudnn.benchmark = True
        cudnn.enabled = True
    return devices_list


def load_weights(net, pth_file):
    state_dict = torch.load(pth_file)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net


def create_model(args, devices_list):
    from models import Estimator, Discrim, Regressor

    estimator = Estimator(stacks=args.hour_stack, msg_pass=args.msg_pass)
    regressor = Regressor(fuse_stages=args.fuse_stage, output=2*dataset_kp_num[args.dataset])
    discrim = Discrim() if args.GAN else None
    
    if args.resume_epoch > 0:
        estimator = load_weights(estimator, args.resume_folder + 'estimator_' + str(args.resume_epoch) + '.pth')
        if not args.regress_only:
            regressor = load_weights(regressor, args.resume_folder + args.dataset+'_regressor_' + str(args.resume_epoch) + '.pth')
            discrim = load_weights(discrim, args.resume_folder + 'discrim_' + str(args.resume_epoch) + '.pth') if args.GAN else None

    if args.cuda:
        estimator = estimator.cuda(device=devices_list[0])
        regressor = regressor.cuda(device=devices_list[0])
        discrim = discrim.cuda(device=devices_list[0]) if args.GAN else None

    return estimator, regressor, discrim


def calc_normalize_factor(dataset, gt_keypoints, normalize_way='inter_pupil'):
    if normalize_way == 'inter_ocular':
        error_normalize_factor = np.sqrt(
            (gt_keypoints[0][lo_eye_corner_index_x[dataset]] - gt_keypoints[0][ro_eye_corner_index_x[dataset]]) *
            (gt_keypoints[0][lo_eye_corner_index_x[dataset]] - gt_keypoints[0][ro_eye_corner_index_x[dataset]]) +
            (gt_keypoints[0][lo_eye_corner_index_y[dataset]] - gt_keypoints[0][ro_eye_corner_index_y[dataset]]) *
            (gt_keypoints[0][lo_eye_corner_index_y[dataset]] - gt_keypoints[0][ro_eye_corner_index_y[dataset]]))
        return error_normalize_factor
    elif normalize_way == 'inter_pupil':
        if l_eye_center_index_x[dataset].__class__ != list:
            error_normalize_factor = np.sqrt(
                (gt_keypoints[0][l_eye_center_index_x[dataset]] - gt_keypoints[0][r_eye_center_index_x[dataset]]) *
                (gt_keypoints[0][l_eye_center_index_x[dataset]] - gt_keypoints[0][r_eye_center_index_x[dataset]]) +
                (gt_keypoints[0][l_eye_center_index_y[dataset]] - gt_keypoints[0][r_eye_center_index_y[dataset]]) *
                (gt_keypoints[0][l_eye_center_index_y[dataset]] - gt_keypoints[0][r_eye_center_index_y[dataset]]))
            return error_normalize_factor
        else:
            length = len(l_eye_center_index_x[dataset])
            l_eye_x_avg, l_eye_y_avg, r_eye_x_avg, r_eye_y_avg = 0., 0., 0., 0.
            for i in range(length):
                l_eye_x_avg += gt_keypoints[0][l_eye_center_index_x[dataset][i]]
                l_eye_y_avg += gt_keypoints[0][l_eye_center_index_y[dataset][i]]
                r_eye_x_avg += gt_keypoints[0][r_eye_center_index_x[dataset][i]]
                r_eye_y_avg += gt_keypoints[0][r_eye_center_index_y[dataset][i]]
            l_eye_x_avg /= length
            l_eye_y_avg /= length
            r_eye_x_avg /= length
            r_eye_y_avg /= length
            error_normalize_factor = np.sqrt((l_eye_x_avg - r_eye_x_avg) * (l_eye_x_avg - r_eye_x_avg) +
                                             (l_eye_y_avg - r_eye_y_avg) * (l_eye_y_avg - r_eye_y_avg))
            return error_normalize_factor


def inverse_affine(args, pred_coords, bbox):
    import copy
    pred_coords = copy.deepcopy(pred_coords)
    pred_coords = pred_coords.squeeze().numpy()
    bbox = bbox.squeeze().numpy()
    for i in range(dataset_kp_num[args.dataset]):
        pred_coords[2 * i] = bbox[0] + pred_coords[2 * i]/(args.crop_size-1)*(bbox[2] - bbox[0])
        pred_coords[2 * i + 1] = bbox[1] + pred_coords[2 * i + 1]/(args.crop_size-1)*(bbox[3] - bbox[1])
    return pred_coords


def calc_error_rate_i(dataset, pred_coords, gt_keypoints, error_normalize_factor):
    temp, error = (pred_coords - gt_keypoints)**2, 0.
    for i in range(dataset_kp_num[dataset]):
        error += np.sqrt(temp[2*i] + temp[2*i+1])
    return error/dataset_kp_num[dataset]/error_normalize_factor


def calc_auc(dataset, split, error_rate, max_threshold):
    error_rate = np.array(error_rate)
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_rate < threshold[i]) * 1.0 / dataset_size[dataset][split]
    return auc(threshold, accuracys) / max_threshold
