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


def get_devices_list(arg):
    devices_list = [torch.device('cpu')]
    if arg.cuda:
        devices_list = []
        for dev in arg.gpu_id.split(','):
            devices_list.append(torch.device('cuda:'+dev))
        cudnn.benchmark = True
        cudnn.enabled = True
    return devices_list


def load_weights(net, pth_file, device):
    state_dict = torch.load(pth_file, map_location=device)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net


def create_model(arg, devices_list):
    from models import Estimator, Discrim, Regressor

    estimator = Estimator(stacks=arg.hour_stack, msg_pass=arg.msg_pass)
    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2*dataset_kp_num[arg.dataset])
    discrim = Discrim() if arg.GAN else None
    
    if arg.resume_epoch > 0:
        estimator = load_weights(estimator, arg.resume_folder + 'estimator_' + str(arg.resume_epoch) + '.pth',
                                 devices_list[0])
        if not arg.regress_only:
            regressor = load_weights(regressor, arg.resume_folder + arg.dataset+'_regressor_' +
                                     str(arg.resume_epoch) + '.pth', devices_list[0])
            discrim = load_weights(discrim, arg.resume_folder + 'discrim_' + str(arg.resume_epoch) + '.pth',
                                   devices_list[0]) if arg.GAN else None

    if arg.cuda:
        estimator = estimator.cuda(device=devices_list[0])
        regressor = regressor.cuda(device=devices_list[0])
        discrim = discrim.cuda(device=devices_list[0]) if arg.GAN else None

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


def inverse_affine(arg, pred_coords, bbox):
    import copy
    pred_coords = copy.deepcopy(pred_coords)
    pred_coords = pred_coords.squeeze().numpy()
    bbox = bbox.squeeze().numpy()
    for i in range(dataset_kp_num[arg.dataset]):
        pred_coords[2 * i] = bbox[0] + pred_coords[2 * i]/(arg.crop_size-1)*(bbox[2] - bbox[0])
        pred_coords[2 * i + 1] = bbox[1] + pred_coords[2 * i + 1]/(arg.crop_size-1)*(bbox[3] - bbox[1])
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
