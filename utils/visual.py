from .dataset_info import *

import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def show_img(pic, name='pic', x=0, y=0, wait=0):
    cv2.imshow(name, pic)
    cv2.moveWindow(name, x, y)
    cv2.waitKey(wait)
    cv2.destroyAllWindows()


def watch_gray_heatmap(gt_heatmap):
    heatmap_sum = gt_heatmap[0]
    for index in range(boundary_num - 1):
        heatmap_sum += gt_heatmap[index + 1]
    show_img(heatmap_sum, 'heatmap_sum')


def watch_pic_kp(dataset, split, pic, kp):
    dataset = '300W' if dataset == 'COFW' and split == 'test68' else dataset
    for kp_index in range(kp_num[dataset]):
        cv2.circle(
            pic,
            (int(kp[2*kp_index]), int(kp[2*kp_index+1])),
            1,
            (0, 0, 255)
        )
    show_img(pic)


def watch_pic_kp_xy(dataset, split, pic, coord_x, coord_y):
    dataset = '300W' if dataset == 'COFW' and split == 'test68' else dataset
    for kp_index in range(kp_num[dataset]):
        cv2.circle(
            pic,
            (int(coord_x[kp_index]), int(coord_y[kp_index])),
            1,
            (0, 0, 255)
        )
    show_img(pic)


def eval_heatmap(arg, heatmaps, img_name, bbox, save_img=False):
    heatmaps = F.interpolate(heatmaps, scale_factor=4, mode='bilinear', align_corners=True)
    heatmaps = heatmaps.squeeze(0).detach().cpu().numpy()
    heatmaps_sum = heatmaps[0]
    for heatmaps_index in range(boundary_num-1):
        heatmaps_sum += heatmaps[heatmaps_index+1]
    plt.axis('off')
    plt.imshow(heatmaps_sum, interpolation='nearest', vmax=1., vmin=0.)
    if save_img:
        import os
        if not os.path.exists('./imgs'):
            os.mkdir('./imgs')
        fig = plt.gcf()
        fig.set_size_inches(2.56 / 3, 2.56 / 3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        name = (img_name[0]).split('/')[-1]
        fig.savefig('./imgs/'+name.split('.')[0]+'_hm.png', format='png', transparent=True, dpi=300, pad_inches=0)

        pic = cv2.imread(dataset_route[arg.dataset] + img_name[0])
        position_before = np.float32([
            [int(bbox[0]), int(bbox[1])],
            [int(bbox[0]), int(bbox[3])],
            [int(bbox[2]), int(bbox[3])]
        ])
        position_after = np.float32([
            [0, 0],
            [0, arg.crop_size - 1],
            [arg.crop_size - 1, arg.crop_size - 1]
        ])
        crop_matrix = cv2.getAffineTransform(position_before, position_after)
        pic = cv2.warpAffine(pic, crop_matrix, (arg.crop_size, arg.crop_size))
        cv2.imwrite('./imgs/' + name.split('.')[0] + '_init.png', pic)
        hm = cv2.imread('./imgs/'+name.split('.')[0]+'_hm.png')
        syn = cv2.addWeighted(pic, 0.4, hm, 0.6, 0)
        cv2.imwrite('./imgs/'+name.split('.')[0]+'_syn.png', syn)
    plt.show()


def eval_pred_points(arg, pred_coords, img_name, bbox, save_img=False):
    pred_coords = pred_coords.squeeze().numpy()
    pic = cv2.imread(dataset_route[arg.dataset] + img_name[0])
    position_before = np.float32([
        [int(bbox[0]), int(bbox[1])],
        [int(bbox[0]), int(bbox[3])],
        [int(bbox[2]), int(bbox[3])]
    ])
    position_after = np.float32([
        [0, 0],
        [0, arg.crop_size - 1],
        [arg.crop_size - 1, arg.crop_size - 1]
    ])
    crop_matrix = cv2.getAffineTransform(position_before, position_after)
    pic = cv2.warpAffine(pic, crop_matrix, (arg.crop_size, arg.crop_size))

    dataset = '300W' if arg.dataset == 'COFW' and arg.split == 'test68' else arg.dataset
    for coord_index in range(kp_num[dataset]):
        cv2.circle(pic, (int(pred_coords[2 * coord_index]), int(pred_coords[2 * coord_index + 1])), 2, (0, 0, 255))
    if save_img:
        import os
        if not os.path.exists('./imgs'):
            os.mkdir('./imgs')
        name = (img_name[0]).split('/')[-1]
        cv2.imwrite('./imgs/'+name.split('.')[0]+'_lmk.png', pic)
    show_img(pic)


def eval_gt_pred_points(arg, gt_coords, pred_coords, img_name, bbox, save_img=False):
    assert arg.dataset in ['300W']
    pred_coords = pred_coords.squeeze().numpy()
    gt_coords = gt_coords.squeeze().numpy()
    pic = cv2.imread(dataset_route[arg.dataset] + img_name[0])
    position_before = np.float32([
        [int(bbox[0]), int(bbox[1])],
        [int(bbox[0]), int(bbox[3])],
        [int(bbox[2]), int(bbox[3])]
    ])
    position_after = np.float32([
        [0, 0],
        [0, arg.crop_size - 1],
        [arg.crop_size - 1, arg.crop_size - 1]
    ])
    crop_matrix = cv2.getAffineTransform(position_before, position_after)
    pic = cv2.warpAffine(pic, crop_matrix, (arg.crop_size, arg.crop_size))

    dataset = '300W' if arg.dataset == 'COFW' and arg.split == 'test68' else arg.dataset
    for coord_index in range(kp_num[dataset]):
        cv2.circle(pic, (int(pred_coords[2 * coord_index]), int(pred_coords[2 * coord_index + 1])), 2, (0, 0, 255))
        cv2.circle(pic, (int(gt_coords[2 * coord_index]), int(gt_coords[2 * coord_index + 1])), 2, (0, 255, 0))
    if save_img:
        import os
        if not os.path.exists('./imgs'):
            os.mkdir('./imgs')
        name = (img_name[0]).split('/')[-1]
        cv2.imwrite('./imgs/'+name.split('.')[0]+'_lmk.png', pic)
    show_img(pic)
