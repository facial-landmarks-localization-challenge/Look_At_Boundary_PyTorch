import tqdm
import time
import numpy as np
from dataset import GeneralDataset
from models import *
from utils import *


def evaluate(arg):
    devices = torch.device('cuda:'+arg.gpu_id)
    error_rate = []
    failure_count = 0
    max_threshold = arg.max_threshold

    testset = GeneralDataset(dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    print('*****  Normal Evaluating  *****')
    print('Evaluating parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Epoch of the model: ' + str(arg.eval_epoch) + '\n' +
          '# Normalize way:      ' + arg.norm_way + '\n' +
          '# Max threshold:      ' + str(arg.max_threshold) + '\n')
    
    print('Loading network ...')
    estimator = Estimator(stacks=arg.hour_stack, msg_pass=arg.msg_pass)
    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2*kp_num[arg.dataset])
    estimator = load_weights(estimator, arg.save_folder+'estimator_'+str(arg.eval_epoch)+'.pth', devices)
    regressor = load_weights(regressor, arg.save_folder+arg.dataset+'_regressor_'+str(arg.eval_epoch)+'.pth', devices)
    if arg.cuda:
        estimator = estimator.cuda(device=devices)
        regressor = regressor.cuda(device=devices)
    estimator.eval()
    regressor.eval()
    print('Loading network done!\nStart testing ...')
    
    time_records = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            start = time.time()

            input_images, gt_coords_xy, gt_heatmap, coords_xy, bbox, img_name = data
            gt_coords_xy = gt_coords_xy.squeeze().numpy()
            bbox = bbox.squeeze().numpy()
            error_normalize_factor = calc_normalize_factor(arg.dataset, coords_xy.numpy(), arg.norm_way) \
                if arg.norm_way in ['inter_pupil', 'inter_ocular'] else (bbox[2] - bbox[0])
            input_images = input_images.unsqueeze(1)
            input_images = input_images.cuda(device=devices)

            pred_heatmaps = estimator(input_images)
            pred_coords = regressor(input_images, pred_heatmaps[-1].detach()).detach().cpu().squeeze().numpy()
            pred_coords_map_back = inverse_affine(arg, pred_coords, bbox)

            time_records.append(time.time() - start)

            error_rate_i = calc_error_rate_i(
                arg.dataset,
                pred_coords_map_back,
                coords_xy[0].numpy(),
                error_normalize_factor
            )

            if arg.eval_visual:
                eval_heatmap(arg, pred_heatmaps[-1], img_name, bbox, save_img=arg.save_img)
                eval_pred_points(arg, pred_coords, img_name, bbox, save_img=arg.save_img)

            failure_count = failure_count + 1 if error_rate_i > max_threshold else failure_count
            error_rate.append(error_rate_i)

    area_under_curve, auc_record = calc_auc(arg.dataset, arg.split, error_rate, max_threshold)
    error_rate = sum(error_rate) / dataset_size[arg.dataset][arg.split] * 100
    failure_rate = failure_count / dataset_size[arg.dataset][arg.split] * 100

    print('\nEvaluating results:\n# AUC:          {:.4f}\n# Error Rate:   {:.2f}%\n# Failure Rate: {:.2f}%\n'.format(
        area_under_curve, error_rate, failure_rate))
    print('Average speed: {:.2f}FPS'.format(1./np.mean(np.array(time_records))))


def evaluate_with_gt_heatmap(arg):
    devices = torch.device('cuda:' + arg.gpu_id)
    error_rate = []
    failure_count = 0
    max_threshold = arg.max_threshold

    testset = GeneralDataset(dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    print('*****  Evaluating with ground truth heatmap  *****')
    print('Evaluating parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Epoch of the model: ' + str(arg.eval_epoch) + '\n' +
          '# Normalize way:      ' + arg.norm_way + '\n' +
          '# Max threshold:      ' + str(arg.max_threshold) + '\n')

    print('Loading network...')
    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2 * kp_num[arg.dataset])
    regressor = load_weights(regressor, arg.save_folder + arg.dataset + '_regressor_' + str(arg.eval_epoch) + '.pth',
                             devices)
    if arg.cuda:
        regressor = regressor.cuda(device=devices)
    regressor.eval()
    print('Loading network done!\nStart testing...')

    time_records = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            start = time.time()

            input_images, gt_coords_xy, gt_heatmap, coords_xy, bbox, img_name = data
            bbox = bbox.squeeze().numpy()
            error_normalize_factor = calc_normalize_factor(arg.dataset, coords_xy.numpy(), arg.norm_way) \
                if arg.norm_way in ['inter_pupil', 'inter_ocular'] else (bbox[2] - bbox[0])
            input_images = input_images.unsqueeze(1)
            input_images = input_images.cuda(device=devices)
            gt_heatmap = gt_heatmap.cuda(device=devices)

            pred_coords = regressor(input_images, gt_heatmap).detach().cpu().squeeze().numpy()
            pred_coords_map_back = inverse_affine(arg, pred_coords, bbox)

            time_records.append(time.time() - start)

            error_rate_i = calc_error_rate_i(
                arg.dataset,
                pred_coords_map_back,
                coords_xy[0].numpy(),
                error_normalize_factor
            )

            if arg.eval_visual:
                eval_gt_pred_points(arg, gt_coords_xy, pred_coords, img_name, bbox, save_img=arg.save_img)

            failure_count = failure_count + 1 if error_rate_i > max_threshold else failure_count
            error_rate.append(error_rate_i)

    area_under_curve, auc_record = calc_auc(arg.dataset, arg.split, error_rate, max_threshold)
    error_rate = sum(error_rate) / dataset_size[arg.dataset][arg.split] * 100
    failure_rate = failure_count / dataset_size[arg.dataset][arg.split] * 100

    print('\nEvaluating results:\n# AUC:          {:.4f}\n# Error Rate:   {:.2f}%\n# Failure Rate: {:.2f}%\n'.format(
        area_under_curve, error_rate, failure_rate))
    print('Average speed: {:.2f}FPS'.format(1. / np.mean(np.array(time_records))))


if __name__ == '__main__':
    evaluate(args)
