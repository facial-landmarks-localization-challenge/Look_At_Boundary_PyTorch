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

    testset = GeneralDataset(dataset=arg.dataset, split=arg.split, eval_flag=1)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    print('Evaluating parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Epoch of the model: ' + str(arg.test_epoch) + '\n' +
          '# Normalize way:      ' + arg.norm_way + '\n' +
          '# Max threshold:      ' + str(arg.max_threshold) + '\n')
    
    print('Loading network...')
    estimator = Estimator(stacks=arg.hour_stack, msg_pass=arg.msg_pass)
    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2*dataset_kp_num[arg.dataset])
    estimator = load_weights(estimator, arg.save_folder+'estimator_'+str(arg.test_epoch)+'.pth', devices)
    regressor = load_weights(regressor, arg.save_folder+arg.dataset+'_regressor_'+str(arg.test_epoch)+'.pth', devices)
    if arg.cuda:
        estimator = estimator.cuda(device=devices)
        regressor = regressor.cuda(device=devices)
    estimator.eval()
    regressor.eval()
    print('Loading network done!\nStart testing...')
    
    time_records = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            start = time.time()

            input_images, gt_keypoints, gt_heatmap, bbox, img_name = data
            error_normalize_factor = calc_normalize_factor(arg.dataset, gt_keypoints.numpy(), arg.norm_way)
            input_images = input_images.unsqueeze(1)
            input_images = input_images.cuda(device=devices)

            pred_heatmaps = estimator(input_images)
            pred_coords = regressor(input_images, pred_heatmaps[-1].detach()).detach().cpu()
            pred_coords_map = inverse_affine(arg, pred_coords, bbox)

            time_records.append(time.time() - start)

            error_rate_i = calc_error_rate_i(
                arg.dataset,
                pred_coords_map,
                gt_keypoints[0].numpy(),
                error_normalize_factor
            )

            if arg.eval_watch and error_rate_i < arg.error_thresh:
                eval_heatmap(arg, pred_heatmaps[-1], img_name, bbox, save_only=arg.save_only)
                eval_points(arg, pred_coords, img_name, bbox, save_only=arg.save_only)

            failure_count = failure_count + 1 if error_rate_i > max_threshold else failure_count
            error_rate.append(error_rate_i)

    area_under_curve = calc_auc(arg.dataset, arg.split, error_rate, max_threshold)
    error_rate = sum(error_rate) / dataset_size[arg.dataset][arg.split] * 100
    failure_rate = failure_count / dataset_size[arg.dataset][arg.split] * 100

    print('\nEvaluating results:\n# AUC:          {:.4f}\n# Error Rate:   {:.2f}%\n# Failure Rate: {:.2f}%\n'.format(
        area_under_curve, error_rate, failure_rate))
    print('Average speed: {:.2f}FPS'.format(1./np.mean(np.array(time_records))))


def evaluate_gthm_reg(arg):
    devices = torch.device('cuda:' + arg.gpu_id)
    error_rate = []
    failure_count = 0
    max_threshold = arg.max_threshold

    testset = GeneralDataset(dataset=arg.dataset, split=arg.split, eval_flag=1)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    print('Evaluating parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Epoch of the model: ' + str(arg.test_epoch) + '\n' +
          '# Normalize way:      ' + arg.norm_way + '\n' +
          '# Max threshold:      ' + str(arg.max_threshold) + '\n')

    print('Loading network...')
    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2 * dataset_kp_num[arg.dataset])
    regressor = load_weights(regressor, arg.save_folder + arg.dataset + '_regressor_' + str(arg.test_epoch) + '.pth',
                             devices)
    if arg.cuda:
        regressor = regressor.cuda(device=devices)
    regressor.eval()
    print('Loading network done!\nStart testing...')

    time_records = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            start = time.time()

            input_images, gt_keypoints, gt_heatmap, bbox, img_name = data
            error_normalize_factor = calc_normalize_factor(arg.dataset, gt_keypoints.numpy(), arg.norm_way)
            input_images = input_images.unsqueeze(1)
            input_images = input_images.cuda(device=devices)
            gt_heatmap = gt_heatmap.cuda(device=devices)

            pred_coords = regressor(input_images, gt_heatmap).detach().cpu()
            pred_coords_map = inverse_affine(arg, pred_coords, bbox)

            time_records.append(time.time() - start)

            error_rate_i = calc_error_rate_i(
                arg.dataset,
                pred_coords_map,
                gt_keypoints[0].numpy(),
                error_normalize_factor
            )

            if arg.eval_watch and error_rate_i < arg.error_thresh:
                eval_heatmap(arg, pred_heatmaps[-1], img_name, bbox, save_only=arg.save_only)
                eval_points(arg, pred_coords, img_name, bbox, save_only=arg.save_only)

            failure_count = failure_count + 1 if error_rate_i > max_threshold else failure_count
            error_rate.append(error_rate_i)

    area_under_curve = calc_auc(arg.dataset, arg.split, error_rate, max_threshold)
    error_rate = sum(error_rate) / dataset_size[arg.dataset][arg.split] * 100
    failure_rate = failure_count / dataset_size[arg.dataset][arg.split] * 100

    print('\nEvaluating results:\n# AUC:          {:.4f}\n# Error Rate:   {:.2f}%\n# Failure Rate: {:.2f}%\n'.format(
        area_under_curve, error_rate, failure_rate))
    print('Average speed: {:.2f}FPS'.format(1. / np.mean(np.array(time_records))))


def evaluate_one_img(arg, img_route):
    devices = torch.device('cuda:' + arg.gpu_id)

    print('Loading network...')
    estimator = Estimator(stacks=arg.hour_stack, msg_pass=arg.msg_pass)
    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2 * dataset_kp_num[arg.dataset])
    estimator = load_weights(estimator, arg.save_folder + 'estimator_' + str(arg.test_epoch) + '.pth', devices)
    regressor = load_weights(regressor, arg.save_folder + arg.dataset + '_regressor_' + str(arg.test_epoch) + '.pth',
                             devices)
    if arg.cuda:
        estimator = estimator.cuda(device=devices)
        regressor = regressor.cuda(device=devices)
    estimator.eval()
    regressor.eval()
    print('Loading network done!\nStart testing...')

    images = cv2.imread(img_route)
    images = cv2.resize(images, (256, 256))

    input_images = convert_img_to_gray(images)
    input_images = torch.Tensor(input_images)
    input_images = input_images.unsqueeze(0)
    input_images = input_images.unsqueeze(0).cuda()

    with torch.no_grad():
        pred_heatmaps = estimator(input_images)
        pred_coords = regressor(input_images, pred_heatmaps[-1].detach()).detach().cpu()

        pred_coords = pred_coords.squeeze().numpy()
        for coord_index in range(dataset_kp_num[arg.dataset]):
            cv2.circle(
                images,
                (int(pred_coords[2 * coord_index]), int(pred_coords[2 * coord_index + 1])),
                2,
                (0, 0, 255)
            )
        cv2.imshow('images', images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    evaluate_gthm_reg(args)
