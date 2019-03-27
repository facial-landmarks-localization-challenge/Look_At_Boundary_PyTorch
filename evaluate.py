import tqdm
from dataset import GeneralDataset
from models import *
from utils import *


def evaluate(arg):
    devices = torch.device('cuda:'+args.gpu_id)
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
    
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            input_images, gt_keypoints, gt_heatmap, bbox, img_name = data
            error_normalize_factor = calc_normalize_factor(arg.dataset, gt_keypoints.numpy(), arg.norm_way)
            input_images = input_images.unsqueeze(1)
            input_images = input_images.cuda(device=devices)

            pred_heatmaps = estimator(input_images)
            pred_coords = regressor(input_images, pred_heatmaps[-1].detach()).detach().cpu()
            pred_coords_map = inverse_affine(arg, pred_coords, bbox)

            error_rate_i = calc_error_rate_i(
                arg.dataset,
                pred_coords_map,
                gt_keypoints[0].numpy(),
                error_normalize_factor
            )

            if args.eval_watch and error_rate_i < args.error_thresh:
                eval_heatmap(arg, pred_heatmaps[-1], img_name, bbox, save_only=True)
                eval_points(arg, pred_coords, img_name, bbox, save_only=True)

            failure_count = failure_count + 1 if error_rate_i > max_threshold else failure_count
            error_rate.append(error_rate_i)

    area_under_curve = calc_auc(arg.dataset, arg.split, error_rate, max_threshold)
    error_rate = sum(error_rate) / dataset_size[arg.dataset][arg.split] * 100
    failure_rate = failure_count / dataset_size[arg.dataset][arg.split] * 100

    print('\nEvaluating results:\n' +
          '# AUC:          ' + str(area_under_curve) + '\n' +
          '# Error Rate:   ' + str(error_rate) + '%\n' +
          '# Failure Rate: ' + str(failure_rate) + '%\n')


if __name__ == '__main__':
    evaluate(args)
