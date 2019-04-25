import os
import torch
import torch.nn as nn
from models import WingLoss, Estimator, Regressor, Discrim
from dataset import GeneralDataset
from utils import *
import tqdm

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.resume_folder):
    os.mkdir(args.resume_folder)


def train(arg):
    epoch = None
    devices = get_devices_list(arg)

    print('*****  Normal Training  *****')
    print('Training parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Batchsize:          ' + str(arg.batch_size) + '\n' +
          '# Num workers:        ' + str(arg.workers) + '\n' +
          '# PDB:                ' + str(arg.PDB) + '\n' +
          '# Use GPU:            ' + str(arg.cuda) + '\n' +
          '# Start lr:           ' + str(arg.lr) + '\n' +
          '# Max epoch:          ' + str(arg.max_epoch) + '\n' +
          '# Loss type:          ' + arg.loss_type + '\n' +
          '# Resumed model:      ' + str(arg.resume_epoch > 0))
    if arg.resume_epoch > 0:
        print('# Resumed epoch:      ' + str(arg.resume_epoch))

    print('Creating networks ...')
    estimator, regressor, discrim = create_model(arg, devices)
    estimator.train()
    regressor.train()
    if discrim is not None:
        discrim.train()
    print('Creating networks done!')

    optimizer_estimator = torch.optim.SGD(estimator.parameters(), lr=arg.lr, momentum=arg.momentum,
                                          weight_decay=arg.weight_decay)
    optimizer_regressor = torch.optim.SGD(regressor.parameters(), lr=arg.lr, momentum=arg.momentum,
                                          weight_decay=arg.weight_decay)
    optimizer_discrim = torch.optim.SGD(discrim.parameters(), lr=arg.lr, momentum=arg.momentum,
                                        weight_decay=arg.weight_decay) if discrim is not None else None

    if arg.loss_type == 'L2':
        criterion = nn.MSELoss()
    elif arg.loss_type == 'L1':
        criterion = nn.L1Loss()
    elif arg.loss_type == 'smoothL1':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = WingLoss(w=arg.wingloss_w, epsilon=arg.wingloss_e)

    print('Loading dataset ...')
    trainset = GeneralDataset(dataset=arg.dataset)
    print('Loading dataset done!')

    d_fake = (torch.zeros(arg.batch_size, 13)).cuda(device=devices[0]) if arg.GAN \
        else torch.zeros(arg.batch_size, 13)

    # evolving training
    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        forward_times_per_epoch, sum_loss_estimator, sum_loss_regressor = 0, 0., 0.
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                                 num_workers=arg.workers, pin_memory=True)

        if epoch in arg.step_values:
            optimizer_estimator.param_groups[0]['lr'] *= arg.gamma
            optimizer_regressor.param_groups[0]['lr'] *= arg.gamma
            optimizer_discrim.param_groups[0]['lr'] *= arg.gamma

        for data in tqdm.tqdm(dataloader):
            forward_times_per_epoch += 1
            input_images, gt_coords_xy, gt_heatmap, _, _, _ = data
            true_batchsize = input_images.size()[0]
            input_images = input_images.unsqueeze(1)
            input_images = input_images.cuda(device=devices[0])
            gt_coords_xy = gt_coords_xy.cuda(device=devices[0])
            gt_heatmap = gt_heatmap.cuda(device=devices[0])

            optimizer_estimator.zero_grad()
            heatmaps = estimator(input_images)
            loss_G = estimator.calc_loss(heatmaps, gt_heatmap)
            loss_A = torch.mean(torch.log2(1. - discrim(heatmaps[-1])))
            loss_estimator = loss_G + loss_A
            loss_estimator.backward()
            optimizer_estimator.step()

            sum_loss_estimator += loss_estimator

            optimizer_discrim.zero_grad()
            loss_D_real = -torch.mean(torch.log2(discrim(gt_heatmap)))
            loss_D_fake = -torch.mean(torch.log2(1.-torch.abs(discrim(heatmaps[-1].detach()) -
                                                              d_fake[:true_batchsize])))
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_discrim.step()

            optimizer_regressor.zero_grad()
            out = regressor(input_images, heatmaps[-1].detach())
            loss_regressor = criterion(out, gt_coords_xy)
            loss_regressor.backward()
            optimizer_regressor.step()

            d_fake = (calc_d_fake(arg.dataset, out.detach(), gt_coords_xy, true_batchsize,
                                  arg.batch_size)).cuda(device=devices[0])

            sum_loss_regressor += loss_regressor

        if (epoch+1) % arg.save_interval == 0:
            torch.save(estimator.state_dict(), arg.save_folder + 'estimator_'+str(epoch+1)+'.pth')
            torch.save(discrim.state_dict(), arg.save_folder + 'discrim_'+str(epoch+1)+'.pth')
            torch.save(regressor.state_dict(), arg.save_folder + arg.dataset+'_regressor_'+str(epoch+1)+'.pth')

        print('\nepoch: {:0>4d} | loss_estimator: {:.2f} | loss_regressor: {:.2f}'.format(
            epoch,
            sum_loss_estimator.item()/forward_times_per_epoch,
            sum_loss_regressor.item()/forward_times_per_epoch
        ))

    torch.save(estimator.state_dict(), arg.save_folder + 'estimator_'+str(epoch+1)+'.pth')
    torch.save(discrim.state_dict(), arg.save_folder + 'discrim_'+str(epoch+1)+'.pth')
    torch.save(regressor.state_dict(), arg.save_folder + arg.dataset+'_regressor_'+str(epoch+1)+'.pth')
    print('Training done!')


def train_with_gt_heatmap(arg):
    epoch = None
    devices = get_devices_list(arg)

    print('*****  Training with ground truth heatmap  *****')
    print('Training parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Batchsize:          ' + str(arg.batch_size) + '\n' +
          '# Num workers:        ' + str(arg.workers) + '\n' +
          '# PDB:                ' + str(arg.PDB) + '\n' +
          '# Use GPU:            ' + str(arg.cuda) + '\n' +
          '# Start lr:           ' + str(arg.lr) + '\n' +
          '# Lr step values:     ' + str(arg.step_values) + '\n' +
          '# Lr step gamma:      ' + str(arg.gamma) + '\n' +
          '# Max epoch:          ' + str(arg.max_epoch) + '\n' +
          '# Loss type:          ' + arg.loss_type + '\n' +
          '# Resumed model:      ' + str(arg.resume_epoch > 0))
    if arg.resume_epoch > 0:
        print('# Resumed epoch:      ' + str(arg.resume_epoch))

    print('Creating networks ...')
    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2 * kp_num[arg.dataset])
    regressor = load_weights(regressor, arg.resume_folder + arg.dataset + '_regressor_' +
                             str(arg.resume_epoch) + '.pth', devices_list[0]) if arg.resume_epoch > 0 else regressor
    regressor = regressor.cuda(device=devices[0])
    regressor.train()
    print('Creating networks done!')

    optimizer_regressor = torch.optim.SGD(regressor.parameters(), lr=arg.lr, momentum=arg.momentum,
                                          weight_decay=arg.weight_decay)

    if arg.loss_type == 'L2':
        criterion = nn.MSELoss()
    elif arg.loss_type == 'L1':
        criterion = nn.L1Loss()
    elif arg.loss_type == 'smoothL1':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = WingLoss(w=arg.wingloss_w, epsilon=arg.wingloss_e)

    print('Loading dataset ...')
    trainset = GeneralDataset(dataset=arg.dataset)
    print('Loading dataset done!')

    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        forward_times_per_epoch, sum_loss_regressor = 0, 0.
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                                 num_workers=arg.workers, pin_memory=True)

        if epoch in arg.step_values:
            optimizer_regressor.param_groups[0]['lr'] *= arg.gamma

        for data in tqdm.tqdm(dataloader):
            forward_times_per_epoch += 1
            input_images, gt_coords_xy, gt_heatmap, _, _, _ = data
            input_images = input_images.unsqueeze(1)
            input_images = input_images.cuda(device=devices[0])
            gt_coords_xy = gt_coords_xy.cuda(device=devices[0])
            gt_heatmap = gt_heatmap.cuda(device=devices[0])

            optimizer_regressor.zero_grad()
            out = regressor(input_images, gt_heatmap)
            loss_regressor = criterion(out, gt_coords_xy)
            loss_regressor.backward()
            optimizer_regressor.step()

            sum_loss_regressor += loss_regressor

        if (epoch + 1) % arg.save_interval == 0:
            torch.save(regressor.state_dict(), arg.save_folder + arg.dataset + '_regressor_' + str(epoch + 1) + '.pth')

        print('\nepoch: {:0>4d} | loss_regressor: {:.2f}'.format(
            epoch,
            sum_loss_regressor.item() / forward_times_per_epoch
        ))

    torch.save(regressor.state_dict(), arg.save_folder + arg.dataset + '_regressor_' + str(epoch + 1) + '.pth')
    print('Training done!')


if __name__ == '__main__':
    train(args)
