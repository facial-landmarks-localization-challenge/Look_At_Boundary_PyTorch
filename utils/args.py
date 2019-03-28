import argparse

parser = argparse.ArgumentParser(description='LAB')

# dataset
parser.add_argument('--dataset_route', default='/media/zhijun/DISK2/Champagne_Jin/datasets/facial_landmark/datasets', type=str,   help='directory of all the facial landmark datasets')
parser.add_argument('--dataset',       default='300W',              type=str,   help='dataset used')
parser.add_argument('--split',         default='train',             type=str,   help='the split of dataset')

# dataloader
parser.add_argument('--crop_size',     default=256,                 type=int,   help='network input img size')
parser.add_argument('--batch_size',    default=8,                   type=int,   help='bastch size')
parser.add_argument('--workers',       default=8,                   type=int,   help='number of workers used in dataload')
parser.add_argument('--shuffle',       default=True,                type=bool,  help='dataloading shuffle(True) or not(False)')
parser.add_argument('--PDB',           default=True,                type=bool,  help='do(True) pose-based data balance or not(False)')
parser.add_argument('--RGB',           default=False,               type=bool,  help='input rgb img(True) or gray img(False)')
parser.add_argument('--trans_ratio',   default=0.1,                 type=float, help='data augment translation ratio of the bbox')
parser.add_argument('--ratote_limit',  default=20.,                 type=float, help='data augment ratotion angle limitation')
parser.add_argument('--scale_ratio',   default=0.08,                type=float, help='data augment rescale +/- ratio')

# devices
parser.add_argument('--cuda',          default=True,                type=bool,  help='use cuda to train model')
parser.add_argument('--gpu_id',        default='0',                 type=str,   help='gpu id, if have more, use 0,2,3 in this way')

# learning parameters
parser.add_argument('--momentum',      default=0.9,                 type=float, help='momentum')
parser.add_argument('--weight_decay',  default=5e-4,                type=float, help='Weight decay for SGD')
parser.add_argument('--lr',            default=4e-6,                type=float, help='initial learning rate')
parser.add_argument('--gamma',         default=0.1,                 type=float, help='Gamma update for SGD')
parser.add_argument('--step_values',   default=[2000],          type=list,  help='lr update epoch list')
parser.add_argument('--max_epoch',     default=2500,                 type=int,   help='max epoch for training')

# losses
parser.add_argument('--loss_type',     default='wingloss',                type=str,   choices=['L1', 'L2', 'smoothL1', 'wingloss'])
parser.add_argument('--wingloss_w',    default=10,                  type=int,   help='param w for wingloss')
parser.add_argument('--wingloss_e',    default=2,                   type=int,   help='param epsilon for wingloss')

# resume parameters
parser.add_argument('--resume_epoch',  default=1600,                type=int,   help='resume epoch for training')
parser.add_argument('--resume_folder', default='./weights/ckpts/',  type=str,   help='directory to load models')
parser.add_argument('--regress_only',  default=False,               type=bool,  help='only fine tune regressor(True) or not(False)')

# net saving parameters
parser.add_argument('--save_folder',   default='./weights/',        type=str,   help='directory to save models')
parser.add_argument('--save_interval', default=100,                 type=int,   help='net saving interval')

# model setting
parser.add_argument('--hour_stack',    default=4,                   type=int,   help='stacks of estimator hourglass network')
parser.add_argument('--msg_pass',      default=True,                type=bool,  help='use msg passing(True) or not(False)')
parser.add_argument('--GAN',           default=True,                type=bool,  help='use GAN(True) or not(False)')
parser.add_argument('--fuse_stage',    default=4,                   type=int,   help='fuse stage of regressor')

# test parameters
parser.add_argument('--test_epoch',    default=1200,                type=int,   help='resume epoch for testing')
parser.add_argument('--max_threshold', default=0.1,                 type=float, help='resume epoch for testing')
parser.add_argument('--norm_way',      default='inter_pupil',       type=str,   choices=['inter_pupil', 'inter_ocular', 'face_size'])
parser.add_argument('--eval_watch',    default=False,               type=bool,  help='use eval_heatmap/points(True) or not')
parser.add_argument('--save_only',     default=True,                type=bool,  help='by eval_watch, without watch, just save pics')
parser.add_argument('--error_thresh',  default=0.043,               type=float, help='the eval_heatmap/points threshold')

args = parser.parse_args()

if args.regress_only:
    assert args.GAN is False

assert args.resume_epoch < args.step_values[0]
assert args.resume_epoch < args.max_epoch
assert args.step_values[-1] < args.max_epoch
