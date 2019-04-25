import argparse

parser = argparse.ArgumentParser(description='LAB')

# dataset
parser.add_argument('--dataset_route', default='/home/jin/new_datasets/', type=str)
parser.add_argument('--dataset',       default='300W',              type=str)
parser.add_argument('--split',         default='train',             type=str)

# dataloader
parser.add_argument('--crop_size',     default=256,                 type=int)
parser.add_argument('--batch_size',    default=4,                   type=int)
parser.add_argument('--workers',       default=8,                   type=int)
parser.add_argument('--shuffle',       default=True,                type=bool)
parser.add_argument('--PDB',           default=False,                type=bool)
parser.add_argument('--RGB',           default=False,                type=bool)
parser.add_argument('--trans_ratio',   default=0.1,                 type=float)
parser.add_argument('--rotate_limit',  default=20.,                 type=float)
parser.add_argument('--scale_ratio',   default=0.1,                 type=float)

# devices
parser.add_argument('--cuda',          default=True,                type=bool)
parser.add_argument('--gpu_id',        default='0',                 type=str)

# learning parameters
parser.add_argument('--momentum',      default=0.9,                 type=float)
parser.add_argument('--weight_decay',  default=5e-4,                type=float)
parser.add_argument('--lr',            default=2e-5,                type=float)
parser.add_argument('--gamma',         default=0.2,                 type=float)
parser.add_argument('--step_values',   default=[1000, 1500],        type=list)
parser.add_argument('--max_epoch',     default=2000,                type=int)

# losses setting
parser.add_argument('--loss_type',     default='smoothL1',          type=str,
                    choices=['L1', 'L2', 'smoothL1', 'wingloss'])
parser.add_argument('--wingloss_w',    default=10,                  type=int)
parser.add_argument('--wingloss_e',    default=2,                   type=int)

# resume training parameters
parser.add_argument('--resume_epoch',  default=0,                   type=int)
parser.add_argument('--resume_folder', default='./weights/ckpts/',  type=str)

# model saving parameters
parser.add_argument('--save_folder',   default='./weights/',        type=str)
parser.add_argument('--save_interval', default=100,                 type=int)

# model setting
parser.add_argument('--hour_stack',    default=4,                   type=int)
parser.add_argument('--msg_pass',      default=True,                type=bool)
parser.add_argument('--GAN',           default=True,                type=bool)
parser.add_argument('--fuse_stage',    default=4,                   type=int)
parser.add_argument('--sigma',         default=1.0,                 type=float)
parser.add_argument('--theta',         default=1.5,                 type=float)
parser.add_argument('--delta',         default=0.8,                 type=float)

# evaluate parameters
parser.add_argument('--eval_epoch',    default=900,                 type=int)
parser.add_argument('--max_threshold', default=0.1,                 type=float)
parser.add_argument('--norm_way',      default='inter_ocular',      type=str,
                    choices=['inter_pupil', 'inter_ocular', 'face_size'])
parser.add_argument('--eval_visual',   default=False,               type=bool)
parser.add_argument('--save_img',      default=False,                type=bool)

args = parser.parse_args()

assert args.resume_epoch < args.step_values[0]
assert args.resume_epoch < args.max_epoch
assert args.step_values[-1] < args.max_epoch
