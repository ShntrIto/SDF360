import logging
import argparse
import torch
from runner import Runner

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# ロガーの設定
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

# 引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=str, default='./confs/base.conf')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--mcube_threshold', type=float, default=0.0)
parser.add_argument('--is_continue', default=False, action="store_true")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--case', type=str, default='')
args = parser.parse_args()

torch.cuda.set_device(args.gpu)

runner = Runner(args.conf, args.mode, args.case, args.is_continue)

if args.mode == 'train':
        runner.train()
elif args.mode == 'validate_mesh':
    runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
elif args.mode == 'validate_image':
    runner.validate_image(idx=1, resolution_level=1)
elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
    _, img_idx_0, img_idx_1 = args.mode.split('_')
    img_idx_0 = int(img_idx_0)
    img_idx_1 = int(img_idx_1)
    runner.interpolate_view(img_idx_0, img_idx_1) # ここで gen_rays_between が必要になる TODO: 実装する