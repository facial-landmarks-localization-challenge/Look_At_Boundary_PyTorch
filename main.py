from utils import *
import numpy as np
import time

anno = get_annotations_list(args.dataset, args.split, ispdb=args.PDB)
time_record = []
for iii, jjj in enumerate(anno):
    start = time.time()
    pic_affine_t, gt_keypoints_t, gt_heatmap = getitem_from(args.dataset, args.split, jjj)
    time_record.append(time.time() - start)
    if iii > 500:
        break
print(np.mean(np.array(time_record)))
