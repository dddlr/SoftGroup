"""
Usage:

    python3 custom_scripts/test_mask_rcnn.py
"""
import glob
import numpy as np
import os.path as osp
import torch

from softgroup.evaluation import (evaluate_semantic_acc, evaluate_semantic_miou)
from softgroup.util import get_root_logger

HAND_ANNOTATED_FOLDER = 'dataset/treesv4/'
MASK_RCNN_FOLDER = 'dataset/trees_rcnn/'

SPLIT = 'test/'

def main():
    logger = get_root_logger()

    gt_folder = osp.join(HAND_ANNOTATED_FOLDER, SPLIT, '*.pth')
    gt_files = glob.glob(gt_folder)
    cnn_folder = osp.join(MASK_RCNN_FOLDER, SPLIT, '*.pth')
    cnn_files = glob.glob(cnn_folder)

    common_files = {a.split('/')[-1] for a in gt_files} & \
            {a.split('/')[-1] for a in cnn_files}

    cnn_semantic_list = []
    gt_semantic_list = []

    for filename in common_files:
        print(f'Processing {filename}')
        gt_filename = osp.join(HAND_ANNOTATED_FOLDER, SPLIT, filename)
        cnn_filename = osp.join(MASK_RCNN_FOLDER, SPLIT, filename)

        gt_coords, _colors, gt_semantic_info, _instance_info = torch.load(gt_filename)
        cnn_coords, _colors, cnn_semantic_info, _instance_info = torch.load(cnn_filename)

        assert len(gt_coords) == len(cnn_coords)
        assert len(gt_semantic_info) == len(cnn_semantic_info)

        # gt and cnn files list points in different orders, so we establish an indexing
        # that allows us to index both in a consistent way
        gt_index = np.lexsort((gt_coords[:,0], gt_coords[:,1], gt_coords[:,2]))
        cnn_index = np.lexsort((cnn_coords[:,0], cnn_coords[:,1], cnn_coords[:,2]))

        # Horribly inefficient, but we're not dealing with enough data for it to matter
        cnn_semantic_list.append(cnn_semantic_info[cnn_index])
        gt_semantic_list.append(gt_semantic_info[gt_index])

    evaluate_semantic_acc(
            cnn_semantic_list,
            gt_semantic_list,
            logger=logger,
    )

    evaluate_semantic_miou(
            cnn_semantic_list,
            gt_semantic_list,
            logger=logger,
    )


if __name__ == '__main__':
    main()
