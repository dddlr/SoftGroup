'''
Generate instance groundtruth .txt files (for evaluation). See
utils/utils_3d.py and utils/eval.py for where this data will be used.
'''

import numpy as np
import glob
import torch
import os

if __name__ == '__main__':
    NOT_TREE, IS_TREE = 1, 2
    split = 'val'
    files = sorted(glob.glob('{}/*.pth'.format(split)))
    scenes = [torch.load(i) for i in files]

    if not os.path.exists(split + '_gt'):
        os.mkdir(split + '_gt')

    for i in range(len(scenes)):
        # seg_labels: list containing 1 (not tree) and 2 (is a tree)
        # instance_labels: list of numbers, representing different
        #                  tree instances
        _coords, _colors, seg_labels, instance_labels = scenes[i]
        scene_name = files[i].split('/')[-1][:-4]
        print('Processing {}/{} {}'.format(i + 1, len(scenes), scene_name))

        instance_labels_that_are_not_trees = set(
            instance_labels[seg_labels == NOT_TREE]
        )
        assert len(instance_labels_that_are_not_trees) == 1
        assert tuple(instance_labels_that_are_not_trees) == (0,)

        np.savetxt(
            os.path.join(split + '_gt', scene_name + '.txt'),
            instance_labels,
            fmt='%d',
        )
