import argparse
import os
import os.path as osp
from operator import itemgetter

import numpy as np
import torch

# Example usage:
# (Assuming that tools/test.py has been run and its results have been saved to the "results/softgroup_train_treesv3_no_fixed_modules" directory)
#
# python3 tools/visualization_trees.py --room_name scene_2135_2_4 --task offset_semantic_pred --prediction_path results/softgroup_train_treesv3_no_fixed_modules


# Disable if CustomDataset has not been changed to export the augmented train/test data
USE_AUGMENTED_DATA = False
AUGMENTED_DATA_FOLDER = 'train_gt/'


# yapf:disable
COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255
# yapf:enable

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array([
    'invalid', 'not_tree', 'is_tree'
])
CLASS_COLOR = {
    'not_tree': [180, 150, 100],
    'is_tree': [100, 200, 100],
}
CLASS_LINE_COLOR = {
    'not_tree': [190, 171, 150],
    'is_tree': [171, 255, 171],
}
SEMANTIC_IDX2NAME = {
    1: 'not_tree',
    2: 'is_tree',
}


def get_coords_color(opt):
    if USE_AUGMENTED_DATA:
        # Use (xyz, rgb, semantic_label, inst_label) exported from data loader
        # (e.g. CustomDataset.__getitem__())
        print('Loading augmented data')
        pointcloud_file = osp.join(AUGMENTED_DATA_FOLDER, opt.room_name + '_gt.pth')
        xyz, rgb, label, inst_label, offset_coords = torch.load(pointcloud_file)
        xyz2 = xyz + offset_coords
        rgb_line = (rgb + 1) * 127.5
    else:
        coord_file = osp.join(opt.prediction_path, 'coords', opt.room_name + '.npy')
        color_file = osp.join(opt.prediction_path, 'colors', opt.room_name + '.npy')
        label_file = osp.join(opt.prediction_path, 'semantic_label', opt.room_name + '.npy')
        inst_label_file = osp.join(opt.prediction_path, 'gt_instance', opt.room_name + '.txt')
        xyz = np.load(coord_file)
        xyz2 = None
        rgb = np.load(color_file)
        rgb_line = None
        label = np.load(label_file)

        if opt.task.startswith('instance'):
            inst_label = np.array(open(inst_label_file).read().splitlines(), dtype=int)
            inst_label = inst_label % 1000 - 1

    rgb = (rgb + 1) * 127.5

    if (opt.task == 'semantic_gt'):
        label = label.astype(int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(
            itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'semantic_pred'):
        semantic_file = os.path.join(opt.prediction_path, 'semantic_pred', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~2
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    # predicted point offsets
    elif (opt.task == 'offset_semantic_pred'):
        semantic_file = os.path.join(opt.prediction_path, 'semantic_pred', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~2
        rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb_line = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_LINE_COLOR))

        offset_file = os.path.join(opt.prediction_path, 'offset_pred', opt.room_name + '.npy')
        assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        offset_coords = np.load(offset_file)
        xyz2 = xyz + offset_coords
        # xyz = offset_coords

        # only_show_trees = label != 1  # use ground truth labels to determine tree-ness
        only_show_trees = label_pred != 1  # use predicted labels to determine tree-ness
        xyz = xyz[only_show_trees]
        xyz2 = xyz2[only_show_trees]
        rgb = rgb[only_show_trees]
        rgb_line = rgb_line[only_show_trees]


    # ground truth(?) point offsets
    elif (opt.task == 'offset_semantic_label'):
        semantic_file = os.path.join(opt.prediction_path, 'semantic_label', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~2
        rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb_line = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_LINE_COLOR))

        offset_file = os.path.join(opt.prediction_path, 'offset_label', opt.room_name + '.npy')
        assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        offset_coords = np.load(offset_file)
        xyz2 = xyz + offset_coords

        only_show_trees = label != 1  # use ground truth labels to determine tree-ness
        # only_show_trees = label_pred != 1  # use predicted labels to determine tree-ness
        xyz = xyz[only_show_trees]
        xyz2 = xyz2[only_show_trees]
        rgb = rgb[only_show_trees]
        rgb_line = rgb_line[only_show_trees]


    # same color order according to instance pointnum
    elif (opt.task == 'instance_gt'):
        inst_label = inst_label.astype(int)
        print('Instance number: {}'.format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        ins_num = inst_label.max() + 1
        ins_pointnum = np.zeros(ins_num)
        for _ins_id in range(ins_num):
            ins_pointnum[_ins_id] = (inst_label == _ins_id).sum()
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            inst_label_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[
                _sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_rgb

    # same color order according to instance pointnum
    elif (opt.task == 'instance_pred'):
        instance_file = os.path.join(opt.prediction_path, 'pred_instance', opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #

        ins_num = len(masks)
        ins_pointnum = np.zeros(ins_num)
        inst_label = -100 * np.ones(rgb.shape[0]).astype(int)

        # sort score such that high score has high priority for visualization
        scores = np.array([float(x[-1]) for x in masks])
        sort_inds = np.argsort(scores)[::-1]
        for i_ in range(len(masks) - 1, -1, -1):
            print('processing mask', i_)
            i = sort_inds[i_]
            mask_path = os.path.join(opt.prediction_path, 'pred_instance', masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                print(f'Skipping mask {i}, masks[i][2] = {masks[i][2]} < 0.09')
                # continue
            mask = np.array(open(mask_path).read().splitlines(), dtype=int)
            print('{} {}: pointnum: {}'.format(i, masks[i], mask.sum()))
            ins_pointnum[i] = mask.sum()
            inst_label[mask == 1] = i
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            print(_sort_id)
            inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[
                _sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_pred_rgb

    if opt.task != 'offset_semantic_label' and opt.task != 'offset_semantic_pred':
        sem_valid = (label != -100)
        xyz = xyz[sem_valid]
        if xyz2 is not None:
            xyz2 = xyz2[sem_valid]
        rgb = rgb[sem_valid]
        if rgb is not None:
            rgb_line = rgb_line[sem_valid]

    return xyz, xyz2, rgb, rgb_line


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2],
                                                            int(color[0] * 255),
                                                            int(color[1] * 255),
                                                            int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction_path', help='path to the prediction results', default='./results')
    parser.add_argument('--room_name', help='room_name', default='scene0011_00')
    parser.add_argument(
        '--task',
        help='input/semantic_gt/semantic_pred/offset_semantic_pred/instance_gt/instance_pred',
        default='instance_pred')
    parser.add_argument('--out', help='output point cloud file in FILE.ply format')
    opt = parser.parse_args()

    # xyz2 is only valid for offset_semantic_pred
    xyz, xyz2, rgb, rgb2 = get_coords_color(opt)
    points = xyz[:, :3]
    if xyz2 is not None:
        points2 = xyz2[:, :3]
    colors = rgb / 255
    if rgb2 is not None:
        colors2 = rgb2 / 255

    if opt.out != '' and opt.out is not None:
        assert '.ply' in opt.out, 'output cloud file should be in FILE.ply format'
        write_ply(points, colors, None, opt.out)
    else:
        import open3d as o3d
        print(f'Using Open3D version {o3d.__version__}')
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        if opt.task == 'offset_semantic_pred':
            # doesn't work
            # vis.get_render_option().line_width = 30.0
            vis.get_render_option().point_size = 5.0
        else:
            vis.get_render_option().point_size = 3.0
        vis.get_render_option().show_coordinate_frame = True
        vis.add_geometry(pc)

        if xyz2 is not None and rgb2 is not None:
            assert len(points) == len(points2)
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.concatenate((points, points2)))
            lines = np.array([(i, i + len(points)) for i in range(len(points))])
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(colors2)

            vis.add_geometry(ls)



        vis.run()
        vis.destroy_window()
