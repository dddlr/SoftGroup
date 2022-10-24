from .custom import CustomDataset
import numpy as np


class TreesDataset(CustomDataset):

    CLASSES = ('not_tree', 'is_tree')

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # Debug code
        #
        # print('instance_num', instance_num)
        # print('instance_pointnum', instance_pointnum)
        # print('instance_cls', instance_cls)
        # print('pt_offset_label', pt_offset_label)

        instance_cls = [x if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, aug_prob=1.0):
        xyz_middle = self.dataAugment(xyz, True, True, True, aug_prob)
        xyz = xyz_middle * self.voxel_cfg.scale

        if 'scale_vector' in self.voxel_cfg:
            print('Scaling coords by', self.voxel_cfg.scale_vector)
            xyz *= self.voxel_cfg.scale_vector

        # Disabled because in self.elastic, the size of the noise array
        # (i.e. np.abs(x).max(0).astype(np.int32) // gran + 3)
        # ends up being so massive it crashes everything
        #
        # if np.random.rand() < aug_prob:
        #     xyz = self.elastic(xyz, 6, 40.)
        #     xyz = self.elastic(xyz, 20, 160.)
        # xyz_middle = xyz / self.voxel_cfg.scale

        xyz = xyz - xyz.min(0)
        max_tries = 5
        while (max_tries > 0):
            xyz_offset, valid_idxs = self.crop(xyz)
            if valid_idxs.sum() >= self.voxel_cfg.min_npoint:
                xyz = xyz_offset
                break
            max_tries -= 1
        if valid_idxs.sum() < self.voxel_cfg.min_npoint:
            return None
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label

    def transform_test(self, xyz, rgb, semantic_label, instance_label):
        xyz_middle = self.dataAugment(xyz, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale

        if 'scale_vector' in self.voxel_cfg:
            print('Scaling coords by', self.voxel_cfg.scale_vector)
            xyz *= self.voxel_cfg.scale_vector

        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, semantic_label, instance_label
