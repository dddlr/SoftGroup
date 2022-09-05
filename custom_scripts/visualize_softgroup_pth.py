import os
import numpy as np
import torch
import open3d as o3d
import random
import sys

assert len(sys.argv) == 3

# e.g. transform_train_scene_2235_0_1.pth
filename = sys.argv[1]

def classification_to_color(number_of_colors: int):
    colors = np.array([(random.random(), random.random(), random.random()) for _ in range(number_of_colors)])
    def classification_to_color_core(classification: int):
        if classification < 0:
            return np.array((0., 0., 0.))
        if classification == 0:
            return np.array((0.5, 0.5, 0.5))
        return np.array(colors[round(classification)])
    
    return classification_to_color_core


print('Processing points...')

coords, colors, sem_labels, inst_labels = torch.load(filename)
unique_instances = np.unique(inst_labels)

print('Colorizing points... (will take a while)')

sem_labels_coloring = o3d.utility.Vector3dVector(
    np.vectorize(classification_to_color(len(sem_labels)), signature='()->(3)')(sem_labels)
)
instance_labels_coloring = o3d.utility.Vector3dVector(
    np.vectorize(classification_to_color(int(max(unique_instances)) + 1), signature='()->(3)')(inst_labels)
)
original_coloring = o3d.utility.Vector3dVector(
    (colors + 1) / 2
)

print('Creating Open3D visualization...')

geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(coords)
# geom.colors = sem_labels_coloring

if sys.argv[2] == 'inst':
    geom.colors = instance_labels_coloring
else:
    geom.colors = original_coloring

o3d.visualization.draw_geometries([geom])
