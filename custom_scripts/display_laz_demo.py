import os
import laspy
import numpy as np
import open3d as o3d
import random

VOXEL_GRID = False
BASE_DIR = '/local/gwo21/softgroup/treesv4_hand_annotated/'
AERIAL_VIEW_PATH = os.path.join(BASE_DIR, 'CL2_BQ31_2019_1000_2137_treecrowns.laz')
TREES_DIRECTORY = os.path.join(BASE_DIR, 'CL2_BQ31_2019_1000_2137_trees')


class KeyCallbacks:
    def __init__(self, geom, trees):
        self.key_to_callback = {
            ord('K'): self.toggle_geom,
        }
        self.show_geom = 0
        self.geom = geom
        self.trees = trees

    def toggle_geom(self, vis: 'o3d.cpu.pybind.visualization.Visualizer') -> 'bool':
        # http://www.open3d.org/docs/0.12.0/tutorial/visualization/customized_visualization.html
        if self.show_geom == 0:
            # 1: show both trees and geom
            for tree in self.trees:
                vis.add_geometry(tree, False)
        elif self.show_geom == 1:
            # 2: only show trees
            vis.remove_geometry(self.geom, False)
        elif self.show_geom == 2:
            # 0: only show geom
            for tree in self.trees:
                vis.remove_geometry(tree, False)
            vis.add_geometry(self.geom, False)
        
        self.show_geom = (self.show_geom + 1) % 3
        print(f'toggled geom from {not self.show_geom} to {self.show_geom}')

        return False


def classification_to_color(classification: 'int') -> 'np.ndarray':
    classifications = {
        0: (0.,0., 0.),
        1: (1.,0.,0.),
        2: (0.75,0.5,0.),
        3: (0.5,0.5,0.),
        4: (0.25,0.,0.75),
        5: (0.,0.,0.75),
        6: (0.5,0.25,1.),
        7: (0.,0.5,1),
        9: (0.,0.,1.),
        18: (0.5,0.,0.5),
    }
    return np.array(classifications[classification])


classification_to_color_vectorized = np.vectorize(classification_to_color, signature='()->(3)')


# ===============
# ===============

# https://medium.com/spatial-data-science/an-easy-way-to-work-and-visualize-lidar-data-in-python-eed0e028996c

# las = laspy.read('honours/TreePointClouds/CL2_BQ31_2019_1000_2137_buffer.laz')
# las = laspy.read('honours/TreePointClouds/CL2_BQ31_2019_1000_2137_buffer_prep_streettrees.laz')
las = laspy.read(AERIAL_VIEW_PATH)
# print(las)
# print(list(las.point_format.dimension_names))
# print(set(list(las.classification)))

point_data = np.vstack((las.X, las.Y, las.Z)).transpose()
print(point_data)
# Use colours from point cloud
colors = np.vstack((las.red / 255, las.green / 255, las.blue / 255)).transpose()
print(colors)
# Create new colours based on classification values
# colors = classification_to_color_vectorized(las.classification)

geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)
geom.colors = o3d.utility.Vector3dVector(colors)

tree_files = os.listdir(TREES_DIRECTORY)
trees = []

print('Loading trees...')

for tree_file in tree_files:
    # if tree_file.startswith('0.'):
        # This is the file that contains all of
        # the not-trees
        # continue
    las = laspy.read(os.path.join(TREES_DIRECTORY, tree_file))
    random_color = np.array((random.random(), random.random(), random.random()))
    point_data = np.vstack((las.X, las.Y, las.Z)).transpose()

    # colors = np.vstack((las.red / 255, las.green / 255, las.blue / 255)).transpose()
    colors = np.vectorize(lambda x: random_color, signature='()->(3)')(las.classification)
    # colors = classification_to_color_vectorized(las.classification)

    tree = o3d.geometry.PointCloud()
    tree.points = o3d.utility.Vector3dVector(point_data)
    tree.colors = o3d.utility.Vector3dVector(colors)

    trees.append(tree)



if VOXEL_GRID:
    # https://towardsdatascience.com/guide-to-real-time-visualisation-of-massive-3d-point-clouds-in-python-ea6f00241ee0
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(geom, voxel_size=0.1)
    o3d.visualization.draw_geometries([voxel_grid])
else:
    key_callbacks = KeyCallbacks(geom, trees)
    o3d.visualization.draw_geometries_with_key_callbacks([geom], key_callbacks.key_to_callback)
