"""
Script to read Landcare Research's LAZ files, and convert them into something
that PointGroup can read.

Grant Wong, 2022

This differs from trees/prepare_data_inst_trees.py in that the points of each
chunk are moved to be centred on the origin.

"""

import os, laspy, numpy as np, torch
import multiprocessing as mp
from itertools import product


BASE_DIR = '/local/gwo21/softgroup/treesv4_hand_annotated/'
OUTPUT_DIRECTORY = '/csse/users/gwo21/uc-notes/honours/SoftGroup/dataset/treesv4/unsorted/'

COORDS_SCALE_FACTOR = 80

NOT_TREE = 1
IS_TREE = 2


def convert_tile(tile_number: str):
    filename = os.path.join(BASE_DIR, f'CL2_BQ31_2019_1000_{tile_number}_treecrowns.laz')
    tree_directory = os.path.join(BASE_DIR, f'CL2_BQ31_2019_1000_{tile_number}_trees')

    file = laspy.read(filename)

    # print('Dimensions available:')
    # for dimension in file.point_format.dimensions:
    #     print(f'{dimension} ', end='')

    total_coords_no = len(file.X)
    coords = np.full((total_coords_no, 3), (0., 0., 0.))
    colors = np.zeros((total_coords_no, 3))
    instance_labels = np.full((total_coords_no,), 0.)
    seg_labels = np.full((total_coords_no,), 0.)

    # We expect tree file names to be in the form
    # - 0.las
    # - 1.las
    # - 2.las
    # - ...
    tree_files = os.listdir(tree_directory)
    coords_ptr = 0

    # Duplicate points within a single tree are ok, but
    # we don't want points duplicated across trees -
    # this array helps us find whether the latter is
    # indeed the case
    #
    # We initialise this to an array of zeros so we can
    # concatenate arrays on to this later.
    unique_points = np.zeros([1, 3])

    for tree_filename in tree_files:
        assert len(tree_filename.split('.')) == 2
        tree_id = int(tree_filename.split('.')[0])

        # We assume that 0.las contains all of the not-tree points
        is_not_tree = (tree_id == 0)
        tree_file = laspy.read(os.path.join(tree_directory, tree_filename))
        tree_colors = np.stack((tree_file.red, tree_file.green, tree_file.blue), axis=-1) / 127.5 - 1
        tree_coords = np.stack((tree_file.x, tree_file.y, tree_file.z), axis=-1)


        # Total number of coordinates across all of the
        # trees shouldn't exceed the number of coordinates
        # in the treecrowns.laz file

        # print(tree_id, tree_filename, coords_ptr, total_coords_no)
        assert coords_ptr + len(tree_coords) <= total_coords_no

        coords[coords_ptr:coords_ptr+len(tree_coords)] = tree_coords
        colors[coords_ptr:coords_ptr+len(tree_coords)] = tree_colors
        instance_labels[coords_ptr:coords_ptr+len(tree_coords)] = tree_id
        seg_labels[coords_ptr:coords_ptr+len(tree_coords)] = NOT_TREE if is_not_tree else IS_TREE

        unique_points = np.concatenate((
            unique_points,
            np.unique(tree_coords, axis=0),
        ))

        coords_ptr += len(tree_coords)

    print()
    print(f'=== Processing tile {tile_number} ===')
    print(f'{tile_number} Total number of points filled: {coords_ptr}')
    print(f'{tile_number} Number of unique coords: {len(unique_points)}')
    print(f'{tile_number} Number of coords duplicated across trees: {len(unique_points) - len(np.unique(unique_points, axis=0))}')
    print(f'{tile_number} Total number of coords: {len(coords)}')

    # Make sure no points are duplicated across trees
    # (Points duplicated within trees are common and
    # harmless so we don't do anything about those)
    assert len(np.unique(unique_points, axis=0)) == len(unique_points)

    # ScanNet data uses 32-bit for these for some reason
    # and PointGroup expects 32-bit, so...
    coords = coords.astype('float32')
    colors = colors.astype('float32')

    # coords.shape = (n, 3)
    # colors.shape = (n, 3)
    # seg_labels.shape = (n,)
    # instance_labels.shape = (n,)
    return coords, colors, seg_labels, instance_labels


def chunk_tile(coords, colors, seg_labels, instance_labels, tile_number):
    # Assume that x and y determine east/west and north/south,
    # and z for elevation
    #
    # Tree data is typically
    # x: -25716 to 27281 (range: 52998)
    # y: -37428 to 39569 (range: 76998)
    # z:  -1235 to  7511 (range: 8747)

    x_min, x_max = coords[:,0].min(), coords[:,0].max()
    y_min, y_max = coords[:,1].min(), coords[:,1].max()

    print(f'x range: {x_min} to {x_max}')
    print(f'y range: {y_min} to {y_max}')

    no_of_chunks = (5, 5)
    chunk_width = (x_max - x_min) // no_of_chunks[0] + 1
    chunk_height = (y_max - y_min) // no_of_chunks[1] + 1

    total_points_chunked = 0
    
    for x_chunk_no, y_chunk_no in product(range(no_of_chunks[0]), range(no_of_chunks[1])):
        x_lower = x_min + x_chunk_no * chunk_width
        x_upper = x_min + (x_chunk_no+1) * chunk_width
        y_lower = y_min + y_chunk_no * chunk_width
        y_upper = y_min + (y_chunk_no+1) * chunk_width
        
        if x_chunk_no < no_of_chunks[0] - 1:
            in_x_range = (x_lower <= coords[:,0]) & (coords[:,0] < x_upper)
        else:
            in_x_range = (x_lower <= coords[:,0])

        if y_chunk_no < no_of_chunks[1] - 1:
            in_y_range = (y_lower <= coords[:,1]) & (coords[:,1] < y_upper)
        else:
            in_y_range = (y_lower <= coords[:,1])

        # Turn chunked coords/colours/seg labels/instance labels into
        # new arrays, because pytorch seems to write the entire array
        # when trying to save an array slice
        #
        # https://github.com/pytorch/pytorch/issues/40157
        chunk_coords = coords[in_x_range & in_y_range].copy()
        chunk_colors = colors[in_x_range & in_y_range].copy()
        chunk_seg_labels = seg_labels[in_x_range & in_y_range].copy()
        chunk_instance_labels = instance_labels[in_x_range & in_y_range].copy()

        # Scale down the coordinates so they are more similar to that
        # of ScanNet;
        # the coordinates of ScanNet data appear to be in the range (-4, 4) ish.
        # chunk_coords /= COORDS_SCALE_FACTOR

        # Move all the points so that the chunk is centred on the origin
        mean = chunk_coords.mean(0, dtype=np.float64)
        chunk_coords -= mean

        print(f'Generated chunk with tile {tile_number}, chunk no. ({x_chunk_no}, {y_chunk_no}) has {len(chunk_coords)} points; x: {x_lower} to {x_upper}, y: {y_lower} to {y_upper}')
        print(f'{tile_number} chunk coords min/max final value (scale {COORDS_SCALE_FACTOR}): (', end='')
        for i in range(3):
            print(f'{chunk_coords[:,i].min():.2f} to {chunk_coords[:,i].max():.2f}, ', end='')
        print(')')

        total_points_chunked += len(chunk_coords)

        tile_data = (chunk_coords, chunk_colors, chunk_seg_labels, chunk_instance_labels)
        tile_info = (tile_number, x_chunk_no, y_chunk_no)
        yield tile_data, tile_info

    # Sanity check: Ensure we didn't screw up our chunking by
    # overlooking/duplicating any points
    assert total_points_chunked == len(coords)


def process_tile(tile_number: str, save_to_file: bool = True):
    """Processes a tile, then splits it into square chunks."""
    print(f'Reading tile number {tile_number}')
    coords, colors, seg_labels, instance_labels = convert_tile(tile_number)
    print('Finished reading tile. Chunking...')

    for chunked_tile in chunk_tile(coords, colors, seg_labels, instance_labels, tile_number):
        tile_data, tile_info = chunked_tile
        tile_number, x_chunk_no, y_chunk_no = tile_info

        if save_to_file:
            torch.save(
                tile_data,
                os.path.join(
                    OUTPUT_DIRECTORY,
                    f'scene_{tile_number}_{x_chunk_no}_{y_chunk_no}.pth',
                ),
            )
            print(f'Saved: scene_{tile_number}_{x_chunk_no}_{y_chunk_no}.pth')
        else:
            raise NotImplementedError('Currently only supports saving to file, sorry!')


if __name__ == '__main__':
    print('Converting tile data...')
    files = ('1836', '2035', '2135', '2137', '2641', '2645')

    p = mp.Pool(processes=mp.cpu_count())
    p.map(process_tile, files)
    p.close()
    p.join()
