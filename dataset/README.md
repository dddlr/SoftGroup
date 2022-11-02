## S3DIS dataset

1\) Download the [S3DIS](http://buildingparser.stanford.edu/dataset.html#Download) dataset

2\) Put the ``Stanford3dDataset_v1.2.zip`` to ``SoftGroup/dataset/s3dis/`` folder and unzip

3\) Preprocess data
```
cd SoftGroup/dataset/s3dis
bash prepare_data.sh
```

After running the script the folder structure should look like below
```
SoftGroup
├── dataset
│   ├── s3dis
│   │   ├── Stanford3dDataset_v1.2
│   │   ├── preprocess
│   │   ├── preprocess_sample
│   │   ├── val_gt
```

## ScanNet v2 dataset

1\) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

2\) Put the downloaded ``scans`` and ``scans_test`` folder as follows.

```
SoftGroup
├── dataset
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

3\) Split and preprocess data
```
cd SoftGroup/dataset/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.
```
SoftGroup
├── dataset
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```

## Tree dataset

1\) Download the tree dataset, which should contain some `CL2_BQ31_2019_1000_XXXX_treecrowns.laz` files (where XXXX represents a four-digit number) and some corresponding `CL2_BQ31_2019_1000_XXXX_trees.zip` archives.

2\) Unzip the zip archives. The resulting folder structure might look like this:

```
CL2_BQ31_2019_1000_1836_treecrowns.laz
CL2_BQ31_2019_1000_1836_trees
CL2_BQ31_2019_1000_1836_trees.zip
CL2_BQ31_2019_1000_2035_treecrowns.laz
CL2_BQ31_2019_1000_2035_trees
CL2_BQ31_2019_1000_2035_trees.zip
CL2_BQ31_2019_1000_2135_treecrowns.laz
CL2_BQ31_2019_1000_2135_trees
CL2_BQ31_2019_1000_2135_trees.zip
CL2_BQ31_2019_1000_2137_treecrowns.laz
CL2_BQ31_2019_1000_2137_trees
CL2_BQ31_2019_1000_2137_trees.zip
CL2_BQ31_2019_1000_2641_treecrowns.laz
CL2_BQ31_2019_1000_2641_trees
CL2_BQ31_2019_1000_2641_trees.zip
CL2_BQ31_2019_1000_2645_treecrowns.laz
CL2_BQ31_2019_1000_2645_trees
CL2_BQ31_2019_1000_2645_trees.zip
```

3\) Run the preparation script:

```
python3 dataset/treesv4/prepare_data_inst_trees.py
```

Be sure to set `BASE_DIR` to the directory that contains the files from Step 2, and `OUTPUT_DIRECTORY` to the desired directory for the output point cloud files.

4\) Move the output files to the folders `train`, `val`, and `test` within `dataset/treesv4/` as desired. (These represent the training, validation, and test datasets.)

5\) Run this script:

```
python3 dataset/treesv4/prepare_data_inst_gttxt_trees.py
```
