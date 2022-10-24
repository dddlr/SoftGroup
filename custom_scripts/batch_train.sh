#!/bin/bash

# time ./tools/dist_train.sh configs/trees_final/trees_ignore_offset.yaml 1 >trees_ignore_offset.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_semantic_weight1.yaml 1 >trees_semantic_weight1.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_semantic_weight2.yaml 1 >trees_semantic_weight2.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_semantic_weight3.yaml 1 >trees_semantic_weight3.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_zscale1.yaml 1 >trees_zscale1.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_zscale2.yaml 1 >trees_zscale2.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_zscale3.yaml 1 >trees_zscale3.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_zscale4.yaml 1 >trees_zscale4.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_zscale5.yaml 1 >trees_zscale5.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_voxel1.yaml 1 >trees_voxel1.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_voxel2.yaml 1 >trees_voxel2.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_voxel3.yaml 1 >trees_voxel3.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_voxel4.yaml 1 >trees_voxel4.log 2>&1

# time ./tools/dist_train.sh configs/trees_final/trees_score1.yaml 1 >trees_score1.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_score2.yaml 1 >trees_score2.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_radius1.yaml 1 >trees_radius1.log 2>&1
# time ./tools/dist_train.sh configs/trees_final/trees_radius2.yaml 1 >trees_radius2.log 2>&1

# echo "trees_best"
# time ./tools/dist_train.sh configs/trees_final/trees_best.yaml 1 >trees_best.log 2>&1

  
# echo "trees_voxel4 (vs cnn)"
# time ./tools/dist_test.sh configs/trees_final/trees_voxel4_cnn.yaml work_dirs/trees_voxel4/latest.pth 1


echo "trees_best"
time ./tools/dist_test.sh configs/trees_final/trees_best.yaml work_dirs/trees_best/latest.pth 1 --out results/trees_best
echo "trees_voxel4"
time ./tools/dist_test.sh configs/trees_final/trees_voxel4.yaml work_dirs/trees_voxel4/latest.pth 1 --out results/trees_voxel4
echo "trees_voxel3"
time ./tools/dist_test.sh configs/trees_final/trees_voxel3.yaml work_dirs/trees_voxel3/latest.pth 1 --out results/trees_voxel3
echo "trees_voxel2"
time ./tools/dist_test.sh configs/trees_final/trees_voxel2.yaml work_dirs/trees_voxel2/latest.pth 1 --out results/trees_voxel2
echo "trees_voxel1"
time ./tools/dist_test.sh configs/trees_final/trees_voxel1.yaml work_dirs/trees_voxel1/latest.pth 1 --out results/trees_voxel1
echo "trees_zscale5"
time ./tools/dist_test.sh configs/trees_final/trees_zscale5.yaml work_dirs/trees_zscale5/latest.pth 1 --out results/trees_zscale5
echo "trees_zscale4"
time ./tools/dist_test.sh configs/trees_final/trees_zscale4.yaml work_dirs/trees_zscale4/latest.pth 1 --out results/trees_zscale4
echo "trees_zscale3"
time ./tools/dist_test.sh configs/trees_final/trees_zscale3.yaml work_dirs/trees_zscale3/latest.pth 1 --out results/trees_zscale3
echo "trees_zscale2"
time ./tools/dist_test.sh configs/trees_final/trees_zscale2.yaml work_dirs/trees_zscale2/latest.pth 1 --out results/trees_zscale2
echo "trees_zscale1"
time ./tools/dist_test.sh configs/trees_final/trees_zscale1.yaml work_dirs/trees_zscale1/latest.pth 1 --out results/trees_zscale1
echo "trees_semantic_weight3"
time ./tools/dist_test.sh configs/trees_final/trees_semantic_weight3.yaml work_dirs/trees_semantic_weight3/latest.pth 1 --out results/tree_semantic_weight3
echo "trees_semantic_weight2"
time ./tools/dist_test.sh configs/trees_final/trees_semantic_weight2.yaml work_dirs/trees_semantic_weight2/latest.pth 1 --out results/tree_semantic_weight2
echo "trees_semantic_weight1"
time ./tools/dist_test.sh configs/trees_final/trees_semantic_weight1.yaml work_dirs/trees_semantic_weight1/latest.pth 1 --out results/tree_semantic_weight1
echo "trees_ignore_offset"
time ./tools/dist_test.sh configs/trees_final/trees_ignore_offset.yaml work_dirs/trees_ignore_offset/latest.pth 1 --out results/trees_ignore_offset
echo "trees_base"
time ./tools/dist_test.sh configs/trees_final/trees_base.yaml work_dirs/trees_base/latest.pth 1 --out results/trees_base
