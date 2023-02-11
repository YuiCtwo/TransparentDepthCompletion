#!/bin/bash
set -e
source activate pytorch1.10
python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_L50.yml --test_scene=val_seq1 --output_filename=depth_l50

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_sobel.yml --test_scene=val_seq1 --output_filename=depth_sobel

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_sn.yml --test_scene=val_seq1 --output_filename=depth_sn

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_rf.yml --test_scene=val_seq1 --output_filename=depth_rf

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_L50.yml --test_scene=val_seq2 --output_filename=depth_l50

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_sobel.yml --test_scene=val_seq2 --output_filename=depth_sobel

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_sn.yml --test_scene=val_seq2 --output_filename=depth_sn

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_rf.yml --test_scene=val_seq2 --output_filename=depth_rf

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_L50.yml --test_scene=val_seq3 --output_filename=depth_l50

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_sobel.yml --test_scene=val_seq3 --output_filename=depth_sobel

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_sn.yml --test_scene=val_seq3 --output_filename=depth_sn

python /home/ctwo/VDCNet/experiments/test_glass.py --cfg=experiments/dfnet_pt_rf.yml --test_scene=val_seq3 --output_filename=depth_rf