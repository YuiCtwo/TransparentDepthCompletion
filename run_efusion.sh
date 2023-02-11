#!/bin/bash
set -e
source activate pytorch1.10

root_dir=/home/ctwo/blender_glass_data/val_seq2

python ./experiments/run_elasticfusion.py --base_dir=$root_dir --first_file=depth --second_file=rgb --drop_num=1

~/repo/pngtoklg/pngtoklg -w "$root_dir" -o  "$root_dir/scene.klg" -s 4000 -t

echo "Running ElasticFusion"
~/repo/ElasticFusion/ElasticFusion -l $root_dir/scene.klg -f -cal $root_dir/camera