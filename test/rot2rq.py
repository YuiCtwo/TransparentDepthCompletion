import os

import numpy as np
import time
from scipy.spatial.transform import Rotation

from cfg.datasets.BlenderGlass_dataset import BlenderGlass
test_scene = "val_seq3"
ds_root = "/home/ctwo/blender_glass_data"
ds_path = os.path.join(ds_root, test_scene)
depth_factor = 4000
dataset = BlenderGlass(root=ds_root,
                       scene=test_scene,
                       img_h=240,
                       img_w=320,
                       rgb_aug=False,
                       max_norm=True,
                       depth_factor=depth_factor
                       )
count = 0
data_length = len(dataset)

"""
| 0 0 1 0|   |    Rx    |
|-1 0 0 0|   |    Ry    |
|0 -1 0 0| * |   Rz     |
|0  0 0 1|    |   Rw    |
"""

factor = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

with open("./gt_trajectory.txt", "w+") as fp:
    fp.write("1,0,0,0,0,0,0,1")
    fp.write("\n")
    for i in range(data_length):
        count += 1
        R = dataset[i]["R_mat"].numpy()
        t = dataset[i]["t_vec"].numpy().tolist()
        rq = Rotation.from_matrix(R).as_quat()
        rq = np.expand_dims(rq, 0)
        rq = np.matmul(factor, rq.T)
        rq = rq.tolist()
        fp.write("{:d},{:f},{:f},{:f},{:f},{:f},{:f},{:f}".format(
            count, t[0], t[1], t[2], rq[0][0], rq[1][0], rq[2][0], rq[3][0]
        ))
        fp.write("\n")
