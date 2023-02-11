import json

import imageio
import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue


class ImageReader(object):
    def __init__(self, ids, timestamps=None, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10  # 10 images ahead of current index
        self.waiting = 1.5  # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if len(img.shape) >= 3:
            # raise ValueError("read failed: {}, with shape {}".format(path, img.shape))
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # print(img.shape)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)

    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue

            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype

    @property
    def shape(self):
        return self[0].shape


class PngDepthReader(ImageReader):

    def read(self, path):
        depth = cv2.imread(path, -1)
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        # print(depth.shape)
        if self.cam is None:
            return depth
        else:
            return self.cam.rectify(depth)


class ICLNUIMDataset(object):
    '''
    path example: 'path/to/your/ICL-NUIM R-GBD Dataset/living_room_traj0_frei_png'
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        481.20, 480.0, 319.5, 239.5, 5000)

    def __init__(self, path):
        path = os.path.expanduser(path)
        self.rgb = ImageReader(self.listdir(os.path.join(path, 'rgb')))
        self.depth = ImageReader(self.listdir(os.path.join(path, 'depth')))
        self.timestamps = None

    def sort(self, xs):
        return sorted(xs, key=lambda x: int(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.rgb)


def make_pair(matrix, threshold=1):
    assert (matrix >= 0).all()
    pairs = []
    base = defaultdict(int)
    while True:
        i = matrix[:, 0].argmin()
        min0 = matrix[i, 0]
        j = matrix[0, :].argmin()
        min1 = matrix[0, j]

        if min0 < min1:
            i, j = i, 0
        else:
            i, j = 0, j
        if min(min1, min0) < threshold:
            pairs.append((i + base['i'], j + base['j']))

        matrix = matrix[i + 1:, j + 1:]
        base['i'] += (i + 1)
        base['j'] += (j + 1)

        if min(matrix.shape) == 0:
            break
    return pairs


class TUMRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        525.0, 525.0, 319.5, 239.5, 5000)

    def __init__(self, path, register=True, depth_dir="depth"):
        path = os.path.expanduser(path)

        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'rgb')
            depth_ids, depth_timestamps = self.listdir(path, depth_dir)
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'rgb')
            depth_imgs, depth_timestamps = self.listdir(path, depth_dir)

            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2 / 3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])

        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

    def sort(self, xs):
        return sorted(xs, key=lambda x: float(x[:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        for name in self.sort(files):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)


class ReconDataset(TUMRGBDDataset):
    cam = namedtuple('camera', 'fx fy cx cy scale')(
        520.90, 521.00, 325.14, 249.70, 1000)


class ReconDataset2(TUMRGBDDataset):
    cam = namedtuple('camera', 'fx fy cx cy scale')(
        786.62, 589.96, 320.0, 240.00, 1000)


class BlenderDataset:
    cam = None

    def __init__(self, path, register=True):
        depth_base_dir = os.path.join(path, "depth")
        rgb_dir = os.path.join(path, "rgb")
        rgb_list = []
        depth_list = []
        self.scale_factor = 10000
        with open(os.path.join(rgb_dir, 'transforms.json'), 'r') as fp:
            metas = json.load(fp)
        for idx, frame in enumerate(metas['frames']):
            img_name = frame['file_path']
            img_name_ = "Image0" + img_name[2:]  # ignore './'
            img_path = os.path.join(rgb_dir, frame['file_path'])
            depth_path = os.path.join(depth_base_dir, img_name_)
            rgb_list.append(img_path)
            depth_list.append(depth_path)

        img0 = imageio.imread(rgb_list[0])
        H, W = img0.shape[:2]
        camera_angle_x = float(metas['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        self.cam = namedtuple('camera', 'fx fy cx cy scale')(
            focal, focal, W / 2, H / 2, self.scale_factor)
        self.timestamps = None
        self.rgb = ImageReader(rgb_list)
        self.depth = PngDepthReader(depth_list)

    def sort(self, xs):
        return sorted(xs, key=lambda x: float(x[:-4]))

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)
