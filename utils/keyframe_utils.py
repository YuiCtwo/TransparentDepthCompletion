import numpy as np
import cv2

from utils.rgbd2pcd import compute_xyz


def get_highgrad_element(img, threshold=150):
    laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    laplacian = cv2.cvtColor(laplacian, cv2.COLOR_RGBA2GRAY)
    thresh, res = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # res = cv2.dilate(res, kernel)
    # res = cv2.erode(res, kernel)
    return res


def reproject(ref_depth, cur_data, R, t, camera):
    # 寻找在当前 cur_data 中对应的 ref_depth 的点的插值出的深度值
    _, h, w = ref_depth.shape
    pt_current = compute_xyz(ref_depth, camera).squeeze().transpose(2, 0, 1).reshape(3, -1)
    XYZ = np.matmul(R, pt_current)
    X = (XYZ[0, :] + t[0]).reshape(h, w)
    Y = (XYZ[1, :] + t[1]).reshape(h, w)
    Z = (XYZ[2, :] + t[2]).reshape(h, w)

    u = camera["fx"] * X / Z + camera["cx"]
    v = camera["fy"] * Y / Z + camera["cy"]
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    u[u < 0] = 0
    v[v < 0] = 0
    reprojected_data = cv2.remap(cur_data, u, v, cv2.INTER_NEAREST)
    # uv_mask = ((u >= 0) + (u < w) + (v >= 0) + (v < h))
    # uv = np.stack([u, v], dim=2).view(h, w, 2)
    return reprojected_data, (u, v)


def depth_fusion(cur_frame, nearest_keyframe, camera):
    """
    将当前帧投影到上一个关键帧然后做深度融合计算
    :param cur_frame: t
    :param nearest_keyframe: ki
    """
    # 在 ki Frame 上对齐 T_t^ki
    pose = np.linalg.inv(cur_frame.pose)
    R = pose[:3, :3]
    t = pose[:3, 3]
    reproject_depth, reproject_coord = reproject(cur_frame.D, nearest_keyframe.D, R, t, camera)
    reporject_umap = cv2.remap(cur_frame.U, reproject_coord[0], reproject_coord[1], cv2.INTER_CUBIC)
    renewed_depth = weighted_sum(nearest_keyframe.D, reproject_depth, nearest_keyframe.U, reporject_umap)
    renewed_umap = (reporject_umap * nearest_keyframe.U) / (reporject_umap + nearest_keyframe.U)
    nearest_keyframe.D = renewed_depth
    nearest_keyframe.U = renewed_umap
    

def weighted_sum(a, b, wa, wb):
    return (a * wa + b * wb) / (wa + wb)


def uncertainty_propagate(cur_frame, nearest_frame, camera, delta_p=0.1):
    """
    计算 current_frame 的不确定图
    :param cur_frame:
    :param nearest_frame:
    :param camera:
    :return:
    """
    # init uncertainty_map
    # pose = np.linalg.inv(cur_frame.pose)
    # R = cur_frame.R
    # t = cur_frame.t
    reproject_depth, _ = reproject(nearest_frame.D, cur_frame.D, cur_frame.R, cur_frame.t, camera)
    # reproject_depth, _ = reproject(cur_frame.D, nearest_frame.D, R, t, camera)
    umap = (cur_frame.D - reproject_depth) ** 2 + 1e-9
    # uncertainty_propagate
    mask = nearest_frame.D > 0
    # ========= avoid zero divide
    tilde_umap = np.zeros_like(nearest_frame.D)
    tilde_umap[mask] = ((reproject_depth / (nearest_frame.D + 1e-9)) * nearest_frame.U + delta_p**2)[mask]
    # =========
    uncertainty_map = (tilde_umap * umap) / (tilde_umap + umap)
    renewed_depth = weighted_sum(nearest_frame.D, reproject_depth, tilde_umap, umap)
    return uncertainty_map, renewed_depth


class KeyFrame:

    def __init__(self, pose, depth, rgb, grey, obj_mask, uncertainty=None):
        self.pose = pose  # 相对于上一个 KeyFrame 的位姿变化
        if self.pose is not None:
            self.R = pose[:3, :3]
            self.t = pose[:3, 3]
        else:
            self.R = np.eye(3)
            self.t = np.random.rand(3, 1) / 1000
            # self.t = np.zeros((3, 1))
        self.D = depth
        self.U = uncertainty
        self.rgb = rgb
        self.grey = grey.squeeze()
        self.use_cuda = False
        self.obj_mask_not = np.logical_not(obj_mask)
        # self._rgb = self.rgb.transpose(2, 0, 1)
        self.high_grad_mask = get_highgrad_element(self.rgb)
        self.zero_mask = self.D > 0
        self.high_grad_mask = np.logical_and(self.high_grad_mask, self.zero_mask).squeeze()
        self.high_grad_mask = np.logical_and(self.high_grad_mask, self.obj_mask_not)
        self.inverse_depth_variance = 0.0
        self._get_inverse_depth_variance()

    def _get_inverse_depth_variance(self):
        mask = np.expand_dims(self.high_grad_mask, axis=0)
        inverse_depth = self.D[mask > 0]
        inverse_depth = (1 / (inverse_depth + 1e-9))
        self.inverse_depth_variance = np.var(inverse_depth)

    def cuda(self):
        return self

    def numpy(self):
        return self

    def update_pose(self, pose):
        self.pose = pose.clone()
        self.R = pose[:3, :3]
        self.t = pose[:3, 3]

    @staticmethod
    def check_keyframe():
        return True
