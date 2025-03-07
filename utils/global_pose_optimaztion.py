import g2o
import numpy as np


def R_to_Quaternion(R: np.ndarray, eps=1e-5):
    """
    将旋转矩阵 R 转变为四元数表示
    :param R: 旋转矩阵 3x3
    :return: [q0, q1, q2, q3]
    """
    if R.shape != (3, 3):
        raise Warning("R must 3x3 rotation matrix")
    else:
        q0 = np.sqrt(np.trace(R) + 1) / 2
        if q0 < eps:
            print("q0 close to 0, try other transfer method!")
        q1 = (R[1][2] - R[2][1]) / (4 * q0)
        q2 = (R[2][0] - R[0][2]) / (4 * q0)
        q3 = (R[0][1] - R[1][0]) / (4 * q0)
        return [q0, q1, q2, q3]


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement,
                 information=np.identity(6),
                 robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            g2o.RobustKernelHuber()
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


if __name__ == '__main__':
    pose_file_path = "../test/sphere.g2o"
    pose_graph = PoseGraphOptimization()
    pose_file = open(pose_file_path)
    line = pose_file.readline()
    while line:
        data = line.split(' ')
        if data[0] == 'VERTEX_SE3:QUAT':
            pose_id = int(data[1])
            pose_info = np.array([float(i) for i in data[2:9]])
            # print(pose_id,pose_info)
            q = g2o.Quaternion(pose_info[6], pose_info[3], pose_info[4], pose_info[5])
            t = g2o.Isometry3d(q, pose_info[0:3])
            pose_graph.add_vertex(pose_id, t)
        if data[0] == 'EDGE_SE3:QUAT':
            pose_id_left = int(data[1])
            pose_id_right = int(data[2])
            pose_info = np.array([float(i) for i in data[3:10]])

            q = g2o.Quaternion(pose_info[6], pose_info[3], pose_info[4], pose_info[5])
            t = g2o.Isometry3d(q, pose_info[0:3])

            pose_graph.add_edge([pose_id_left, pose_id_right], t)

        line = pose_file.readline()

    pose_file.close()

    pose_graph.optimize(20)
    pose_graph.save('result.g2o')
