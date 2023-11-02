import os

import open3d as o3d
import trimesh
import numpy as np

from utils.visualize import make_dir

background_color = np.asarray([15, 14, 78]) / 255
pt_color = np.asarray([192, 192, 192]) / 255


def pt2mesh(ply_name):
    pcd = o3d.io.read_point_cloud(ply_name)
    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2])
    )

    tri_mesh = trimesh.Trimesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals)
    )
    return tri_mesh


def vis_density_pt(ply_path, save_dir, method, frame_num=240, start_angle=0, end_angle=np.pi):
    save_dir_path = make_dir(save_dir, "{}".format(method))
    vis_windows = o3d.visualization.Visualizer()
    vis_windows.create_window(width=640, height=480)
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd.paint_uniform_color(pt_color)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd.transform()
    # pcd.transform([[0.829238, 0.0994262, -0.54998, 1.58976],
    #                [0.261621, -0.938632, 0.224775, 0.873959],
    #                [-0.49388, -0.330279, -0.804362, 0.719803],
    #                [0, 0, 0, 1]])

    R = pcd.get_rotation_matrix_from_xyz((0, start_angle, 0))
    pcd.rotate(R)
    rotate_dt = (end_angle - start_angle) / frame_num
    vis_windows.add_geometry(pcd)
    opt = vis_windows.get_render_option()
    opt.background_color = background_color
    view_ctl = vis_windows.get_view_control()
    view_ctl.set_zoom(0.6)
    for i in range(frame_num):
        vis_windows.update_geometry(pcd)
        vis_windows.poll_events()
        vis_windows.update_renderer()
        vis_save_path = os.path.join(save_dir_path, "{}.png".format(i))
        vis_windows.capture_screen_image(vis_save_path, do_render=False)
        R = pcd.get_rotation_matrix_from_xyz((0, rotate_dt, 0))
        pcd.rotate(R)
    vis_windows.destroy_window()


if __name__ == "__main__":

    base_dir_list = ["/home/ctwo/blender_glass_data", "/home/ctwo/blender_glass_data/real_capture"]
    parameter_dict = {
        "val_seq1": (0, 0, np.pi * 1.5),
        "val_seq2": (0, 0, np.pi * 1.5),
        "seq1": (1, -np.pi / 3, np.pi / 3),
        "seq2": (1, -np.pi / 3, np.pi / 10),
        "seq3": (1, -np.pi / 2, -np.pi / 6),
        "seq5": (1, -np.pi / 4, np.pi / 8)
    }

    base_dir = "/home/ctwo/blender_glass_data"
    name_prefix = ".klg.ply"
    compared_method = ["sn", "cg", "trans"]
    # scene_list = ["seq1", "seq2", "seq3", "seq5"]
    scene_list = ["seq3"]
    for scene in scene_list:
        scene_render_parameter = parameter_dict[scene]
        base_dir = base_dir_list[scene_render_parameter[0]]
        for m in compared_method:
            ply_name = "scene_" + m + name_prefix
            scene_path = os.path.join(base_dir, scene)
            ply_path = os.path.join(scene_path, ply_name)
            vis_density_pt(
                ply_path,
                scene_path,
                m,
                start_angle=scene_render_parameter[1],
                end_angle=scene_render_parameter[2]
            )
