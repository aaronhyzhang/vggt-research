import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import cv2
import json
import open3d as o3d
from pathlib import Path
from PIL import Image

def read_exr_depth(filepath):
    """Read EXR depth file"""
    depth_array = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(depth_array.shape) == 3:
        depth_array = depth_array[:, :, 0]
    return depth_array

def read_camera_params(json_path):
    """Read camera parameters from JSON file"""
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def load_rgb_image(image_path):
    """Load RGB image and normalize to [0, 1]"""
    img = Image.open(image_path)
    img_array = np.array(img) / 255.0
    return img_array

def visualize_base_only(rgb_mode=True):
    
    predicted_dir = Path("predicted_output")
    
    base_points = np.load(predicted_dir / "predicted_points_3d.npy")
    valid_mask = np.isfinite(base_points).all(axis=1)
    base_points_valid = base_points[valid_mask]
    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(base_points_valid)
    colors_file = predicted_dir / "predicted_colors.npy"
    colors = np.load(colors_file)
    colors_valid = colors[valid_mask]
    base_pcd.colors = o3d.utility.Vector3dVector(colors_valid)
    title = "Point Cloud: RGB from saved colors"

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(base_pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    vis.run()
    vis.destroy_window()
    
    o3d.io.write_point_cloud("base_only.ply", base_pcd)


def visualize_gt_vs_predicted(pic):
    base_dir = Path(r"C:\Users\jerry\Documents\Code\0000")
    predicted_dir = Path("predicted_output")
    
    depth_path = base_dir / "depths" / f"00{pic}.exr"
    pose_path = base_dir / "poses" / f"00{pic}.json"
    

    gt_depth = read_exr_depth(str(depth_path)).astype(np.float64)
    camera_params = read_camera_params(str(pose_path))
    gt_h, gt_w = gt_depth.shape    

    pred_points = np.load(predicted_dir / "predicted_points_3d.npy")
    
    # Keep only finite predicted points
    pred_valid_mask = np.isfinite(pred_points).all(axis=1)
    pred_points = pred_points[pred_valid_mask]

    

    fx = camera_params["f_x"]
    fy = camera_params["f_y"]
    cx = camera_params["c_x"]
    cy = camera_params["c_y"]
    
    eps = 1e-3
    valid_mask_gt = np.isfinite(gt_depth) & (gt_depth > eps)
    
    u_coords, v_coords = np.meshgrid(np.arange(gt_w), np.arange(gt_h))
    z = gt_depth[valid_mask_gt]
    u = u_coords[valid_mask_gt]
    v = v_coords[valid_mask_gt]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    gt_points = np.stack([x, y, z], axis=1)

    gt_norms = np.linalg.norm(gt_points, axis=1)
    pred_norms = np.linalg.norm(pred_points, axis=1)
    
    gt_norm_mean = np.mean(gt_norms)
    pred_norm_mean = np.mean(pred_norms)
    
    if pred_norm_mean < 1e-8:
        scale = 1.0
    else:
        scale = gt_norm_mean / pred_norm_mean
    
    mu_gt = gt_points.mean(axis=0)
    mu_pred = pred_points.mean(axis=0)
    
    translation = mu_gt - scale * mu_pred
    
    pred_points_aligned = scale * pred_points + translation
    
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    gt_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # GREEN
    
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_points_aligned)
    pred_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # RED

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="GT (GREEN) vs Predicted (RED)",
        width=1280,
        height=720,
    )
    vis.add_geometry(gt_pcd)
    vis.add_geometry(pred_pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.0, 0.0, 0.0])
    
    vis.run()
    vis.destroy_window()

    o3d.io.write_point_cloud("gt_green.ply", gt_pcd)
    o3d.io.write_point_cloud("pred_red_aligned.ply", pred_pcd)

if __name__ == "__main__":
    MODE = "norm"
    
    if MODE == "norm":
        visualize_base_only(rgb_mode=True)
    
    elif MODE == "compare":
        visualize_gt_vs_predicted(73)
