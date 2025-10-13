import numpy as np
import OpenEXR
import json
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

def read_exr_depth(filepath):
    with OpenEXR.File(filepath, separate_channels=True) as infile:
        channels = infile.channels()
        channel_name = list(channels.keys())[0]
        depth_array = channels[channel_name].pixels
        return depth_array

def read_camera_params(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def depth_to_3d_points(depth_map, camera_params):
    height, width = depth_map.shape
    
    fx = camera_params['f_x']
    fy = camera_params['f_y']
    cx = camera_params['c_x']
    cy = camera_params['c_y']
    extrinsic = np.array(camera_params['extrinsic'])
    
    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    valid_mask = np.isfinite(depth_map) & (depth_map > 0.001)
    
    u_valid = u_coords[valid_mask]
    v_valid = v_coords[valid_mask]
    z_valid = depth_map[valid_mask]
    
    x_camera = (u_valid - cx) * z_valid / fx
    y_camera = (v_valid - cy) * z_valid / fy
    z_camera = z_valid
    
    points_camera = np.stack([x_camera, y_camera, z_camera, np.ones_like(z_camera)], axis=-1)
    camera_to_world = extrinsic
    
    points_world = points_camera @ camera_to_world.T
    
    points_3d = points_world[:, :3]
    points_3d[:, 0] *= -1 # had to flip x since words were mirrored
    return points_3d, valid_mask

def create_point_cloud(points_3d, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_point_cloud(pcd, window_name="3D Point Cloud"):
    pcd.estimate_normals()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    vis.add_geometry(pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    vis.run()
    vis.destroy_window()

def load_rgb_image(image_path):
    from PIL import Image
    img = Image.open(image_path)
    img_array = np.array(img) / 255.0
    return img_array

def main():
    base_dir = Path(r"C:\Users\jerry\Documents\Code\0000")
    depth_path = base_dir / "depths" / "0000.exr"
    pose_path = base_dir / "poses" / "0000.json"
    image_path = base_dir / "images" / "0000.png"
    
    depth_map = read_exr_depth(str(depth_path))
    depth_map = depth_map.astype(np.float64)
    
    camera_params = read_camera_params(str(pose_path))
    
    points_3d, valid_mask = depth_to_3d_points(depth_map, camera_params)
    
    colors = None
    if image_path.exists():
        rgb_image = load_rgb_image(str(image_path))
        colors = rgb_image[valid_mask]
    
    pcd = create_point_cloud(points_3d, colors)
    
    output_path = "reconstructed_scene.ply"
    o3d.io.write_point_cloud(output_path, pcd)
    
    visualize_point_cloud(pcd)
    return pcd

if __name__ == "__main__":
    pcd = main()
