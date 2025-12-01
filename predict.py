import torch
import numpy as np
import json
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, align_quads_to_base
from PIL import Image
import cv2 as cv
import shutil
import os        


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# # Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
# images = load_and_preprocess_images(image_names).to(device)

output_dir = Path("predicted_output")
output_dir.mkdir(exist_ok=True)

base_img = "C:\\Users\\jerry\\Documents\\Code\\0000\\images\\mug.png"
gt_pose_path = Path(r"C:\Users\jerry\Documents\Code\0000\poses\0000.json")



def build_image_paths(mode: str):
    """
    mode: 'base', 'quads', or 'base+quads'
    Returns a list of image paths in the exact order used for VGGT + colors.
    """
    mode = mode.lower()
    base_path = Path(base_img)
    paths = []

    # Only compute quadrants if needed
    quads = []
    if mode in ("quads", "base+quads"):
        img = Image.open(base_img)
        img_np = np.array(img)           # (H, W, 3)
        h, w, _ = img_np.shape
        h_half = h // 2
        w_half = w // 2
        quads = get_quadrants(img_np, h_half=h_half, w_half=w_half, h=h, w=w)  # [tl, tr, bl, br]

    if mode in ("base", "base+quads"):
        paths.append(base_path)

    if mode in ("quads", "base+quads"):
        paths.extend(quads)

    if not paths:
        raise ValueError(f"Unknown mode '{mode}'. Use 'base', 'quads', or 'base+quads'.")

    # Convert to strings for load_and_preprocess_images
    return [str(p) for p in paths]


def load_colors_for_paths(image_paths, height, width):
    """
    Given a list of image paths and target (height, width),
    return a (N_total, 3) array of colors aligned with flattened points.
    """
    colors_list = []
    for p in image_paths:
        img = Image.open(p).resize((width, height))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr[:, :, :3]                  # ensure RGB
        colors_list.append(arr.reshape(-1, 3))
    colors = np.vstack(colors_list)          # (S*H*W, 3)
    return colors


def predict_points_and_colors(mode: str, save=True):
    """
    mode: 'base', 'quads', or 'base+quads'
    Runs VGGT, unprojects depth to world coords, and returns:
        points_3d: (N, 3)
        colors:    (N, 3)
    If save=True, also writes npy files to output_dir.
    """
    # 1) Build image path list in the exact order we want
    image_paths = build_image_paths(mode)

    # 2) Run VGGT
    images = load_and_preprocess_images(image_paths).to(device)  # shape: (1, S, H, W, 3-like)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images)

    # 3) Depth + camera params
    depth = predictions["depth"].cpu().numpy()[0]  # (S, H, W, 1)
    S, H, W, _ = depth.shape

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        (images.shape[2], images.shape[3]),
    )
    extrinsic = extrinsic.cpu().numpy()[0]  # (S, 3, 4)
    intrinsic = intrinsic.cpu().numpy()[0]  # (S, 3, 3)
    world_points, point_masks = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)  # (S, H, W, 3), (S, H, W)

    if mode == "base+quads":
        world_points = align_quads_to_base(world_points, point_masks, base_index=0)

    # 6) Flatten to (N, 3) in the same order as image_paths
    points_3d = world_points.reshape(-1, 3)  # (S*H*W, 3)
    # aligned_points3d = world_points_aligned.reshape(-1, 3)
    # final_3d = np.vstack(points_3d, aligned_points3d)

    colors = load_colors_for_paths(image_paths, H, W)  # (S*H*W, 3)

    if save:
        np.save(output_dir / "predicted_points_3d.npy", points_3d)
        np.save(output_dir / "predicted_colors.npy", colors)

    return points_3d, colors

def get_quadrants(img, h_half, w_half, h, w) -> list:
    quad_dir = Path("temp_quadrants")
    quad_dir.mkdir(exist_ok=True)
    file_paths = []
    # top left
    top_left = img[:h_half, :w_half]
    img_tl = Image.fromarray(top_left)
    tl_path = quad_dir / "topleft.png"
    img_tl.save(tl_path)
    file_paths.append(tl_path)
    # print(tl_path)

    # top right
    top_right = img[:h_half, w_half: w]
    img_tr = Image.fromarray(top_right)
    tr_path = quad_dir / "topright.png"
    img_tr.save(tr_path)
    file_paths.append(tr_path)
    
    # bottom left
    bot_left = img[h_half:h, :w_half]
    img_bl = Image.fromarray(bot_left)
    bl_path = quad_dir / "bot_left.png"
    img_bl.save(bl_path)
    file_paths.append(bl_path)

    # bottom right
    bot_right = img[h_half:h, w_half:w]
    img_br = Image.fromarray(bot_right)
    br_path = quad_dir / "bot_right.png"
    img_br.save(br_path)
    file_paths.append(br_path)

    print("Finished saving quadrants")
    return file_paths

def test(mode):
    points_3d, colors = predict_points_and_colors(mode, save=True)
    print(f"Mode={mode}, points={points_3d.shape}, colors={colors.shape}")
    return points_3d, colors

if __name__ == "__main__":
    test("quads")
