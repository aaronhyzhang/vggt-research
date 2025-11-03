import torch
import numpy as np
import json
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
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

def main():
    print("Starting prediction for translation cropped zoom")
    base_img = "C:\\Users\\jerry\\Documents\\Code\\0000\\images\\0000.png"
    img = Image.open(base_img)
    img = np.array(img) # (720, 1280, 3)
    h, w, _ = img.shape
    h_half = h // 2
    w_half = w //2
    file_paths = []

    file_paths.append(base_img)
    files = get_quadrants(img, h_half=h_half, w_half=w_half, h=h, w=w)
    for file in files:
        file_paths.append(file)
    # print(file_paths)
    images = load_and_preprocess_images(file_paths).to(device)
    base_image = load_and_preprocess_images([base_img]).to(device)

    debug = False
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / "five_predictions.npy"

    if cache_file.exists() and debug:
        depth = np.load(cache_dir / "depth.npy").squeeze(0)
        depth_conf = np.load(cache_dir / "depth_conf.npy").squeeze(0)
        base_depth = np.load(cache_dir / "base_depth.npy").squeeze(0)
        base_depth_conf = np.load(cache_dir / "base_depth_conf.npy").squeeze(0)
    else:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                five_predictions = model(images)
                base_prediction = model(base_image)

        depth = five_predictions['depth'].cpu().numpy()
        depth_conf = five_predictions['depth_conf'].cpu().numpy()
        base_depth = base_prediction['depth'].cpu().numpy()
        base_depth_conf = base_prediction['depth_conf'].cpu().numpy()

        np.save(cache_dir / "depth.npy", depth)
        np.save(cache_dir / "depth_conf.npy", depth_conf)
        np.save(cache_dir / "base_depth.npy", base_depth)
        np.save(cache_dir / "base_depth_conf.npy", base_depth_conf)

        depth = depth.squeeze(0)
        depth_conf = depth_conf.squeeze(0)
        base_depth = base_depth.squeeze(0).squeeze(0)
        base_depth_conf = base_depth_conf.squeeze(0).squeeze(0)

        # print(base_depth.shape)
        # print(base_depth_conf.shape)
        # print(depth.shape)
        # print(depth_conf.shape)

    # depth should be a 2D image with HxW, each coord having a depth value
    # base depth shape (1,294,518,1)
    # base depth_conf shape (1,294,518)
    # depth shape = (5, 294, 518, 1)
    # depth_conf shape = (5, 294, 518)
    height = depth.shape[1] #294
    width = depth.shape[2] # 518
    fx,fy,cx,cy = save_camera_params(height, width)

    # the base image is 720x1280 and crops are 360x640


    half_h = height//2
    half_w = width // 2

    base_height = base_depth.shape[1]
    base_width = base_depth.shape[2]

    # set 1% conf threshold
    conf_thresh = 0.01

    first_img = depth[0, :, :,0]
    quad_tl = depth[1, :, :, 0] # (294, 518)
    quad_tr = depth[2, :,:,0]
    quad_bl = depth[3,:,:,0]
    quad_br = depth[4,:,:,0]

    conf_first = depth_conf[0]
    conf_tl = depth_conf[1] # (294, 518)
    conf_tr = depth_conf[2]
    conf_bl = depth_conf[3]
    conf_br = depth_conf[4]
    '''
    VGGT is fed 720x1280 base image and 4 crops of size 360x640. However, the load_and_preprocess function takes
    all of them and puts it into same size of divisible by 14 whatever since the model expects this. Therefore,
    in order to have the same shape matching, zoom in on the crops to reshape it without losing fidelity while
    still slicing the base image depth. 
    '''   

    # First zoom in the image to change the size so we can get proper sizes for the 3D point array
    from scipy.ndimage import zoom
    scale_h = half_h / height
    scale_w = half_w / width
    
    points_3d_tl_down = zoom(quad_tl, (scale_h, scale_w), order=1)  # (147, 259)
    points_3d_tr_down = zoom(quad_tr, (scale_h, scale_w), order=1)
    points_3d_bl_down = zoom(quad_bl, (scale_h, scale_w), order=1)
    points_3d_br_down = zoom(quad_br, (scale_h, scale_w), order=1)
    
    conf_tl_down = zoom(conf_tl, (scale_h, scale_w), order=1)
    conf_tr_down = zoom(conf_tr, (scale_h, scale_w), order=1)
    conf_bl_down = zoom(conf_bl, (scale_h, scale_w), order=1)
    conf_br_down = zoom(conf_br, (scale_h, scale_w), order=1)
    
    def depth_to_3d_points(depth_map, fx, fy, cx, cy, height_dim, width_dim):
        ys, xs = np.meshgrid(np.arange(height_dim), np.arange(width_dim), indexing='ij')
        X = (xs - cx) * depth_map / fx
        Y = (ys - cy) * depth_map / fy
        Z = depth_map
        return np.stack((X, Y, Z), axis=0).reshape(3, height_dim * width_dim)
    
    def depth_to_3d_points_spatial(depth_map, fx, fy, cx, cy):
        h, w = depth_map.shape
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        X = (xs - cx) * depth_map / fx
        Y = (ys - cy) * depth_map / fy
        Z = depth_map
        return np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    
    base_tl_points = depth_to_3d_points(first_img[0:half_h, 0:half_w], fx, fy, cx, cy, half_h, half_w)
    base_tr_points = depth_to_3d_points(first_img[0:half_h, half_w:width], fx, fy, cx, cy, half_h, half_w)
    base_bl_points = depth_to_3d_points(first_img[half_h:height, 0:half_w], fx, fy, cx, cy, half_h, half_w)
    base_br_points = depth_to_3d_points(first_img[half_h:height, half_w:width], fx, fy, cx, cy, half_h, half_w)

    points_3d_tl_spatial = depth_to_3d_points_spatial(points_3d_tl_down, fx, fy, cx, cy)  # (147, 259, 3)
    points_3d_tr_spatial = depth_to_3d_points_spatial(points_3d_tr_down, fx, fy, cx, cy)
    points_3d_bl_spatial = depth_to_3d_points_spatial(points_3d_bl_down, fx, fy, cx, cy)
    points_3d_br_spatial = depth_to_3d_points_spatial(points_3d_br_down, fx, fy, cx, cy)
    
    points_3d_tl_flat = depth_to_3d_points(points_3d_tl_down, fx, fy, cx, cy, half_h, half_w)  # (3, 147*259)
    points_3d_tr_flat = depth_to_3d_points(points_3d_tr_down, fx, fy, cx, cy, half_h, half_w)
    points_3d_bl_flat = depth_to_3d_points(points_3d_bl_down, fx, fy, cx, cy, half_h, half_w)
    points_3d_br_flat = depth_to_3d_points(points_3d_br_down, fx, fy, cx, cy, half_h, half_w)

    def find_mean(points):
        return points.mean(axis=1)

    base_tl_mean = find_mean(base_tl_points)
    base_tr_mean = find_mean(base_tr_points)
    base_bl_mean = find_mean(base_bl_points)
    base_br_mean = find_mean(base_br_points)

    tl_mean = find_mean(points_3d_tl_flat)
    tr_mean = find_mean(points_3d_tr_flat)
    bl_mean = find_mean(points_3d_bl_flat)
    br_mean = find_mean(points_3d_br_flat)

    final_points_3d = depth_to_3d_points_spatial(first_img, fx, fy, cx, cy)  # (294, 518, 3)
        
    quadrants = [
        (points_3d_tl_spatial, conf_tl_down, 0, half_h, 0, half_w, base_tl_mean, tl_mean),
        (points_3d_tr_spatial, conf_tr_down, 0, half_h, half_w, width, base_tr_mean, tr_mean),
        (points_3d_bl_spatial, conf_bl_down, half_h, height, 0, half_w, base_bl_mean, bl_mean),
        (points_3d_br_spatial, conf_br_down, half_h, height, half_w, width, base_br_mean, br_mean)
    ]

    for crop_3d_spatial, crop_conf, h1, h2, w1, w2, base_mean, crop_mean in quadrants:
        
        base_region_3d = final_points_3d[h1:h2, w1:w2, :]  # (147, 259, 3)
        base_conf = conf_first[h1:h2, w1:w2]  # (147, 259)
        
        valid_mask = (base_conf >= conf_thresh) & (crop_conf >= conf_thresh)
        
        if valid_mask.sum() > 0:
            
            translation = base_mean - crop_mean  # (3,)
            crop_3d_aligned = crop_3d_spatial + translation[np.newaxis, np.newaxis, :]  # (147, 259, 3)
            
            # use crop where it has higher confidence
            use_crop_mask = crop_conf > base_conf
            
            # apply mask to each 3D dimension, so either uses crop points or the original base depending on conf
            for dim in range(3):  # X, Y, Z
                final_points_3d[h1:h2, w1:w2, dim] = np.where(use_crop_mask, crop_3d_aligned[:, :, dim], base_region_3d[:, :, dim])
        else:
            print("no valid regions")
    
    
    final_depth = final_points_3d[:, :, 2]  # (294, 518)
    
    print(f"Final depth shape: {final_depth.shape}")
    print(f"Final 3D points shape: {final_points_3d.shape}")
    
    np.save(output_dir / "predicted_depth.npy", final_depth)
    np.save(output_dir / "predicted_points_3d.npy", final_points_3d)
    # np.save(output_dir / "base_depth.npy", first_img)
    
    if Path("temp_quadrants").exists():
        shutil.rmtree(Path("temp_quadrants"))
    print("Finished predictions")


def save_camera_params(height, width):
    gt_pose_path = Path(r"C:\Users\jerry\Documents\Code\0000\poses\0000.json")
    with open(gt_pose_path, 'r') as f:
        gt_camera_params = json.load(f)

    # scale the camera params since VGGT outputs 294x518 while GT image is 720x1280.
    # multiply focal xy and principal point xy by ratio 
    
    fx = gt_camera_params['f_x'] * (width / 1280)
    fy = gt_camera_params['f_y'] * (height / 720)
    cx = gt_camera_params['c_x'] * (width / 1280)
    cy = gt_camera_params['c_y'] * (height / 720)
    
    camera_params = {
        'intrinsics': {
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy)
        },
        'resolution': {
            'height': int(height),
            'width': int(width)
        }
    }
    
    with open(output_dir / "predicted_camera_params.json", 'w') as f:
        json.dump(camera_params, f, indent=2)

    return fx,fy,cx,cy
    

if __name__ == "__main__":
    main()
