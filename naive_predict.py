import torch
import numpy as np
import json
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
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

base_img = "C:\\Users\\jerry\\Documents\\Code\\0000\\images\\pic.png"
gt_pose_path = Path(r"C:\Users\jerry\Documents\Code\0000\poses\0000.json")


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

def base_4():
    print("Starting prediction for translation cropped zoom")
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
                base_prediction = model(base_image)
                five_predictions = model(images)

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
    fx,fy,cx,cy = save_camera_params(five_predictions, images.shape[2], images.shape[3])

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
    
    def depth_to_3d_points_spatial(depth_map, fx, fy, cx, cy):
        h, w = depth_map.shape
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        X = (xs - cx) * depth_map / fx
        Y = (ys - cy) * depth_map / fy
        Z = depth_map
        return np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    
    cx_tl = cx / 2  # Top-left quadrant center
    cy_tl = cy / 2
    cx_tr = cx / 2  # Top-right
    cy_tr = cy / 2
    cx_bl = cx / 2  # Bottom-left
    cy_bl = cy / 2
    cx_br = cx / 2  # Bottom-right
    cy_br = cy / 2
    
    base_tl_spatial = depth_to_3d_points_spatial(first_img[0:half_h, 0:half_w], fx, fy, cx, cy)
    base_tr_spatial = depth_to_3d_points_spatial(first_img[0:half_h, half_w:width], fx, fy, cx - half_w, cy)
    base_bl_spatial = depth_to_3d_points_spatial(first_img[half_h:height, 0:half_w], fx, fy, cx, cy - half_h)
    base_br_spatial = depth_to_3d_points_spatial(first_img[half_h:height, half_w:width], fx, fy, cx - half_w, cy - half_h)

    points_3d_tl_spatial = depth_to_3d_points_spatial(points_3d_tl_down, fx, fy, cx_tl, cy_tl)  # (147, 259, 3)
    points_3d_tr_spatial = depth_to_3d_points_spatial(points_3d_tr_down, fx, fy, cx_tr, cy_tr)
    points_3d_bl_spatial = depth_to_3d_points_spatial(points_3d_bl_down, fx, fy, cx_bl, cy_bl)
    points_3d_br_spatial = depth_to_3d_points_spatial(points_3d_br_down, fx, fy, cx_br, cy_br)
    
    def find_median_translation(base_3d, crop_3d):
        diff = base_3d - crop_3d  # (H, W, 3)
        return np.median(diff.reshape(-1, 3), axis=0)  # (3,)

    tl_translation = find_median_translation(base_tl_spatial, points_3d_tl_spatial)
    tr_translation = find_median_translation(base_tr_spatial, points_3d_tr_spatial)
    bl_translation = find_median_translation(base_bl_spatial, points_3d_bl_spatial)
    br_translation = find_median_translation(base_br_spatial, points_3d_br_spatial)

    final_points_3d = depth_to_3d_points_spatial(first_img, fx, fy, cx, cy)  # (294, 518, 3)
        
    quadrants = [
        (points_3d_tl_spatial, conf_tl_down, 0, half_h, 0, half_w, tl_translation),
        (points_3d_tr_spatial, conf_tr_down, 0, half_h, half_w, width, tr_translation),
        (points_3d_bl_spatial, conf_bl_down, half_h, height, 0, half_w, bl_translation),
        (points_3d_br_spatial, conf_br_down, half_h, height, half_w, width, br_translation)
    ]

    for crop_3d_spatial, crop_conf, h1, h2, w1, w2, translation in quadrants:
        
        base_conf = conf_first[h1:h2, w1:w2]  # (147, 259)
        
        valid_mask = (base_conf >= conf_thresh) & (crop_conf >= conf_thresh)
        
        if valid_mask.sum() > 0:
            crop_3d_aligned = crop_3d_spatial + translation[np.newaxis, np.newaxis, :]  # (147, 259, 3)
            final_points_3d[h1:h2, w1:w2, :] = crop_3d_aligned
        else:
            print("no valid regions")
    
    
    final_depth = final_points_3d[:, :, 2]  # (294, 518)
    
    print(f"Final depth shape: {final_depth.shape}")
    print(f"Final 3D points shape: {final_points_3d.shape}")
    
    np.save(output_dir / "predicted_depth.npy", final_depth)
    print(final_points_3d.shape)
    np.save(output_dir / "predicted_points_3d.npy", final_points_3d)
    # np.save(output_dir / "base_depth.npy", first_img)
    
    if Path("temp_quadrants").exists():
        shutil.rmtree(Path("temp_quadrants"))
    print("Finished predictions")

def gt_camera_params(prediction, height, width):
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
    
    with open(output_dir / "gt_camera_params.json", 'w') as f:
        json.dump(camera_params, f, indent=2)

    return fx,fy,cx,cy

def save_camera_params(pose_enc, height, width):
    # print(f"pose {pose_enc.shape}")

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (height, width))
    
    # extrinsic shape: (batch, 3, 4) - compact [R|t] format
    # intrinsic shape: (batch, num_images, 3, 3)

    intrinsic_np = intrinsic.cpu().numpy()[0, 0]  # (3, 3)

    # print("exr")
    # print(extrinsic.shape) (1,3,4)
    # print(extrinsic)
    extrinsic_3x4 = extrinsic.cpu().numpy()[0]  # (3, 4)
    
    # Convert to homogeneous 4x4 matrix
    extrinsic_np = np.eye(4)  # identity matrix so it's a 1 under the t of the matrix
    extrinsic_np[:3, :] = extrinsic_3x4  # Copy [R|t] into top 3 rows
    
    fx = float(intrinsic_np[0, 0])
    fy = float(intrinsic_np[1, 1])
    cx = float(intrinsic_np[0, 2])
    cy = float(intrinsic_np[1, 2])

    camera_params = {
        'intrinsics': {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'intrinsic_matrix': intrinsic_np.tolist()
        },
        'extrinsics': {
            'extrinsic_matrix': extrinsic_np.tolist(),
        },
        'resolution': {
            'height': int(height),
            'width': int(width)
        },
    }
    
    with open(output_dir / "predicted_camera_params.json", 'w') as f:
        json.dump(camera_params, f, indent=2)

    return fx, fy, cx, cy, extrinsic_np

def base_only():
    print("Starting prediction for BASE IMAGE ONLY")
    
    images = load_and_preprocess_images([base_img]).to(device)
    # print(images.shape)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            prediction = model(images)
    # print(prediction)
    depth = prediction['depth'].cpu().numpy()[0, 0, :, :, 0]  # (294, 518)
    depth_conf = prediction['depth_conf'].cpu().numpy()[0, 0]  # (294, 518)
    print(images.shape)
    height, width = depth.shape
    pose_enc = prediction['pose_enc']

    fx, fy, cx, cy, extrinsic = save_camera_params(pose_enc, images.shape[2], images.shape[3])
    
    def depth_to_3d_points_spatial(depth_map, fx, fy, cx, cy):
        h, w = depth_map.shape
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        X = (xs - cx) * depth_map / fx
        Y = (ys - cy) * depth_map / fy
        Z = depth_map
        return np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    
    points_3d_spatial = depth_to_3d_points_spatial(depth, fx, fy, cx, cy)  # (518, 518, 3)
    
    # Load and resize RGB image
    rgb_image = np.array(Image.open(base_img).resize((width, height))) / 255.0
    rgb_image = rgb_image[:, :, :3]  # Ensure only RGB, no alpha
    
    # Flatten to (N, 3) format
    points_3d = points_3d_spatial.reshape(-1, 3)  # (268324, 3)
    colors = rgb_image.reshape(-1, 3)  # (268324, 3)
    
    print(f"Flattened points_3d shape: {points_3d.shape}")
    print(f"Flattened colors shape: {colors.shape}")
    
    np.save(output_dir / "base_only_depth.npy", depth)
    np.save(output_dir / "predicted_points_3d.npy", points_3d)
    np.save(output_dir / "predicted_colors.npy", colors)
    np.save(output_dir / "base_only_conf.npy", depth_conf)
    
    print("Finished base-only predictions")
    return depth, points_3d, depth_conf

def test():
    img = Image.open(base_img)
    img = np.array(img)  # (720, 1280, 3)
    h, w, _ = img.shape
    h_half = h // 2
    w_half = w // 2
    quad_paths = []
    quad_paths.append(base_img)
    # quads = get_quadrants(img, h_half=h_half, w_half=w_half, h=h, w=w)
    # for q in quads:
    #     quad_paths.append(q)
    # print(quad_paths)
    images = load_and_preprocess_images(quad_paths).to(device)
    # print(images.shape)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images)
    
    depth = predictions['depth'].cpu().numpy().squeeze(0)  # (5, 518, 518, 1)
    depth_conf = predictions['depth_conf'].cpu().numpy().squeeze(0)  # (5, 518, 518)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions['pose_enc'],
        (images.shape[2], images.shape[3]),
    )
    world_points = unproject_depth_map_to_point_map(depth, extrinsic[0,:,:,:], intrinsic[0,:,:,:])
    points_3d = np.vstack([world_points[i].reshape(-1, 3) for i in range(world_points.shape[0])])  # (5*H*W, 3)

    np.save(output_dir / "predicted_points_3d.npy", points_3d)
    _, _, height, width = images.shape  # 518, 518
    get_rgb_base_only(height, width)

def five_basic():
    img = Image.open(base_img)
    img = np.array(img)  # (720, 1280, 3)
    h, w, _ = img.shape
    h_half = h // 2
    w_half = w // 2
    quad_paths = []
    quad_paths.append(base_img)
    quads = get_quadrants(img, h_half=h_half, w_half=w_half, h=h, w=w)
    for q in quads:
        quad_paths.append(q)
    print(quad_paths)
    images = load_and_preprocess_images(quad_paths).to(device)
    # print(images.shape)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images)
    
    depth = predictions['depth'].cpu().numpy().squeeze(0)  # (5, 518, 518, 1)
    depth_conf = predictions['depth_conf'].cpu().numpy().squeeze(0)  # (5, 518, 518)
    # print(depth.shape)

    base_depth = depth[0,:,:,0]
    quad_tl = depth[1, :, :, 0]  # (294, 518)
    quad_tr = depth[2, :, :, 0]
    quad_bl = depth[3, :, :, 0]
    quad_br = depth[4, :, :, 0]
    
    base_conf = depth_conf[0]
    conf_tl = depth_conf[1]
    conf_tr = depth_conf[2]
    conf_bl = depth_conf[3]
    conf_br = depth_conf[4]
    
    _, _, height, width = images.shape  # 518, 518
    pose_enc = predictions['pose_enc'] # (1,5,9)
    # print(pose_enc[:, 0, :])
    # print(pose_enc[:, 1, :])

    fx_b, fy_b, cx_b, cy_b, extrinsic_b = save_camera_params(pose_enc[:, 0, :], height, width)
    fx_tl, fy_tl, cx_tl, cy_tl, extrinsic_tl = save_camera_params(pose_enc[:, 1, :], height, width)
    fx_tr, fy_tr, cx_tr, cy_tr, extrinsic_tr = save_camera_params(pose_enc[:, 2, :], height, width)
    fx_bl, fy_bl, cx_bl, cy_bl, extrinsic_bl = save_camera_params(pose_enc[:, 3, :], height, width)
    fx_br, fy_br, cx_br, cy_br, extrinsic_br = save_camera_params(pose_enc[:, 4, :], height, width)

    def depth_to_3d_points_spatial(depth_map, fx, fy, cx, cy):
        h, w = depth_map.shape
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        X = (xs - cx) * depth_map / fx
        Y = (ys - cy) * depth_map / fy
        Z = depth_map
        return np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    
    
    # Convert depth to 3D points in camera space
    points_3d_base_cam = depth_to_3d_points_spatial(base_depth, fx_b, fy_b, cx_b, cy_b)
    points_3d_tl_cam = depth_to_3d_points_spatial(quad_tl, fx_tl, fy_tl, cx_tl, cy_tl)
    points_3d_tr_cam = depth_to_3d_points_spatial(quad_tr, fx_tr, fy_tr, cx_tr, cy_tr)
    points_3d_bl_cam = depth_to_3d_points_spatial(quad_bl, fx_bl, fy_bl, cx_bl, cy_bl)
    points_3d_br_cam = depth_to_3d_points_spatial(quad_br, fx_br, fy_br, cx_br, cy_br)
    
    # Transform to world space
    points_3d_base = transform_to_world(points_3d_base_cam, extrinsic_b)
    points_3d_tl = transform_to_world(points_3d_tl_cam, extrinsic_tl)
    points_3d_tr = transform_to_world(points_3d_tr_cam, extrinsic_tr)
    points_3d_bl = transform_to_world(points_3d_bl_cam, extrinsic_bl)
    points_3d_br = transform_to_world(points_3d_br_cam, extrinsic_br)
    
    
    # Flatten all 5 to (N, 3) format and stack
    points_3d = np.vstack([
        points_3d_base.reshape(-1, 3),
        points_3d_tl.reshape(-1, 3),
        points_3d_tr.reshape(-1, 3),
        points_3d_bl.reshape(-1, 3),
        points_3d_br.reshape(-1, 3)
    ])
    
    
    
    np.save(output_dir / "predicted_points_3d.npy", points_3d)
    
    # if Path("temp_quadrants").exists():
    #     shutil.rmtree(Path("temp_quadrants"))
    
    return depth, depth_conf
def get_rgb_base_only(height, width):
    img = Image.open(base_img).resize((width, height))
    arr = np.array(img, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3]              # (H, W, 3)
    colors = rgb.reshape(-1, 3)      # (H*W, 3)
    np.save(output_dir / "predicted_colors.npy", colors)
    return colors
def get_rgb(height, width):
    quad_dir = Path("temp_quadrants")

    def load_rgb(path):
        img = Image.open(path).resize((width, height))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[:, :, :3]   # ensure RGB only

    # Order must match your predict() frames order
    rgb_base = load_rgb(base_img)
    rgb_tl   = load_rgb(quad_dir / "topleft.png")
    rgb_tr   = load_rgb(quad_dir / "topright.png")
    rgb_bl   = load_rgb(quad_dir / "bot_left.png")
    rgb_br   = load_rgb(quad_dir / "bot_right.png")

    # Stack to (5, H, W, 3)
    colors_5 = np.stack(
        [rgb_base, rgb_tl, rgb_tr, rgb_bl, rgb_br],
        axis=0
    )  # (5, H, W, 3)

    # Flatten to (5*H*W, 3) to match your world_points.reshape(-1, 3)
    colors = colors_5.reshape(-1, 3)

    np.save(output_dir / "predicted_colors.npy", colors)
    return colors


def cam_to_world(points_camera, extrinsic_3x4):
    """
    points_camera: (H, W, 3), in camera coords
    extrinsic_3x4: [R|t] that maps world -> cam: x_cam = R x_world + t
    """
    R = extrinsic_3x4[:, :3]   # (3, 3)
    t = extrinsic_3x4[:, 3]    # (3,)

    R_inv = R.T
    t_inv = -R_inv @ t

    h, w, _ = points_camera.shape
    pts = points_camera.reshape(-1, 3).T   # (3, N)

    pts_world = (R_inv @ pts) + t_inv[:, None]  # (3, N)
    pts_world = pts_world.T.reshape(h, w, 3)    # (H, W, 3)
    return pts_world

if __name__ == "__main__":
    test()
