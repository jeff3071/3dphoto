import argparse
import copy
import os
import time
from functools import partial
import subprocess
import shlex
import cv2
import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm
from PIL import Image

import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering
from boostmonodepth_utils import run_boostmonodepth
from mesh import output_3d_photo, read_ply, write_ply
from MiDaS.monodepth_net import MonoDepthNet
from MiDaS.run import run_depth, run_depth_1
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from utils import get_MiDaS_samples, read_MiDaS_depth
from utils import path_planning
from fastapi import FastAPI, File, UploadFile
import io
import pydantic
import vispy
import numpy
from skimage.transform import resize

#cmd = "Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset"
#subprocess.Popen(shlex.split(cmd))
#os.environ["DISPLAY"] = ":1"

vispy.use(app='egl')

def load_inpaint_model(config, device):
  print(f"Loading edge model")
  depth_edge_model = Inpaint_Edge_Net(init_weights=True)
  depth_edge_weight = torch.load(config["depth_edge_model_ckpt"], map_location=torch.device(device))
  depth_edge_model.load_state_dict(depth_edge_weight)
  depth_edge_model = depth_edge_model.to(device)
  depth_edge_model.eval()

  print(f"Loading depth model")
  depth_feat_model = Inpaint_Depth_Net()
  depth_feat_weight = torch.load(config["depth_feat_model_ckpt"], map_location=torch.device(device))
  depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
  depth_feat_model = depth_feat_model.to(device)
  depth_feat_model.eval()
  depth_feat_model = depth_feat_model.to(device)
  print(f"Loading rgb model")
  rgb_model = Inpaint_Color_Net()
  rgb_feat_weight = torch.load(config["rgb_feat_model_ckpt"], map_location=torch.device(device))
  rgb_model.load_state_dict(rgb_feat_weight)
  rgb_model.eval()
  rgb_model = rgb_model.to(device)
  
  return depth_edge_model, depth_feat_model, rgb_model


def predict(img, effect='circle'):
  
  img = numpy.array(img)
  image = img[:, :, :].copy()
  
  print("Running 3D Photo Inpainting .. ")
  config = yaml.safe_load(open("argument.yml", "r"))
  config["offscreen_rendering"] = True
  os.makedirs(config["depth_folder"], exist_ok=True)
  os.makedirs(config["mesh_folder"], exist_ok=True)
  os.makedirs(config["video_folder"], exist_ok=True)
  
  traj_types_dict = {"dolly-zoom-in": "double-straight-line",
                           'zoom-in': 'double-straight-line',
                           'circle': 'circle',
                           'swing': 'circle'}

  shift_range_dict = {"circle": [[-0.015], [-0.015], [-0.05]],
                      "swing": [[-0.015], [-0.00], [-0.05]]}

  config["traj_types"] = [traj_types_dict[effect]]
  config["x_shift_range"], config["y_shift_range"], config["z_shift_range"] = shift_range_dict[effect]
  
  print(f"Running depth extraction")

  run_depth_1(
    img,
    config["depth_folder"],
    config["MiDaS_model_ckpt"],
    MonoDepthNet,
    MiDaS_utils,
    target_w=640,
  )
  
  #load deapth, path: config["depth_folder"]/image.npy
  depth_path = config["depth_folder"] + '/image.npy'
  config["output_h"], config["output_w"] = np.load(depth_path).shape[:2]

  frac = config["longer_side_len"] / max(config["output_h"], config["output_w"])
  config["output_h"], config["output_w"] = int(config["output_h"] * frac), int(config["output_w"] * frac)
  config["original_h"], config["original_w"] = config["output_h"], config["output_w"]

  image = cv2.resize(image, (config["output_w"], config["output_h"]), interpolation=cv2.INTER_AREA)
  depth = read_MiDaS_depth(depth_path, 3.0, config["output_h"], config["output_w"])

  mean_loc_depth = depth[depth.shape[0] // 2, depth.shape[1] // 2]

  vis_photos, vis_depths = sparse_bilateral_filtering(
      depth.copy(), image.copy(), config, num_iter=config["sparse_iter"], spdb=False
  )
  depth = vis_depths[-1]
  model = None
  torch.cuda.empty_cache()
  
  print("Start Running 3D_Photo ...")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"running on device {device}")
  
  depth_edge_model, depth_feat_model, rgb_model = load_inpaint_model(config, device)
  graph = None

  print(f"Writing depth ply (and basically doing everything)")
  H, W = img.shape[:2]
  int_mtx = np.array([[max(H, W), 0, W//2], [0, max(H, W), H//2], [0, 0, 1]]).astype(np.float32)
  if int_mtx.max() > 1:
    int_mtx[0, :] = int_mtx[0, :] / float(W)
    int_mtx[1, :] = int_mtx[1, :] / float(H)
  
  mesh_fi = os.path.join(config["mesh_folder"], "image.ply")
  
  rt_info = write_ply(
      image,
      depth,
      int_mtx,
      mesh_fi,
      config,
      rgb_model,
      depth_edge_model,
      depth_edge_model,
      depth_feat_model,
  )

  rgb_model = None
  color_feat_model = None
  depth_edge_model = None
  depth_feat_model = None
  torch.cuda.empty_cache()
  
  verts, colors, faces, Height, Width, hFov, vFov = rt_info

  print(f"Making video at {time.time()}")
  top = config.get("original_h") // 2 - int_mtx[1, 2] * config["output_h"]
  left = config.get("original_w") // 2 - int_mtx[0, 2] * config["output_w"]
  down, right = top + config["output_h"], left + config["output_w"]
  border = [int(xx) for xx in [top, down, left, right]]

  generic_pose = np.eye(4)
  tgt_pose = [[generic_pose * 1]]
  tgts_poses = []
  for traj_idx in range(len(config['traj_types'])):
      tgt_poses = []
      sx, sy, sz = path_planning(config['num_frames'], config['x_shift_range'][traj_idx], config['y_shift_range'][traj_idx],
                                  config['z_shift_range'][traj_idx], path_type=config['traj_types'][traj_idx])
      for xx, yy, zz in zip(sx, sy, sz):
          tgt_poses.append(generic_pose * 1.)
          tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
      tgts_poses += [tgt_poses]    
  tgt_pose = generic_pose * 1
  videos_poses, video_basename = copy.deepcopy(tgts_poses), 'image'
  config["video_postfix"] = [effect]
  
  normal_canvas, all_canvas = None, None
  normal_canvas, all_canvas, output_path = output_3d_photo(
      verts.copy(),
      colors.copy(),
      faces.copy(),
      copy.deepcopy(Height),
      copy.deepcopy(Width),
      copy.deepcopy(hFov),
      copy.deepcopy(vFov),
      copy.deepcopy(tgt_pose),
      config['video_postfix'],
      copy.deepcopy(generic_pose),
      copy.deepcopy(config["video_folder"]),
      image.copy(),
      copy.deepcopy(int_mtx),
      config,
      image,
      videos_poses,
      video_basename,
      config.get("original_h"),
      config.get("original_w"),
      border=border,
      depth=depth,
      normal_canvas=normal_canvas,
      all_canvas=all_canvas,
      mean_loc_depth=mean_loc_depth,
  )

  print(f'Done. Saving to output path: {output_path}')
  return str(output_path)
  

app = FastAPI()

from fastapi.responses import FileResponse

@app.post("/uploadimage/", response_class=FileResponse)
async def create_upload_image(image: UploadFile = File(...)):
    image_bytes = image.file.read()
    img_array = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    out = predict(img_array)
        
    return out

if __name__ == "__main__":
  start_time = time.time()
  print(f'start time: {start_time}')
  img = Image.open('image/image_8.jpg')
  out = predict(img)
  print(f'end time: {time.time()}')
  print(f'use time: {time.time() - start_time}')
