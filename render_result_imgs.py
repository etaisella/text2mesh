import clip
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
import torch
from neural_style_field import NeuralStyleField
from utils import device 
from render import Renderer
from mesh import Mesh
from Normalization import MeshNormalizer
import numpy as np
import random
import copy
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms

def run_branched(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #mesh_path = Path(args.obj_path)
    color_path = Path(args.color_path)
    mesh, colors = import_results(args.obj_path, color_path)

    render = Renderer()

    rendered_images, elev, azim = render.render_front_views(mesh, 
                                                            num_views=args.n_views,
                                                            show=args.show,
                                                            center_azim=args.frontview_center[0],
                                                            center_elev=args.frontview_center[1],
                                                            std=args.frontview_std,
                                                            return_views=True,
                                                            background=torch.tensor([1, 1, 1]).to(device).float())
    
    for i in range(rendered_images.shape[0]):
        img_t = rendered_images[i]
        img = img_t.cpu()
        #mask = masks[i].cpu()
        ## Manually add alpha channel using background color
        #alpha = torch.ones(img.shape[1], img.shape[2])
        #alpha[torch.where(mask == 0)] = 0
        #img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(args.output_dir, f"{i}.png"))

    print("done")
    



def import_results(mesh_path, color_path):
    mesh = Mesh(mesh_path)
    colors = torch.load(color_path)

    return mesh, colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='meshes/mesh1.obj')
    parser.add_argument('--color_path', type=str, default='meshes/mesh1.pt')
    parser.add_argument('--output_dir', type=str, default='round2/alpha5')
    parser.add_argument('--sigma', type=float, default=10.0)
    parser.add_argument('--normsigma', type=float, default=10.0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--colordepth', type=int, default=2)
    parser.add_argument('--normdepth', type=int, default=2)
    parser.add_argument('--normwidth', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--normal_learning_rate', type=float, default=0.0005)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--no_pe', dest='pe', default=True, action='store_false')
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--n_views', type=int, default=5)
    parser.add_argument('--n_augs', type=int, default=0)
    parser.add_argument('--n_normaugs', type=int, default=0)
    parser.add_argument('--n_iter', type=int, default=6000)
    parser.add_argument('--encoding', type=str, default='gaussian')
    parser.add_argument('--normencoding', type=str, default='xyz')
    parser.add_argument('--layernorm', action="store_true")
    parser.add_argument('--run', type=str, default=None)
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--clamp', type=str, default="tanh")
    parser.add_argument('--normclamp', type=str, default="tanh")
    parser.add_argument('--normratio', type=float, default=0.1)
    parser.add_argument('--frontview', action='store_true')
    parser.add_argument('--no_prompt', default=False, action='store_true')
    parser.add_argument('--exclude', type=int, default=0)

    # Training settings 
    parser.add_argument('--frontview_std', type=float, default=8)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[5.4, 0.4])
    parser.add_argument('--clipavg', type=str, default=None)
    parser.add_argument('--geoloss', action="store_true")
    parser.add_argument('--samplebary', action="store_true")
    parser.add_argument('--promptviews', nargs="+", default=None)
    parser.add_argument('--mincrop', type=float, default=1)
    parser.add_argument('--maxcrop', type=float, default=1)
    parser.add_argument('--normmincrop', type=float, default=0.1)
    parser.add_argument('--normmaxcrop', type=float, default=0.1)
    parser.add_argument('--splitnormloss', action="store_true")
    parser.add_argument('--splitcolorloss', action="store_true")
    parser.add_argument("--nonorm", action="store_true")
    parser.add_argument('--cropsteps', type=int, default=0)
    parser.add_argument('--cropforward', action='store_true')
    parser.add_argument('--cropdecay', type=float, default=1.0)
    parser.add_argument('--decayfreq', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--background', nargs=3, type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_render', action="store_true")
    parser.add_argument('--input_normals', default=False, action='store_true')
    parser.add_argument('--symmetry', default=False, action='store_true')
    parser.add_argument('--only_z', default=False, action='store_true')
    parser.add_argument('--standardize', default=False, action='store_true')

    # CLIP model settings 
    parser.add_argument('--clipmodel', type=str, default='ViT-B/32')
    parser.add_argument('--jit', action="store_true")
    
    args = parser.parse_args()

    run_branched(args)
