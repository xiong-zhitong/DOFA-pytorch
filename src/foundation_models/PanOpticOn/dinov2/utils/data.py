import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torch
from typing import List

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def load_ds_cfg(ds_name):
    """ load chn_props and metainfo of dataset from file structure"""
    
    root = os.environ.get('DATA_CONFIG_DIR', 'dinov2/configs/data/') # assumes current working directory in PanOpticOn/

    # get dataset
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != 'satellites']
    ds = {}
    for d in dirs:
        for r, d, f in os.walk(os.path.join(root, d)):
            for file in f:
                if file[-5:] == '.yaml':
                    ds[file.split('.')[0]] = os.path.join(r, file)
    assert ds_name in ds, f'Dataset "{ds_name}" not found at {root} in folders {dirs}'
    ds_cfg = read_yaml(ds[ds_name])

    # get satellites
    sats = {}
    for r,d,f in os.walk(os.path.join(root, 'satellites')):
        for file in f:
            if file[-5:] == '.yaml':
                sats[file.split('.')[0]] = os.path.join(r, file) 

    # build chn_props
    chn_props = []
    sat_cfgs = {}
    for b in ds_cfg['bands']:
        sat_id, band_id = b['id'].split('/')
        if sat_id not in sat_cfgs:
            sat_cfgs[sat_id] = read_yaml(sats[sat_id])
        band_cfg = sat_cfgs[sat_id]['bands'][band_id]
        band_cfg['id'] = b['id']
        chn_props.append(band_cfg)
    metainfo = {k:v for k,v in ds_cfg.items() if k != 'bands'}
    return {'ds_name': ds_name, 'bands': chn_props, 'metainfo': metainfo}

# def extract_wavemus(ds_cfg, return_sigmas=False):
#     if not return_sigmas:
#         return torch.tensor([b['gaussian']['mu'] for b in ds_cfg['bands']], dtype=torch.int32)
#     return 


def extract_wavemus(ds_cfg, return_sigmas=False):
    mus = [b['gaussian']['mu'] for b in ds_cfg['bands']]

    if not return_sigmas:
        return torch.tensor(mus, dtype=torch.float32)
    
    sigmas = [b['gaussian']['sigma'] for b in ds_cfg['bands']]
    return torch.tensor(list(zip(mus, sigmas)), dtype=torch.float32)

def dict_to_device(x_dict, device, keys: List = None, **kwargs):
    if keys is None:
        keys = x_dict.keys()
    for k in keys:
        x_dict[k] = x_dict[k].to(device, **kwargs)
    return x_dict

def plot_ds(*ds_names, log_gsd=True):
    if isinstance(ds_names[0], list):
        ds_names = ds_names[0]

    # load data
    out = []
    for ds_name in ds_names:
        ds_cfg = load_ds_cfg(ds_name)
        vals = [dict(
            ds=ds_name, mu=b['gaussian']['mu'], sat=b['id'].split('/')[0], gsd=b['GSD']) 
            for b in ds_cfg['bands']]
        out += vals
    df = pd.DataFrame(out)
    df['gsd_jitter'] = df['gsd'] + np.random.randn(len(df))*0.1

    # colored by satellite within dataset
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='mu', y='gsd_jitter', hue='ds', ax=ax)
    if log_gsd:
        ax.set_yscale('log')


def getimgsatl(id, dir, sensor, root_dir='/data/panopticon/datasets/satlas'):

    if sensor == 's2':
        bands = ['tci', 'b05', 'b06', 'b07', 'b08', 'b11', 'b12']
        root_dir = os.path.join(root_dir, 'sentinel2')
    elif sensor == 'ls':
        bands = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11']
        root_dir = os.path.join(root_dir, 'landsat')
    else:
        raise NotImplementedError()

    sat_img = {}
    for b in bands:
        # img = read_image(os.path.join(root_dir, dir, b, f'{id}.png'))
        img = Image.open(os.path.join(root_dir, dir, b, f'{id}.png'))
        img = pil_to_tensor(img)
        sat_img[b] = img
    return sat_img

def plot_rgb(img):
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img.permute(1,2,0))
    plt.show()