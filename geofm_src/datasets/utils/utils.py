import torch
import numpy as np
from typing import List, Tuple
import os, yaml

# ------------------------------------------------------------------------------
# Hyperspectral Eval Augmentations
# ------------------------------------------------------------------------------

class ChannelSampler(torch.nn.Module):
    """
    Sample channels from a hyperspectral image.
    """
    def __init__(self, band_ids: List[int]):
        super().__init__()
        self.band_ids = band_ids


    def __call__(self, x):
        return x[self.band_ids]



class ChannelSimulator(torch.nn.Module):
    def __init__(self,
                 source_chn_ids,
                 target_chn_ids,
                 wavelength_grid_size=1000, 
                 ):
        """
        Initialize the SpectralConvolutionTransform.

        Parameters:
        - source_chn_ids: Source channel parameters (means, stds) shape (C, 2)
        - wavelength_grid_size: Number of steps in the wavelength grid for numerical integration.
        """
        super().__init__()
        self.wavelength_grid_size = wavelength_grid_size
        self.source_chn_ids = source_chn_ids
        self.target_chn_ids = target_chn_ids

    def _gaussian_srf(self, wavelength, mean, std):
        """Calculate Gaussian SRF for a given wavelength."""
        return torch.exp(-0.5 * ((wavelength - mean) / std) ** 2)


    def spectral_convolution(self,
                        source_img,
                        target_srf_mean,
                        target_srf_std,
                        wavelength_grid_size=1000):
        """
        Vectorized spectral convolution on a hyperspectral image using individual SRFs for each source channel.
        Produces a single channel image.

        Parameters:
        - source_img: Hyperspectral image with shape (C, H, W)
        - target_srf_mean: Mean wavelength of the target SRF
        - target_srf_std: Standard deviation of the target SRF
        - wavelength_grid_size: Number of points in wavelength grid

        Returns:
        - R: The resulting single-channel image with shape (H, W)
        """
        source_srf_means, source_srf_stds = self.source_chn_ids[:, 0], self.source_chn_ids[:, 1]

        # Set the integration range to ±3 stddevs from the target mean
        lambda_min = target_srf_mean - 3 * target_srf_std
        lambda_max = target_srf_mean + 3 * target_srf_std
        target_srf_mean = torch.tensor(target_srf_mean, device=source_img.device)
        target_srf_std = torch.tensor(target_srf_std, device=source_img.device)

        # Identify channels where the source SRF overlaps with the target SRF
        overlap_mask = ((source_srf_means + 3 * source_srf_stds) >= lambda_min) & \
                    ((source_srf_means - 3 * source_srf_stds) <= lambda_max)
        relevant_channels = torch.nonzero(overlap_mask).squeeze()

        # Define wavelength grid for integration
        wavelength_grid = torch.linspace(lambda_min, lambda_max, steps=wavelength_grid_size, 
                                    device=source_img.device)  # Shape: (W_grid,)

        # Calculate the target SRF values over the wavelength grid
        target_srf_values = self._gaussian_srf(wavelength_grid, target_srf_mean, target_srf_std)  # Shape: (W_grid,)

        # Calculate normalization factor
        combined_norm_factor = torch.trapz(target_srf_values, wavelength_grid)
        

        # Vectorized processing of all relevant channels
        # Calculate source SRFs for all relevant channels at once
        wavelength_grid_expanded = wavelength_grid[:, None]  # Shape: (W_grid, 1)
        source_srf_values = self._gaussian_srf(
            wavelength_grid_expanded,
            source_srf_means[relevant_channels],  # Shape: (n_relevant,)
            source_srf_stds[relevant_channels]    # Shape: (n_relevant,)
        )  # Shape: (W_grid, n_relevant)
        
        # Apply threshold
        source_srf_values = torch.where(source_srf_values > 1e-5, source_srf_values, 
                                    torch.zeros_like(source_srf_values))
        
        # Calculate combined SRF values for all channels at once
        combined_srf_values = source_srf_values * target_srf_values.unsqueeze(1)  # Shape: (W_grid, n_relevant)
        
        # Integrate for all channels simultaneously
        combined_srf_integrals = torch.trapz(combined_srf_values, wavelength_grid, dim=0)  # Shape: (n_relevant,)
        
        # Final multiplication and sum across channels
        R = (source_img[relevant_channels] * combined_srf_integrals.view(-1, 1, 1)).sum(dim=0)
        
        # Normalize
        R /= combined_norm_factor
        return R

    def _get_indices(self, data_obj):
        chn_indices = [(s,i) for s in range(len(data_obj['imgs'])) 
                    for i in range(len(data_obj['imgs'][s]))]
        return chn_indices

    def __call__(self, source_img):
        """
        source_img - tensor of shape (C, H, W): HS image
        target_ch_ids - tensor of shape (num_chns, 2) with the mean wavelength and standard deviation of each target channel SRF.
        """

        device = source_img.device

        if isinstance(self.target_chn_ids, torch.Tensor):
            target_srf_mean, target_srf_std = self.target_chn_ids[:,0], self.target_chn_ids[:,1]
        elif isinstance(self.target_chn_ids, list):
            target_srf_mean, target_srf_std = zip(*self.target_chn_ids)
            target_srf_mean, target_srf_std = torch.tensor(target_srf_mean, device=device).float(), torch.tensor(target_srf_std, device=device).float()
            # print(f'target_srf_mean: {target_srf_mean}, target_srf_std: {target_srf_std}')

        out_chns, out_chn_ids = [], []

        for mu, sigma in zip(target_srf_mean, target_srf_std):
            # print(f'Sampling target SRF: mean={target_srf_mean[i]}, std={target_srf_std[i]}')
            out_chns.append(self.spectral_convolution(
                            source_img=source_img,
                            target_srf_mean=mu,
                            target_srf_std=sigma,
                            wavelength_grid_size=self.wavelength_grid_size
                            ))
        #stack to tensor
        out_img = torch.stack(out_chns, dim=0)
        return out_img #, out_chn_ids
    
class MaskTensor(torch.nn.Module): #for HyperviewDataset
    def __init__(self, mask: bool = True, mask_value: float = 0.):
        super().__init__()
        self.mask = mask
        self.mask_value = mask_value

    def __call__(self, sample):
        """
        sample: (x, mask)
        x: C, H, W
        mask: H, W
        output: C, H, W with masked values set to 0
        """
        x, mask = sample
        #set the masked values to 0
        if self.mask:
            return torch.masked_fill(x, mask, self.mask_value)
        else:
            return x



def read_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def load_ds_cfg(ds_name):
    """ load chn_props and metainfo of dataset from file structure"""
    
    root = os.environ.get('REPO_PATH', 'dinov2/configs/') # assumes current working directory in PanOpticOn/
    root = os.path.join(root, 'src/configs/sensor')

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


def extract_wavemus(ds_cfg, return_sigmas=False):
    mus = [b['gaussian']['mu'] for b in ds_cfg['bands']]

    if not return_sigmas:
        return np.array(mus, dtype=np.int16)
    
    sigmas = [b['gaussian']['sigma'] for b in ds_cfg['bands']]
    return np.array(list(zip(mus, sigmas)), dtype=np.int16)
