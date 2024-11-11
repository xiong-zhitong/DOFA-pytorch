import torch
import torch.nn as nn
import logging
from dinov2.layers.attention import QueryCrossAttention
from dinov2.layers.patch_embed import make_2tuple
from dinov2.layers.block import Block, drop_add_residual_stochastic_depth
from typing import Callable, Optional
from torch import Tensor

# inspired by https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/arch.py#L185

logger = logging.getLogger("dinov2")
    
class ChnAttnPatchEmb(nn.Module):
    def __init__(self, 
                 patch_emb_cfg: dict = {}, 
                 chnAttnBlock_cfg: dict = {}, 
                 chnEmb_cfg: dict = {}):
        super().__init__()
        self.patch_emb = ChnWisePatchEmb(**patch_emb_cfg)

        assert 'attn_class' not in chnAttnBlock_cfg
        chnAttnBlock_cfg['attn_class'] = QueryCrossAttention
        id_attn_block = chnAttnBlock_cfg.pop('id_attn_block', 'ChnAttnBlockSimple')
        logger.info(f'ChnAttnPatchEmb: id_attn_block: {id_attn_block}')
        if id_attn_block == 'ChnAttnBlock':
            blk = ChnAttnBlock(**chnAttnBlock_cfg)
        elif id_attn_block == 'ChnAttnBlockSimple':
            blk = ChnAttnBlockSimple(**chnAttnBlock_cfg)
        else:
            raise ValueError(f'Unknown id_attn_block: {id_attn_block}') 
        self.chnattnblock = blk

        self.chnemb = ChnEmb(**chnEmb_cfg)

    def forward(self, x_dict: dict) -> Tensor:
        imgs = x_dict['imgs']
        w,h = imgs.shape[-2:]
        mask = x_dict.get('spec_masks', None)
        
        imgs = self.patch_emb(imgs)

        chn_embs = self.chnemb(x_dict['chn_ids'])
        imgs += chn_embs.unsqueeze(2)

        imgs = self.chnattnblock(imgs, mask=mask)
        return imgs, h, w
    

class ChnAttnBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.attn, QueryCrossAttention) # MemEffQueryCrossAttention is child of QueryCrossAttention
        self.query = nn.Parameter(torch.zeros(1, 1, kwargs['dim']))

    def _forward_block(self, q: Tensor, kv: Tensor, mask=None) -> Tensor:
        # as in Block.forward, only query added in attention as argument
        def attn_residual_func(q: Tensor, kv: Tensor) -> Tensor:
            return self.ls1(self.attn(q, self.norm1(kv), key_padding_mask=mask))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            q = q + attn_residual_func(q, kv)
            q = drop_add_residual_stochastic_depth(
                q,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            q = q + self.drop_path1(attn_residual_func(q, kv))
            q = q + self.drop_path1(ffn_residual_func(q))  # FIXME: drop_path2
        else:
            q = q + attn_residual_func(q, kv)
            q = q + ffn_residual_func(q)
        return q

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # shape of x: (B, C, L, D), shape of mask: (B, C)
        B, C, L, D = x.shape

        x = x.permute(0, 2, 1, 3).flatten(0, 1) # BL,C,D 
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, L, -1).flatten(0, 1) # BL,C
        query = self.query.expand(x.shape[0], 1, -1)

        x = self._forward_block(query, x, mask=mask)
        x = x.reshape(B, L, D)
        return x

class ChnAttnBlockSimple(ChnAttnBlock):

    def __init__(self, *args, norm_input=True, skip_conn=True, norm_output=False, use_layer_scale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_input = norm_input
        self.skip_conn = skip_conn
        self.norm_output = norm_output
        self.use_layer_scale = use_layer_scale
        logger.info(f'ChnAttnBlockSimple: norm_input: {norm_input}, skip_conn: {skip_conn}, norm_output: {norm_output}, use_layer_scale: {use_layer_scale}')

    def _forward_block(self, q: Tensor, kv: Tensor, mask=None) -> Tensor:
        if self.sample_drop_ratio > 0:
            raise NotImplementedError('Simple block does not support stochastic depth')
        def attn_residual_func(q: Tensor, kv: Tensor) -> Tensor:
            q = self.attn(q, kv, key_padding_mask=mask)
            if self.use_layer_scale:
                q = self.ls1(q)
            return q

        if self.norm_input:
            kv = self.norm1(kv)

        if self.skip_conn:
            q = q + attn_residual_func(q, kv)
        else:
            q = attn_residual_func(q, kv)

        if self.norm_output:
            q = self.norm2(q)

        return q

class ChnWisePatchEmb(nn.Module):
    def __init__(
            self, 
            patch_size: int,
            embed_dim: int, 
            norm_layer: Optional[Callable] = None,):
        super().__init__()
        self.patch_size = make_2tuple(patch_size)
        patch_CHW = (1, *self.patch_size)
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_CHW, stride=patch_CHW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """ (B, C, H, W) -> (B, C, L, D)"""
        B, C, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
        
        x = self.proj(x.unsqueeze(1)).squeeze(1) # B D C Hp Wp
        x = x.flatten(-2).permute(0, 2, 3, 1) # B C L D
        x = self.norm(x)

        return x
    
class ChnEmb(torch.nn.Module):
    """ cpu sequential mapping, then gpu torch.index_select. Might not be most efficient,
        but works for now. """
    def __init__(self, embed_dim: int, n_learnable_embs: int = 12, mean_sar_both=False):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.n_cached_optical_embs = 0
        self.register_buffer('cached_optical_embs', torch.zeros(0, self.embed_dim, requires_grad=False).float())

        self.n_learnable_embs = n_learnable_embs
        self.learnable_embs = nn.Parameter(torch.zeros(n_learnable_embs, embed_dim))
        self.mu_map = {-(i+1): i for i in range(n_learnable_embs)} # pre-allocate learnable embs
        self.mean_sar_both = mean_sar_both

    def _fun(self,x):
        x = int(x) #use only the integer part to map
        with torch.no_grad():
            if x not in self.mu_map:
                # assert x >= self.n_learnable_embs, f'x: {x}, n_learnable_embs: {self.n_learnable_embs}'
                device = self.learnable_embs.device
                self.cached_optical_embs = torch.cat([
                    self.cached_optical_embs.to(device), 
                    self._get_emb(x, device)])
                self.mu_map[x] = self.n_learnable_embs + self.n_cached_optical_embs
                self.n_cached_optical_embs += 1
        return self.mu_map[x]


    def apply_fun2(self, tensor):
        """
        Fully vectorized implementation that applies the IPE to the input tensor containing (mu, sigma) pairs
        
        Args:
            tensor: Input tensor of shape (B, C, 2) where:
                - B is the batch size
                - C is the number of channels
                - Last dimension contains (mu, sigma) pairs
        
        Returns:
            Tensor of shape (B, C) containing the mapped indices
        """

        device = self.learnable_embs.device
        # tensor = tensor.to(device)
            
        mus = tensor[..., 0].long()     # Shape: (B, C)
        sigmas = tensor[..., 1].long()  # Shape: (B, C)

        with torch.no_grad(): #none of these ops need to be tracked in the computation graph
        
            # Get unique (mu, sigma) pairs across all batches
            # Flatten batch and channel dimensions for unique operation
            mu_sigma_pairs = torch.stack([mus, sigmas], dim=-1).view(-1, 2)  # Shape: (B*C, 2)
            unique_pairs, _ = torch.unique(mu_sigma_pairs, dim=0, return_inverse=True) # non-differentiable, but not in gradient flow path
            
            # Filter out pairs that are already in the map
            new_pairs_mask = torch.tensor([mu.item() not in self.mu_map for mu in unique_pairs[:, 0]], device=device)
            new_pairs = unique_pairs[new_pairs_mask]
            
            if len(new_pairs) > 0:
                # Generate embeddings for all new pairs at once
                new_mus = new_pairs[:, 0]
                new_sigmas = new_pairs[:, 1]
                assert len(new_mus) == len(new_sigmas), f'ERROR: len(new_mus): {len(new_mus)} NOT== len(new_sigmas): {len(new_sigmas)}'
                new_embeddings = self._get_emb((new_mus, new_sigmas), device)
                
                # Add new embeddings and update map
                self.cached_optical_embs = torch.cat([
                    self.cached_optical_embs.to(device),
                    new_embeddings
                ])
                
                # Update mapping for all new pairs
                for idx, (mu, _) in enumerate(new_pairs.tolist()):
                    self.mu_map[mu] = self.n_learnable_embs + self.n_cached_optical_embs + idx
                
                self.n_cached_optical_embs += len(new_pairs)
            
            # Vectorized mapping
            mapped_indices = torch.tensor([self.mu_map[mu.item()] for mu in mus.view(-1)], dtype=torch.long, device=device)
            mapped_indices = mapped_indices.view(mus.shape)
        
        return mapped_indices
        

    def _get_emb(self, x, device):
        #check if x is a tuple
        if not isinstance(x, tuple):
            return self.get_1d_sincos_pos_embed_from_grid_torch(self.embed_dim, torch.tensor([x], device=device))
        else:
            mus, sigmas = x  # Now these can be vectors
            return self.get_1d_sincos_ipe(mu=mus, sigma=sigmas, D=self.embed_dim, device=device)
        
    def get_1d_sincos_pos_embed_from_grid_torch(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out) # (M, D/2)
        emb_cos = torch.cos(out) # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb #.double() -> unsure why the authors wanted to cast to double??

    def get_1d_sincos_ipe(self, mu: Tensor, sigma: Tensor, D:int, device, temperature=10000, N=2101, range_sigma=3):
        """
        mu: (M,) tensor of means, or central wavelengths
        sigma: (M,) tensor of standard deviations, or bandwidths
        temperature: scalar, temperature for the denominator = 10000
        D: scalar, output dimension for each position, embedding dimension
        N: scalar, number of samples for the integral = 2101
        kernel: function, kernel function for the integral = gaussian_kernel
        range_sigma: scalar, +- standard deviations to model for the kernel
        
        """
        
        # Create meshgrid for vectorized computation
        d_mesh = torch.arange(D, dtype=torch.float32, device=device)
        mu_mesh = mu.unsqueeze(1).expand(-1, D)
        sigma_mesh = sigma.unsqueeze(1).expand(-1, D)
        
        # Compute denominator
        DENOMINATOR = temperature ** (2 * d_mesh / D)
        
        # Create lambda_j tensor
        lambda_j = torch.linspace(-range_sigma, range_sigma, N, dtype=torch.float32, device=device)
        lambda_j = (lambda_j.unsqueeze(0).unsqueeze(0) * sigma_mesh.unsqueeze(-1) + mu_mesh.unsqueeze(-1))

        def gaussian_kernel(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        
        # Compute kernel values
        g_j = gaussian_kernel(lambda_j, mu_mesh.unsqueeze(-1), sigma_mesh.unsqueeze(-1))
        
        # Compute sum for even and odd dimensions
        sum_even = torch.sum(torch.sin(lambda_j / DENOMINATOR.unsqueeze(0).unsqueeze(-1)) * g_j, dim=-1)
        sum_odd = torch.sum(torch.cos(lambda_j / DENOMINATOR.unsqueeze(0).unsqueeze(-1)) * g_j, dim=-1)
        
        # Combine even and odd sums
        sum_total = torch.where(d_mesh % 2 == 0, sum_even, sum_odd)
        
        # Normalize by total kernel values
        g_total = torch.sum(g_j, dim=-1)
        IPE = sum_total / g_total

        return IPE

    def forward(self, input: Tensor) -> Tensor:
        
        if input.ndim == 2: # B,C (mus)
            device = input.device
            input = input.cpu() #because _apply on works on cpu
            input.apply_(self._fun)
            input = input.to(device).long()
        elif input.ndim == 3: # B,C,2 : B, mus, sigmas
            input = self.apply_fun2(input)
        
        if not self.mean_sar_both:
            weights = torch.cat([self.learnable_embs, self.cached_optical_embs])
        else:
            torch.cat([
                self.learnable_embs.view(3,4,-1)[1:].mean(0),
                self.learnable_embs[4:], 
                self.cached_optical_embs])
        return torch.nn.functional.embedding(input, weights)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict:dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict.pop('cached_optical_embs', None)
        return state_dict

    def load_state_dict(self, state_dict, strict = True):
        state_dict.pop('cached_optical_embs', None)
        return super().load_state_dict(state_dict, strict, strict=False)

    