import torch.nn as nn
import torch
from functools import partial

def backbone_to_features(x_list, use_n_blocks, pooling, norm:nn.Module=None):
    x_list = x_list[-use_n_blocks:] # list of output tensors of last n blocks

    if pooling == 'avgpool': # corresponds to DINOv2 avgpool=True
        x_list = [norm(out) for out in x_list]
        class_tokens = [out[:, 0] for out in x_list]

        output = torch.cat(
            (
                torch.cat(class_tokens, dim=-1),
                torch.mean(x_list[-1][:, 1:], dim=1),
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)

    elif pooling == 'cls': # corresponds to DINOv2 avgpool=False
        x_list = [norm(out) for out in x_list]
        class_tokens = [out[:, 0] for out in x_list]
        output = torch.cat(class_tokens, dim=-1)

    elif pooling == 'DOFA_globalpool':
        output = x_list[-1][:, 1:].mean(dim=1)
        output = norm(output) 

    elif pooling == 'DOFA_no_globalpool':
        output = x_list[-1][:, 0]
        output = norm(output.unsqueeze(0)).squeeze(0)

    elif pooling == 'knn': # consistent with vanilla DINOv2
        output = norm(x_list[-1])[:,0]
        output = nn.functional.normalize(output, dim=1, p=2)

    else:
        raise ValueError(f"Pooling {pooling} not supported")

    return output.float()


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model: nn.Module, n_last_blocks, autocast_dtype=None, bb_to_feat_adapter=None):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_dtype = autocast_dtype
        self.return_class_token = True
        self.reshape = False # probably only needed for pixel level tasks
        self.norm = False # norm is handeled in create_linear_input in LinearClassifier
        self.bb_to_feat_adapter = bb_to_feat_adapter
        for p in feature_model.parameters():
            p.requires_grad = False

    def get_intermediate_layers(self, x_dict):
        raise NotImplementedError()

    def forward(self, x_dict):

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.autocast_dtype is not None, dtype=self.autocast_dtype):
                features = self.get_intermediate_layers(x_dict)
                if self.bb_to_feat_adapter is not None:
                    features = self.bb_to_feat_adapter(features, norm=self.norm)
        return features


class DINOv2Wrapper(ModelWithIntermediateLayers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = self.feature_model.norm

    def get_intermediate_layers(self, x_dict):
        # just copied from dinov2/models/vision_transformer.DinoVisionTransformer.get_intermediate_layers
        if self.feature_model.chunked_blocks:
            outputs = self.feature_model._get_intermediate_layers_chunked(x_dict, self.n_last_blocks)
        else:
            outputs = self.feature_model._get_intermediate_layers_not_chunked(x_dict, self.n_last_blocks)
        return outputs
        # return self.feature_model.get_intermediate_layers(
        #     x_dict, self.n_last_blocks, return_class_token=self.return_class_token, reshape=self.reshape)


class DOFAWrapper(ModelWithIntermediateLayers):
    """ very close to https://github.com/zhu-xlab/DOFA/blob/master/downstream/segmentation/models/dofa_vit.py,
        img_size is 224
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = self.feature_model.norm

        self.out_indices = range(-self.n_last_blocks, 0)
        for idx in self.out_indices:
            self.feature_model.blocks[idx].register_forward_hook(
                lambda m, i, o: self._cache_block(o))

    def _cache_block(self,x):
        self.cache.append(x)

    def get_intermediate_layers(self, x_dict):
        self.cache = []
        imgs = x_dict['imgs']
        waves = x_dict['chn_ids'][0].float() / 1e3 # assume that all images have the same wave_list
        self.feature_model.forward_features_nopool(imgs, waves)
        output = self.cache
        self.cache = [] # remove from model
        return output