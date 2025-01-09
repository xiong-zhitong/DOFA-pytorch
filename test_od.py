import torch
from mmdet.models.dense_heads import YOLOV3Head
from mmdet.models.necks import YOLOV3Neck
from einops import rearrange
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmengine.config import Config
from mmdet.models.utils.misc import samplelist_boxtype2tensor


# Define a custom encoder (ViT as an example)
class DinoV2Encoder(torch.nn.Module):
    def __init__(self):
        super(DinoV2Encoder, self).__init__()
        # Example: Use Vision Transformer from MMClassification
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

    def forward(self, x):
        # Output single-scale features
        outputs = self.backbone.get_intermediate_layers(x, [6, 10, 11])
        features = [
            rearrange(out, "n (h w) c -> n c h w", h=int(out.size(1) ** 0.5))
            for out in outputs
        ]
        ms_outs = []
        ms_outs.append(F.interpolate(features[0], scale_factor=2))
        ms_outs.append(features[1])
        ms_outs.append(F.interpolate(features[2], scale_factor=0.5))
        return ms_outs


# Combine Custom Encoder with YOLOV3 Neck and Head
class YOLOModel(torch.nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOModel, self).__init__()
        # Initialize the backbone, neck, and head
        self.backbone = DinoV2Encoder()
        self.neck = YOLOV3Neck(
            in_channels=[1024, 1024, 1024],  # Input channels from encoder
            out_channels=[256, 512, 1024],
            num_scales=3,
        )
        self.bbox_head = YOLOV3Head(
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            out_channels=[256, 512, 1024],
            anchor_generator=dict(
                type="YOLOAnchorGenerator",
                base_sizes=[
                    [(116, 90), (156, 198), (373, 326)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)],
                ],
                strides=[32, 16, 8],
            ),
            train_cfg=Config(
                dict(
                    assigner=dict(
                        type="GridAssigner",
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0,
                    )
                )
            ),
            test_cfg=Config(
                dict(
                    nms_pre=1000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    conf_thr=0.005,
                    nms=dict(type="nms", iou_threshold=0.45),
                    max_per_img=100,
                )
            ),
            bbox_coder=dict(type="YOLOBBoxCoder"),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_conf=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_xy=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=2.0),
            loss_wh=dict(type="MSELoss", loss_weight=2.0),
        )

    def forward(self, x, gt_instances=None, img_metas=None):
        # Encoder generates feature maps
        outputs = self.backbone(x)
        # Neck processes multi-scale feature maps
        neck_features = self.neck(outputs)  # Repeat for multi-scale
        # Head predicts detections

        (pred_maps,) = self.bbox_head(neck_features)

        # predict: backbone->neck->bbox_head->predict_by_feat
        results_list = self.bbox_head.predict_by_feat(
            pred_maps, gt_instances, rescale=True
        )

        # loss: backbone->neck->bbox_head->loss_by_feat
        empty_gt_losses = self.bbox_head.loss_by_feat(
            pred_maps, gt_instances, img_metas
        )

        print(empty_gt_losses)
        return pred_maps

    # Copied from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/base.py#L130
    def add_pred_to_datasample(
        self, data_samples: SampleList, results_list: InstanceList
    ) -> SampleList:
        """Add predictions to `DetDataSample`."""
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples


# Initialize the model
num_classes = 80  # For COCO dataset
model = YOLOModel(num_classes=num_classes)

# Test with dummy data
input_tensor = torch.randn(1, 3, 616, 616)  # Batch size 2, image size 616x616

gt_instances = InstanceData()
gt_instances.bboxes = torch.tensor(
    [
        [0.4997, 0.7707, 0.0595, 0.4188],
        [0.8101, 0.3105, 0.5123, 0.6263],
        [0.4997, 0.7707, 0.0595, 0.4188],
        [0.8101, 0.3105, 0.5123, 0.6263],
    ]
)
gt_instances.labels = torch.tensor([2, 3, 2, 3])

img_metas = [
    {"img_shape": (616, 616, 3), "scale_factor": 1.0, "pad_shape": (616, 616, 3)}
]

model(input_tensor, [gt_instances], img_metas)
