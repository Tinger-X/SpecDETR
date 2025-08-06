import torch
from torch import nn, Tensor

from components.backbone import *
from components.bbox_head import *
from components.decoder import *
from components.encoder import *
from components.neck import *
from components.positional_encoding import *


# class BaseDataPreprocessor(nn.Module):
#     def __init__(self, non_blocking: Optional[bool] = False):
#         super(BaseDataPreprocessor, self).__init__()
#         self._non_blocking = non_blocking
#
#
# class HSIImgDataPreprocessor(BaseDataPreprocessor):
#     def __init__(
#             self,
#             mean: Optional[Sequence[Union[float, int]]] = None,
#             std: Optional[Sequence[Union[float, int]]] = None,
#             pad_size_divisor: int = 1,
#             pad_value: Union[float, int] = 0,
#             non_blocking: Optional[bool] = False
#     ):
#         super(HSIImgDataPreprocessor, self).__init__(non_blocking)
#         assert (mean is None) == (std is None), "mean and std should be both None or tuple"
#         if mean is not None:
#             self._enable_normalize = True
#             self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
#             self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
#         else:
#             self._enable_normalize = False
#         self.pad_size_divisor = pad_size_divisor
#         self.pad_value = pad_value
#
#
# class HSIDetDataPreprocessor(HSIImgDataPreprocessor):
#     def __init__(
#             self,
#             mean: Sequence[Number] = None,
#             std: Sequence[Number] = None,
#             pad_size_divisor: int = 1,
#             pad_value: Union[float, int] = 0,
#             pad_mask: bool = False,
#             mask_pad_value: int = 0,
#             pad_seg: bool = False,
#             seg_pad_value: int = 255,
#             boxtype2tensor: bool = True,
#             non_blocking: Optional[bool] = False,
#             batch_augments: Optional[nn.ModuleList] = None
#     ):
#         super(HSIDetDataPreprocessor, self).__init__(
#             mean=mean,
#             std=std,
#             pad_size_divisor=pad_size_divisor,
#             pad_value=pad_value,
#             non_blocking=non_blocking
#         )
#         self.batch_augments = batch_augments
#         self.pad_mask = pad_mask
#         self.mask_pad_value = mask_pad_value
#         self.pad_seg = pad_seg
#         self.seg_pad_value = seg_pad_value
#         self.boxtype2tensor = boxtype2tensor
#
#
# class PatchEmbed(nn.Module):
#     def __init__(self):
#         super(PatchEmbed, self).__init__()
#         self.adap_padding = AdaptivePadding()
#         self.projection = nn.Conv2d(30, 256, kernel_size=1)
#         self.norm = nn.LayerNorm(256)
#
#
# class FocalLoss(nn.Module):
#     def __init__(self):
#         super(FocalLoss, self).__init__()
#
#
# class L1Loss(nn.Module):
#     def __init__(self):
#         super(L1Loss, self).__init__()
#
#
# class GIoULoss(nn.Module):
#     def __init__(self):
#         super(GIoULoss, self).__init__()
#
#
# class SpecDetrHead(nn.Module):
#     def __init__(self):
#         super(SpecDetrHead, self).__init__()
#         self.loss_cls = FocalLoss()
#         self.loss_bbox = L1Loss()
#         self.loss_iou = GIoULoss()
#         self.cls_branches = nn.ModuleList([
#             nn.Linear(256, 8)
#             for _ in range(7)
#         ])
#         self.reg_branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(256, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, 256),
#                 nn.ReLU(),
#                 nn.Linear(256, 4)
#             )
#             for _ in range(7)
#         ])


class SinePositionalEncoding(nn.Module):
    def __init__(self):
        super(SinePositionalEncoding, self).__init__()


class MultiScaleDeformableAttention_1(nn.Module):
    def __init__(self):
        super(MultiScaleDeformableAttention_1, self).__init__()
        self.dropout = nn.Dropout(0.0)
        self.sampling_offsets = nn.Linear(256, 128)
        self.attention_weights = nn.Linear(256, 64)
        self.value_proj = nn.Linear(256, 256)
        self.output_proj = nn.Linear(256, 256)


class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(256, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.0),
            ),
            nn.Linear(2048, 256),
            nn.Dropout(0.0),
        )
        self.dropout_layer = nn.Identity()
        self.gamma2 = nn.Identity()


class SpecDetrTransformerEncoderLayer(nn.Module):
    def __init__(self):
        super(SpecDetrTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiScaleDeformableAttention_1()
        self.ffn = FFN()
        self.norms = nn.ModuleList([
            nn.LayerNorm(256, eps=1e-5, elementwise_affine=True)
            for _ in range(2)
        ])


class SpecDetrTransformerEncoder(nn.Module):
    def __init__(self):
        super(SpecDetrTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            SpecDetrTransformerEncoderLayer()
            for _ in range(6)
        ])


class SpecDetrTransformerDecoderLayer(nn.Module):
    def __init__(self):
        super(SpecDetrTransformerDecoderLayer, self).__init__()
        self.cross_attn = MultiScaleDeformableAttention_1()
        self.ffn = FFN()
        self.norms = nn.ModuleList([
            nn.LayerNorm(256, eps=1e-5, elementwise_affine=True)
            for _ in range(2)
        ])


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(512, 256),
            nn.Linear(256, 256),
        ])


class SpecDetrTransformerDecoder(nn.Module):
    def __init__(self):
        super(SpecDetrTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            SpecDetrTransformerDecoderLayer()
            for _ in range(6)
        ])
        self.ref_point_head = MLP()
        self.norm = nn.LayerNorm(256, eps=1e-5, elementwise_affine=True)


class CdnQueryGenerator(nn.Module):
    def __init__(self):
        super(CdnQueryGenerator, self).__init__()


class BaseModel(nn.Module):
    def __init__(
            self,
            # data_preprocessor: OptConfigType = None,
            # init_cfg: OptMultiConfig = None
    ):
        super(BaseModel, self).__init__()
        self.data_processor = HSIDetDataPreprocessor()
        # self.init_cfg = init_cfg


class BaseDetector(BaseModel):
    def __init__(
            self,
            # data_preprocessor: OptConfigType = None,
            # init_cfg: OptMultiConfig = None
    ):
        super(BaseDetector, self).__init__()


class DetectionTransformer(BaseDetector):
    def __init__(
            self,
            backbone: nnBackbone = None,  # nnBackbone -> SpodBackbone
            encoder: nnEncoder = None,  # nnEncoder -> SpodEncoder
            decoder: nnDecoder = None,  # nnDecoder -> SpodDecoder
            bbox_head: nnBboxHead = None,  # nnBboxHead -> SpodBboxHead
            positional_encoding: nnPositionalEncoding = None,  # nnPositionalEncoding -> SpodPositionalEncoding
            neck: nnNeck = None,  # nnNeck -> SpodNeck
            num_queries: int = 900,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            # data_preprocessor: OptConfigType = None,
            # init_cfg: OptMultiConfig = None
    ):
        super(DetectionTransformer, self).__init__()

        self.backbone = backbone or No_backbone_ST()
        self.bbox_head = bbox_head or SpecDetrHead()
        self.neck = neck

        self._init_layers()

    def _init_layers(self):
        self.encoder = encoder or SpodEncoder()
        self.decoder = decoder or SpodDecoder()
        self.positional_encoding = positional_encoding or SpodPositionalEncoding()

        # TODO: 确认
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_queries = num_queries
        self.encoder_layers_num = encoder.num_layers


class DeformableDETR(DetectionTransformer):
    def __init__(
            self,
            decoder: nnDecoder = None,
            bbox_head: nnBboxHead = None,
            with_box_refine: bool = True,
            as_two_stage: bool = True,
            num_feature_levels: int = 2,
            **kwargs
    ):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels

        if bbox_head is not None:
            bbox_head.share_pred_layer = not with_box_refine
            bbox_head.num_pred_layer = decoder.num_layers + self.as_two_stage
            bbox_head.as_two_stage = as_two_stage

        super(DeformableDETR, self).__init__(decoder=decoder, bbox_head=bbox_head, **kwargs)


class SpecDetr(DeformableDETR):
    def __init__(
            self, *,
            candidate_bboxes_size: float = 0.01,
            scale_gt_bboxes_size: float = 0,
            training_dn: bool = True,
            dn_only_pos: bool = False,
            remove_last_candidate: bool = True,
            num_query_per_cat: int = 5,
            num_fix_query: int = 0,
            anchor_iou_th: float = 0.5,
            query_interval: int = 1,
            dn_type: str = "CDN",
            query_initial: str = "one",
            dn_cfg: OptConfigType = None,
            **kwargs
    ):
        self.query_initial = query_initial
        super(SpecDetr, self).__init__(**kwargs)

        self.data_preprocessor = HSIDetDataPreprocessor()
        self.backbone = No_backbone_ST()
        self.bbox_head = SpecDetrHead()
        self.positional_encoding = SinePositionalEncoding()
        self.encoder = SpecDetrTransformerEncoder()
        self.decoder = SpecDetrTransformerDecoder()
        self.memory_trans_fc = nn.Linear(256, 256)
        self.memory_trans_norm = nn.LayerNorm(256, eps=1e-5, elementwise_affine=True)
        self.dn_query_generator = CdnQueryGenerator()

    def _init_layers(self):
        ...  # TODO
