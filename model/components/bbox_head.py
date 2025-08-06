from torch import nn
import copy
from .cost import *
from .loss import *

__all__ = ["nnBboxHead", "SpodBboxHead", "SpecDetrHead"]


class DynamicIOUHungarianAssigner(nn.Module):
    def __init__(
            self,
            match_costs: list[Cost] = None,
            base_match_num: int = 1,
            match_num: int = 10,
            iou_loss_th: float = 0.05,
            total_num_max: int = 100,
            dynamic_match: bool = True,
    ):
        super(DynamicIOUHungarianAssigner, self).__init__()
        self.match_num = match_num
        self.dynamic_match = dynamic_match
        self.base_match_num = base_match_num
        self.iou_loss_th = iou_loss_th
        self.total_num_max = total_num_max
        if match_costs is None:
            match_costs = [FocalLossCost(), BBoxL1Cost(), IoUCost(), IoULossCost()]
        self.match_costs = match_costs


class FNN(nn.Module):
    def __init__(
            self,
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            ffn_drop=0.,
            dropout_layer=None,
            add_identity=True,
            init_cfg=None,
            layer_scale_init_value=0.
    ):
        super(FNN, self).__init__()


class nnBboxHead(nn.Module):
    def __init__(
            self,
            num_classes: int,
            embed_dims: int = 256,
            num_reg_fcs: int = 2,
            sync_cls_avg_factor: bool = True,
            use_nms: bool = True,
            neg_cls: bool = True,
            cls_loss: Loss = None,
            bbox_loss: Loss = None,
            iou_loss: Loss = None
    ):
        super(nnBboxHead, self).__init__()
        self.assigner = DynamicIOUHungarianAssigner()
        self.cls_loss = cls_loss or FocalLoss()
        self.bbox_loss = bbox_loss or L1Loss()
        self.iou_loss = iou_loss or GIoULoss()

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.use_nms = use_nms
        self.neg_cls = neg_cls
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()

    def _init_layers(self):
        self.fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        # reg branch
        self.activate = nn.ReLU()
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            dict(type="ReLU", inplace=True),
            dropout=0.0,
            add_residual=False
        )
        self.fc_reg = nn.Linear(self.embed_dims, 4)


class SpodBboxHead(nnBboxHead):
    def __init__(self, num_classes: int):
        super(SpodBboxHead, self).__init__(num_classes)


class SpecDetrHead(nnBboxHead):
    def __init__(
            self,
            share_pred_layer: bool = False,
            num_pred_layer: int = 7,
            as_two_stage: bool = True,
            pre_bboxes_round: bool = False,
            decoupe_dn: bool = False,
            dn_only_pos: bool = False,
            dn_loss_weight: List[float] = None,
            iou_threshold: float = 0.01,
            class_wise_nms: bool = False,
            two_stage_training: bool = False,
            **kwargs
    ):
        if dn_loss_weight is None:
            dn_loss_weight = [1, 1, 1]
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
        self.pre_bboxes_round = pre_bboxes_round
        assert not (decoupe_dn and dn_only_pos), "Both decoupe_dn and dn_only_pos cannot be True at the same time."
        self.decoupe_dn = decoupe_dn
        self.dn_only_pos = dn_only_pos
        self.dn_loss_weight = dn_loss_weight
        self.iou_threshold = iou_threshold
        self.class_wise_nms = class_wise_nms
        self.two_stage_training = two_stage_training
        self.two_stage_training_staring = False
        super(SpecDetrHead, self).__init__(**kwargs)

    def _init_layers(self):
        fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList([copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)])
