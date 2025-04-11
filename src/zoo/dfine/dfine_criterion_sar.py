"""
D-FINE: SAR图像增强版损失函数
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

该文件包含针对SAR图像的增强版损失函数，主要改进包括：
1. 增加边界框定位精度的权重
2. 添加边缘感知损失，提高对SAR图像中模糊边界的敏感度
3. 针对SAR图像中小目标的优化
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance
from .dfine_criterion import DFINECriterion


@register()
class SARDFINECriterion(DFINECriterion):
    """针对SAR图像的增强版D-FINE损失函数"""

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=7,  # SAR飞机数据集的类别数
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        giou_weight=2.5,  # 增加GIoU损失的权重
        bbox_weight=2.0,  # 增加边界框回归损失的权重
        small_object_scale=1.5,  # 小目标缩放因子
    ):
        """创建SAR图像增强版损失函数
        
        Args:
            matcher: 计算目标和预测之间匹配的模块
            weight_dict: 包含损失名称和相应权重的字典
            losses: 要应用的所有损失列表
            alpha: Focal Loss的alpha参数
            gamma: Focal Loss的gamma参数
            num_classes: 对象类别数量，不包括特殊的无对象类别
            reg_max: D-FINE中离散箱的最大数量
            boxes_weight_format: 框权重格式(iou等)
            share_matched_indices: 是否共享匹配的索引
            giou_weight: GIoU损失的权重，增强对边界框准确性的重视
            bbox_weight: L1边界框损失的权重
            small_object_scale: 小目标损失缩放因子
        """
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
        )
        
        # 增强版损失参数
        self.giou_weight = giou_weight
        self.bbox_weight = bbox_weight
        self.small_object_scale = small_object_scale
        
        # 更新权重字典以反映新的权重
        if 'loss_giou' in self.weight_dict:
            self.weight_dict['loss_giou'] *= self.giou_weight
        if 'loss_bbox' in self.weight_dict:
            self.weight_dict['loss_bbox'] *= self.bbox_weight
            
    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """计算与边界框相关的损失：L1回归损失和GIoU损失
        
        目标字典必须包含键"boxes"，包含维度为[nb_target_boxes, 4]的张量
        目标框的格式为(center_x, center_y, w, h)，由图像大小归一化
        
        针对SAR图像增强：
        1. 增加小目标的损失权重
        2. 提高边界框损失和GIoU损失权重
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # 计算目标的面积，用于识别小目标
        box_areas = target_boxes[:, 2] * target_boxes[:, 3]  # w * h
        
        # 根据面积创建小目标权重
        # 小于0.01（相对于图像大小1.0）的目标被视为小目标
        small_targets_mask = box_areas < 0.01
        small_target_weights = torch.ones_like(box_areas)
        small_target_weights[small_targets_mask] = self.small_object_scale
        
        losses = {}
        
        # 应用小目标权重到L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox * small_target_weights.unsqueeze(1)
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        
        # 计算并应用小目标权重到GIoU损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou * small_target_weights
        
        # 应用额外的boxes_weight（如果有）
        if boxes_weight is not None:
            loss_giou = loss_giou * boxes_weight
            
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        
        return losses
        
    def loss_edge_aware(self, outputs, targets, indices, num_boxes):
        """边缘感知损失 - 鼓励预测框更好地捕捉SAR图像中的目标边缘
        
        这种损失对于SAR图像中模糊的目标边界特别有用
        """
        if "pred_corners" not in outputs:
            return {"loss_edge": torch.as_tensor(0.0, device=outputs["pred_boxes"].device)}
            
        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # 获取预测的角点分布
        pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
        
        # 获取参考点
        ref_points = outputs["ref_points"][idx].detach()
        
        # 计算正确的边框距离目标
        with torch.no_grad():
            target_distances = bbox2distance(
                ref_points,
                box_cxcywh_to_xyxy(target_boxes),
                self.reg_max,
                outputs["reg_scale"],
                outputs["up"],
            )[0]
        
        # 计算角点分布的峰值和方差
        pred_prob = F.softmax(pred_corners, dim=1)
        peak_values, peak_indices = torch.max(pred_prob, dim=1)
        
        # 计算分布的离散程度 - 更集中的分布表明更确定的边缘
        entropy = -torch.sum(pred_prob * torch.log(pred_prob + 1e-10), dim=1)
        
        # 计算预测峰值与目标位置的一致性
        target_indices = torch.argmax(target_distances, dim=1)
        peak_alignment = F.smooth_l1_loss(
            peak_indices.float(), target_indices.float(), reduction='none'
        )
        
        # 组合损失：鼓励更确定（低熵）的预测，并与目标位置一致
        edge_loss = (entropy + peak_alignment) * peak_values
        
        # 对小目标加权
        box_areas = target_boxes[:, 2] * target_boxes[:, 3]
        small_targets_mask = box_areas < 0.01
        small_target_weights = torch.ones_like(box_areas)
        small_target_weights[small_targets_mask] = self.small_object_scale
        
        edge_loss = edge_loss * small_target_weights.repeat(4)  # 每个框有4个边
        
        return {"loss_edge": edge_loss.mean()}
        
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """根据损失名称调用相应的损失函数"""
        loss_map = {
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'boxes': self.loss_boxes,
            'fgl': self.loss_local,
            'ddf': self.loss_local,
            'edge': self.loss_edge_aware,  # 添加新的边缘感知损失
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """前向计算损失"""
        # 使用父类的forward方法
        losses = super().forward(outputs, targets, **kwargs)
        
        # 在调试时打印实际损失值
        if torch.distributed.get_rank() == 0 and torch.rand(1).item() < 0.01:
            print_losses = {k: v.item() for k, v in losses.items() if not k.startswith('aux_')}
            print(f"SAR损失函数: {print_losses}")
            
        return losses 