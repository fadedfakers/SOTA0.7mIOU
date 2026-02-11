import torch
from torch.nn import functional as F
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models.builder import (DETECTORS, build_backbone, build_neck, 
                                    build_head, build_voxel_encoder, build_middle_encoder)
from mmdet3d.ops import Voxelization

@DETECTORS.register_module()
class RailFusionNet(Base3DDetector):
    def __init__(self,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 backbone=None,
                 pts_neck=None,
                 img_backbone=None,
                 img_neck=None,
                 neck=None,
                 bbox_head=None,
                 rail_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RailFusionNet, self).__init__(init_cfg=init_cfg)

        if voxel_layer:
            self.voxel_layer = Voxelization(**voxel_layer)
        if voxel_encoder:
            self.voxel_encoder = build_voxel_encoder(voxel_encoder)
        if middle_encoder:
            self.middle_encoder = build_middle_encoder(middle_encoder)
        
        if backbone:
            self.pts_backbone = build_backbone(backbone)
        if pts_neck:
            self.pts_neck = build_neck(pts_neck)

        if img_backbone:
            self.img_backbone = build_backbone(img_backbone)
        if img_neck:
            self.img_neck = build_neck(img_neck)

        if neck:
            self.neck = build_neck(neck)

        if bbox_head:
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.train_cfg = train_cfg.get('pts', train_cfg) if train_cfg else None
            self.bbox_head.test_cfg = test_cfg.get('pts', test_cfg) if test_cfg else None
            
        if rail_head:
            self.rail_head = build_head(rail_head)
            
        self.debug_counter = 0

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @torch.no_grad()
    def voxelize(self, points):
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors

    def extract_img_feat(self, img, img_metas):
        if img is None: return None
        input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, points):
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if hasattr(self, 'pts_neck') and self.pts_neck is not None:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        img_feats = None
        if self.with_img_backbone and img is not None:
            img_feats = self.extract_img_feat(img, img_metas)

        pts_feats = self.extract_pts_feat(points)

        if hasattr(self, 'neck') and self.neck is not None:
            fused_feats = self.neck(pts_feats, img_feats, img_metas)
            return fused_feats
        else:
            return pts_feats

    # [关键修改] 加入 gt_masks 参数
    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, gt_masks=None, gt_poly_3d=None, img=None, **kwargs):
        x = self.extract_feat(points, img, img_metas)
        losses = dict()
        
        # 兼容性处理
        feat = x[0] if isinstance(x, (list, tuple)) else x
        if not isinstance(feat, (list, tuple)):
            feat = [feat] # [B, C, H, W]

        # --- [DEBUG 探针] ---
        self.debug_counter += 1
        if self.debug_counter < 3:
            print(f"\n🔍 [DEBUG Forward] Step {self.debug_counter}")
            print(f"   Feature Shape: {feat[0].shape}")
            if gt_masks is not None:
                print(f"   GT Masks Received: Yes, Shape={gt_masks.shape}, Max={gt_masks.max()}")
            else:
                print("   ⚠️ GT Masks Received: NO (None)")
                # 如果没有收到 gt_masks，尝试从 kwargs 里找
                if 'gt_masks' in kwargs:
                    print(f"   Found gt_masks in kwargs!")
                    gt_masks = kwargs['gt_masks']

        # 1. 轨道分割 Loss (BEV Seg)
        if self.rail_head:
            # 无论 gt_masks 是否为 None，都先跑 forward，防止 DDP 报错
            seg_preds = self.rail_head(feat[0]) 
            
            if gt_masks is not None:
                loss_seg = self.rail_head.loss(seg_preds, gt_masks)
                losses.update(loss_seg)
            else:
                # 如果真的没有 mask，给一个 0 loss 占位
                losses['loss_seg'] = seg_preds.sum() * 0.0
            
        # 2. 检测头 Loss
        if self.bbox_head:
            bbox_outs = self.bbox_head(feat)
            loss_bbox = self.bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs, img_metas=img_metas)
            losses.update(loss_bbox)
            
        return losses

    def simple_test(self, points, img_metas, img=None, **kwargs):
        x = self.extract_feat(points, img, img_metas)
        feat = x[0] if isinstance(x, (list, tuple)) else x
        if not isinstance(feat, (list, tuple)): 
            feat = [feat]
        
        # --- BEV Mask 预测 ---
        rail_results = [None for _ in range(len(img_metas))]
        if self.rail_head:
            seg_preds = self.rail_head(feat[0]) # [B, 1, H, W]
            seg_masks = self.rail_head.get_seg_masks(seg_preds) # Sigmoid -> [0, 1]
            
            for i in range(len(img_metas)):
                # 转 numpy [H, W]
                mask = seg_masks[i, 0].detach().cpu().numpy()
                rail_results[i] = mask

        # --- BBox 预测 ---
        bbox_results = [dict() for _ in range(len(img_metas))]
        if self.bbox_head:
            bbox_outs = self.bbox_head(feat)
            bbox_list = self.bbox_head.get_bboxes(bbox_outs, img_metas, rescale=True)
            for i, (bboxes, scores, labels) in enumerate(bbox_list):
                from mmdet3d.core import bbox3d2result
                bbox_results[i] = bbox3d2result(bboxes, scores, labels)
        
        # --- 融合 ---
        for i in range(len(bbox_results)):
            if self.rail_head:
                bbox_results[i]['bev_seg_mask'] = rail_results[i]
                
        return bbox_results

    def aug_test(self, points, img_metas, img=None, **kwargs):
        return self.simple_test(points, img_metas, img, **kwargs)