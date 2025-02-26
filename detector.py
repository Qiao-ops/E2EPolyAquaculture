import cv2
import torch
import torch.nn.functional as F

from math import log
from torch import nn
from hisup.backbones import build_backbone
from hisup.utils.polygon import generate_polygon
from hisup.utils.polygon import get_pred_junctions
from skimage.measure import label, regionprops


def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)

    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])

    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        t = ((mask == 1) | (mask == 2)).float()
        w = t.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(t/w)

    return loss.mean()

# Copyright (c) 2019 BangguWu, Qilong Wang
# Modified by Bowen Xu, Jiakun Xu, Nan Xue and Gui-song Xia

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        C = channel

        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        y = self.avg_pool(x1 + x2)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1 ,-2).unsqueeze(-1)
        y = self.sigmoid(y)

        out = self.out_conv(x2 * y.expand_as(x2))
        return out

class BuildingDetector(nn.Module):
    def __init__(self, cfg, test=False):
        super(BuildingDetector, self).__init__()
        # 新增：读取类别数配置
        self.num_classes = cfg.MODEL.NUM_CLASSES  # 需在配置文件中定义
        
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME
        self.junc_loss = nn.CrossEntropyLoss()
        self.test_inria = 'inria' in cfg.DATASETS.TEST[0]
        
        if not test:
            from hisup.encoder import Encoder
            self.encoder = Encoder(cfg)

        self.pred_height = cfg.DATASETS.TARGET.HEIGHT
        self.pred_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS
        
        # 修改所有输出层通道数为num_classes
        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

        self.a2m_att = ECA(dim_in)
        self.a2j_att = ECA(dim_in)

        # 修改预测器输出通道数
        self.mask_predictor = self._make_predictor(dim_in, self.num_classes)  # 原为2
        self.jloc_predictor = self._make_predictor(dim_in, 3)                 # 保持3通道（凹/凸/背景）
        self.afm_predictor = self._make_predictor(dim_in, self.num_classes)   # 原为2
        
        self.refuse_conv = self._make_conv(self.num_classes, dim_in//2, dim_in)  # 输入通道改为num_classes
        self.final_conv = self._make_conv(dim_in*2, dim_in, self.num_classes) # 原为2

        self.train_step = 0

    def forward(self, images, annotations=None):
        if self.training:
            return self.forward_train(images, annotations)
        else:
            return self.forward_test(images, annotations)

    def forward_test(self, images, annotations=None):
        device = images.device
        outputs, features = self.backbone(images)

        # 特征提取
        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        # 注意力机制
        mask_att_feature = self.a2m_att(afm_feature, mask_feature)
        jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)

        # 预测头
        mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        # 最终掩膜预测
        afm_conv = self.refuse_conv(afm_pred)
        remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))

        # 后处理
        joff_pred = outputs[:, :].sigmoid() - 0.5
        mask_pred = mask_pred.softmax(1)  # 多类概率 [B, C, H, W]
        jloc_convex_pred = jloc_pred.softmax(1)[:, 2:3]  # 凸点预测
        jloc_concave_pred = jloc_pred.softmax(1)[:, 1:2]  # 凹点预测
        remask_pred = remask_pred.softmax(1)  # 多类概率 [B, C, H, W]

        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons = []
        batch_masks = []
        batch_scores = []
        batch_juncs = []

        for b in range(remask_pred.size(0)):
            # 获取类别预测结果
            class_mask = remask_pred[b].argmax(dim=0).cpu().numpy()  # [H, W]
            mask_pred_per_im = cv2.resize(class_mask.astype(np.uint8), 
                                         (self.origin_width, self.origin_height),
                                         interpolation=cv2.INTER_NEAREST)
            
            # 获取关键点预测
            juncs_pred = get_pred_junctions(jloc_concave_pred[b], 
                                           jloc_convex_pred[b], 
                                           joff_pred[b])
            juncs_pred[:, 0] *= scale_x
            juncs_pred[:, 1] *= scale_y

            if not self.test_inria:
                polys, scores = [], []
                # 处理每个类别
                for cls_id in range(1, self.num_classes):  # 0为背景
                    cls_mask = (mask_pred_per_im == cls_id).astype(np.uint8)
                    props = regionprops(label(cls_mask))
                    for prop in props:
                        poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(
                            prop, cls_mask, juncs_pred, 0, self.test_inria
                        )
                        if juncs_sa.shape[0] == 0:
                            continue
                        polys.append(poly)
                        scores.append(score)
                batch_polygons.append(polys)
                batch_scores.append(scores)
            
            batch_masks.append(mask_pred_per_im)
            batch_juncs.append(juncs_pred)

        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        return output, {}

    def forward_train(self, images, annotations=None):
        self.train_step += 1
        device = images.device
        targets, metas = self.encoder(annotations)
        outputs, features = self.backbone(images)

        loss_dict = {
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_mask': 0.0,
            'loss_afm': 0.0,
            'loss_remask': 0.0
        }

        # 特征提取（同测试阶段）
        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        mask_att_feature = self.a2m_att(afm_feature, mask_feature)
        jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)

        # 预测头
        mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        # 最终掩膜预测
        afm_conv = self.refuse_conv(afm_pred)
        remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))

        # 损失计算
        if targets is not None:
            # 确保标签格式正确 [B, H, W]
            target_mask = targets['mask'].squeeze(dim=1).long()
            
            loss_dict['loss_jloc'] = self.junc_loss(jloc_pred, targets['jloc'].squeeze(dim=1))
            loss_dict['loss_joff'] = sigmoid_l1_loss(outputs[:, :], targets['joff'], -0.5, targets['jloc'])
            loss_dict['loss_mask'] = F.cross_entropy(mask_pred, target_mask)
            loss_dict['loss_afm'] = F.l1_loss(afm_pred, targets['afmap'])
            loss_dict['loss_remask'] = F.cross_entropy(remask_pred, target_mask)

        return loss_dict, {}

    # 以下辅助函数保持不变
    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, 3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, 3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        return nn.Sequential(
            nn.Conv2d(dim_in, m, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, dim_out, 1)
        )

# class BuildingDetector(nn.Module):
#     def __init__(self, cfg, test=False):
#         super(BuildingDetector, self).__init__()
#         self.backbone = build_backbone(cfg)
#         self.backbone_name = cfg.MODEL.NAME
#         self.junc_loss = nn.CrossEntropyLoss()
#         self.test_inria = 'inria' in cfg.DATASETS.TEST[0]
#         if not test:
#             from hisup.encoder import Encoder
#             self.encoder = Encoder(cfg)

#         self.pred_height = cfg.DATASETS.TARGET.HEIGHT
#         self.pred_width = cfg.DATASETS.TARGET.WIDTH
#         self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
#         self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

#         dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS
#         self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
#         self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
#         self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

#         self.a2m_att = ECA(dim_in)
#         self.a2j_att = ECA(dim_in)

#         self.mask_predictor = self._make_predictor(dim_in, 2)
#         self.jloc_predictor = self._make_predictor(dim_in, 3)
#         self.afm_predictor = self._make_predictor(dim_in, 2)

#         self.refuse_conv = self._make_conv(2, dim_in//2, dim_in)
#         self.final_conv = self._make_conv(dim_in*2, dim_in, 2)

#         self.train_step = 0
        
#     def forward(self, images, annotations = None):
#         if self.training:
#             return self.forward_train(images, annotations=annotations)
#         else:
#             return self.forward_test(images, annotations=annotations)

#     def forward_test(self, images, annotations = None):
#         device = images.device
#         outputs, features = self.backbone(images)

#         mask_feature = self.mask_head(features)
#         jloc_feature = self.jloc_head(features)
#         afm_feature = self.afm_head(features)

#         mask_att_feature = self.a2m_att(afm_feature, mask_feature)
#         jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)

#         mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
#         jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
#         afm_pred = self.afm_predictor(afm_feature)

#         afm_conv = self.refuse_conv(afm_pred)
#         remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))

#         joff_pred = outputs[:, :].sigmoid() - 0.5
#         mask_pred = mask_pred.softmax(1)[:,1:]
#         jloc_convex_pred = jloc_pred.softmax(1)[:, 2:3]
#         jloc_concave_pred = jloc_pred.softmax(1)[:, 1:2]
#         remask_pred = remask_pred.softmax(1)[:, 1:]
        
#         scale_y = self.origin_height / self.pred_height
#         scale_x = self.origin_width / self.pred_width

#         batch_polygons = []
#         batch_masks = []
#         batch_scores = []
#         batch_juncs = []

#         for b in range(remask_pred.size(0)):
#             mask_pred_per_im = cv2.resize(remask_pred[b][0].cpu().numpy(), (self.origin_width, self.origin_height))
#             juncs_pred = get_pred_junctions(jloc_concave_pred[b], jloc_convex_pred[b], joff_pred[b])
#             juncs_pred[:,0] *= scale_x
#             juncs_pred[:,1] *= scale_y

#             if not self.test_inria:
#                 polys, scores = [], []
#                 props = regionprops(label(mask_pred_per_im > 0.5))
#                 for prop in props:
#                     poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im, \
#                                                                             juncs_pred, 0, self.test_inria)
#                     if juncs_sa.shape[0] == 0:
#                         continue

#                     polys.append(poly)
#                     scores.append(score)
#                 batch_scores.append(scores)
#                 batch_polygons.append(polys)
            
#             batch_masks.append(mask_pred_per_im)
#             batch_juncs.append(juncs_pred)

#         extra_info = {}
#         output = {
#             'polys_pred': batch_polygons,
#             'mask_pred': batch_masks,
#             'scores': batch_scores,
#             'juncs_pred': batch_juncs
#         }
#         return output, extra_info

#     def forward_train(self, images, annotations = None):
#         self.train_step += 1

#         device = images.device
#         targets, metas = self.encoder(annotations)
#         outputs, features = self.backbone(images)

#         loss_dict = {
#             'loss_jloc': 0.0,
#             'loss_joff': 0.0,
#             'loss_mask': 0.0,
#             'loss_afm' : 0.0,
#             'loss_remask': 0.0
#         }

#         mask_feature = self.mask_head(features)
#         jloc_feature = self.jloc_head(features)
#         afm_feature = self.afm_head(features)

#         mask_att_feature = self.a2m_att(afm_feature, mask_feature)
#         jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)

#         mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
#         jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
#         afm_pred = self.afm_predictor(afm_feature)

#         afm_conv = self.refuse_conv(afm_pred)
#         remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))

#         if targets is not None:
#             loss_dict['loss_jloc'] += self.junc_loss(jloc_pred, targets['jloc'].squeeze(dim=1))
#             loss_dict['loss_joff'] += sigmoid_l1_loss(outputs[:, :], targets['joff'], -0.5, targets['jloc'])
#             loss_dict['loss_mask'] += F.cross_entropy(mask_pred, targets['mask'].squeeze(dim=1).long())
#             loss_dict['loss_afm'] += F.l1_loss(afm_pred, targets['afmap'])
#             loss_dict['loss_remask'] += F.cross_entropy(remask_pred, targets['mask'].squeeze(dim=1).long())
#         extra_info = {}

#         return loss_dict, extra_info
    
#     def _make_conv(self, dim_in, dim_hid, dim_out):
#         layer = nn.Sequential(
#             nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
#             nn.BatchNorm2d(dim_hid),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
#             nn.BatchNorm2d(dim_hid),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(dim_out),
#             nn.ReLU(inplace=True),
#         )
#         return layer

#     def _make_predictor(self, dim_in, dim_out):
#         m = int(dim_in / 4)
#         layer = nn.Sequential(
#                     nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(m, dim_out, kernel_size=1),
#                 )
#         return layer


def get_pretrained_model(cfg, dataset, device, pretrained=True):
    PRETRAINED = {
        'crowdai': 'https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/crowdai_hrnet48_e100.pth',
        'inria': 'https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/inria_hrnet48_e5.pth'
    }

    model = BuildingDetector(cfg, test=True)
    if pretrained:
        url = PRETRAINED[dataset]
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
        state_dict = {k[7:]:v for k,v in state_dict['model'].items() if k[0:7] == 'module.'}
        model.load_state_dict(state_dict)
        model = model.eval()
        return model
    return model
