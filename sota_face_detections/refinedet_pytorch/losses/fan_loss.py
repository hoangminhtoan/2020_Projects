'''
Implement mask loss from Face Attention Network (FAN) : https://arxiv.org/abs/1711.07246
'''

import numpy as np 
import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih 
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih 
    IoU = intersection / ua 

    return IoU

class AttentionLoss(nn.Module):

    def forward(self, img_batch_shape, attention_mask, bboxs):

        h, w = img_batch_shape[2], img_batch_shape[3]

        mask_losses = []

        batch_size = bboxs.shape[0]
        for j in range(batch_size):

            bbox_annotation = bboxs[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            cond1 = torch.le(bbox_annotation[:, 0], w)
            cond2 = torch.le(bbox_annotation[:, 1], h)
            cond3 = torch.le(bbox_annotation[:, 2], w)
            cond4 = torch.le(bbox_annotation[:, 3], h)
            cond = cond1 * cond2 * cond3 * cond4

            bbox_annotation = bbox_annotation[cond, :]

            if bbox_annotation.shape[0] == 0:
                mask_losses.append(torch.tensor(0).float().cuda())
                continue

            bbox_area = (bbox_annotation[:, 2] - bbox_annotation[:, 0]) * (bbox_annotation[:, 3] - bbox_annotation[:, 1])

            mask_loss = []
            for id in range(len(attention_mask)):

                attention_map = attention_mask[id][j, 0, :, :]

                min_area = (2 ** (id + 5)) ** 2 * 0.5
                max_area = (2 ** (id + 5) * 1.58) ** 2 * 2

                level_bbox_indice1 = torch.ge(bbox_area, min_area)
                level_bbox_indice2 = torch.le(bbox_area, max_area)

                level_bbox_indice = level_bbox_indice1 * level_bbox_indice2

                level_bbox_annotation = bbox_annotation[level_bbox_indice, :].clone()

                #level_bbox_annotation = bbox_annotation.clone()

                attention_h, attention_w = attention_map.shape

                if level_bbox_annotation.shape[0]:
                    level_bbox_annotation[:, 0] *= attention_w / w
                    level_bbox_annotation[:, 1] *= attention_h / h
                    level_bbox_annotation[:, 2] *= attention_w / w
                    level_bbox_annotation[:, 3] *= attention_h / h

                mask_gt = torch.zeros(attention_map.shape)
                mask_gt = mask_gt.cuda()

                for i in range(level_bbox_annotation.shape[0]):

                    x1 = max(int(level_bbox_annotation[i, 0]), 0)
                    y1 = max(int(level_bbox_annotation[i, 1]), 0)
                    x2 = min(math.ceil(level_bbox_annotation[i, 2]) + 1, attention_w)
                    y2 = min(math.ceil(level_bbox_annotation[i, 3]) + 1, attention_h)

                    mask_gt[y1:y2, x1:x2] = 1

                mask_gt = mask_gt[mask_gt >= 0]
                mask_predict = attention_map[attention_map >= 0]

                mask_loss.append(F.binary_cross_entropy(mask_predict, mask_gt))
            mask_losses.append(torch.stack(mask_loss).mean())

        return torch.stack(mask_losses).mean(dim=0, keepdim=True)