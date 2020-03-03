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
    def forward(self, image_batch_shape, attention_mask, annotations):
        h, w = image_batch_shape[2], image_batch_shape[3]

        attention_losses = []

        batch_size = annotations.shape[0]

        for j in range(batch_size):
            annotation = annotations[j, :, :]
            annotation = annotation[annotation[:, 4] != -1]

            cond1 = torch.le(annotation[:, 0], w)
            cond2 = torch.le(annotation[:, 1], h)
            cond3 = torch.le(annotation[:, 2], w)
            cond4 = torch.le(annotation[:, 3], h)
            cond = cond1 * cond2 * cond3 * cond4

            annotation = annotation[cond, :]

            if annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    attention_losses.append(torch.tensor(0).float().cuda())
                else:
                    attention_losses.append(torch.tensor(0).float())
                continue
            
            area = (annotation[:, 2] - annotation[:, 0]) * (annotation[:, 3] - annotation[:, 1])

            attention_loss = []
            for id in range(len(attention_mask)):
                attention_map = attention_mask[id][j, 0, :, :]

                min_area = (2 ** (id+ 5)) ** 2 * 0.5
                max_area = (2 ** (id + 5) * 1.58) ** 2 * 2

                indice1 = torch.ge(area, min_area)
                indice2 = torch.le(area, max_area) 
                indice = indice1 * indice2

                level_annotation = annotation[indice, :].clone()

                attention_h, attention_w = attention_map.shape 

                if level_annotation.shape[0]:
                    level_annotation[:, 0] *= attention_w / w
                    level_annotation[:, 1] *= attention_h / h
                    level_annotation[:, 2] *= attention_w / w
                    level_annotation[:, 3] *= attention_h / h
                
                gt_mask = torch.zeros(attention_map.shape)

                if torch.cuda.is_available():
                    gt_mask = gt_mask.cuda()
                
                for i in range(level_annotation.shape[0]):
                    x1 = max(int(level_annotation[i, 0]), 0)
                    y1 = max(int(level_annotation[i, 1]), 0)
                    x2 = min(math.ceil(level_annotation[i, 2]) + 1, attention_w)
                    y2 = min(math.ceil(level_annotation[i, 3]) + 1, attention_h)

                    gt_mask[y1:y2, x1:x2] = 1
                
                gt_mask = gt_mask[gt_mask >= 0]
                predicti_mask = attention_map[attention_map >= 0]

                attention_loss.append(F.binary_cross_entropy(predicti_mask, gt_mask))
            attention_losses.append(torch.stack(attention_loss).mean())

        return torch.stack(attention_losses).mean(dim=0, keepdim=True)
