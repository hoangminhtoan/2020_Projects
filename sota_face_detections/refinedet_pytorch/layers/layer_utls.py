import torch 
import torch.nn as nn 
import numpy as np 

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def basic_conv(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, group=1, bias=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=group, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample 
        self.stride = stride 

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual 
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride 

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual 
        out = self.relu(out)

        return out 

class BasicSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(BasicSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BottleneckSELayer(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(BottleneckSELayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes *4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = BasicSELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# transfer connection block (TCB)
class TCB(nn.Module):
    def __init__(self, feature_size=256):
        super(TCB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size)
        )

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2(out)

        return out

# receptive context module (rcm)
class RCM(nn.Module):
    def __init__(self, in_planes, out_planes=256, stride=1, scale=0.1):
        super(RCM, self).__init__()

        self.scale = scale
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            basic_conv(in_planes, inter_planes, kernel_size=1, stride=1),
            basic_conv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, group=inter_planes)
        )

        self.branch1 = nn.Sequential(
            basic_conv(in_planes, inter_planes, kernel_size=1, stride=1),
            basic_conv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            basic_conv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, group=inter_planes)
        )

        self.branch2 = nn.Sequential(
            basic_conv(in_planes, inter_planes, kernel_size=1, stride=1),
            basic_conv(inter_planes, inter_planes, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            basic_conv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, group=inter_planes)
        )

        self.branch3 = nn.Sequential(
            basic_conv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            basic_conv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            basic_conv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            basic_conv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, group=inter_planes)
        )

        self.output = basic_conv(inter_planes * 4, 1, kernel_size=1, stride=1)
        self.act = nn.Sigmod()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.output(out)
        out = out * self.scale + x
        out = self.act(out)

        return out

class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__() 
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda() 
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean 

        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda() 
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std 

    def forward(self, boxes, deltas):
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths 
        ctr_y = boxes[:, :, 1] + 0.5 * heights 

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths 
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths 
        pred_h = torch.exp(dh) * heights 

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w 
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h 
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w 
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h 

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes 

class ClipBoxes(nn.Module):
    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes
        