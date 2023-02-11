import math

import torch.nn as nn
import numpy as np

# closure setting
__all__ = [
    "ResNet34_8s"
]

# pretrained models
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def dilation_conv3x3(in_channels, out_channels, stride=1, dilation=1):
    """
    dilated convolution with fixed kernel size---3x3
    https://arxiv.org/abs/1511.07122
    """
    kernel_size = np.asarray((3, 3))
    up_sampling_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    full_padding = (up_sampling_kernel_size - 1) // 2
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, down_sampling=None, stride=1, dilation=1):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            dilation_conv3x3(in_channels, out_channels, stride, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            dilation_conv3x3(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.down_sampling = down_sampling

        # shortcut
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.residual_function(x)
        if self.down_sampling:
            out += self.down_sampling(self.shortcut(x))
        else:
            out += self.shortcut(x)
        return nn.ReLU(inplace=True)(out)


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 num_blocks,
                 inp_ch=3,
                 num_classes=1000,
                 remove_avg_pool_layer=False,
                 fully_conv=False,
                 output_stride=32):

        self.in_planes = 64
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1

        self.remove_avg_pool_layer = remove_avg_pool_layer
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.fully_conv:
            self.avg_pool = nn.AvgPool2d(7, padding=3, stride=1)
        else:
            self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channels, num_block, stride=1):
        """
        make resnet layers,
        one layer may contain more than one residual block
        """
        down_sampling = None
        if stride != 1 or self.in_planes != out_channels * block.expansion:

            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:

                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride

            # 1x1 convolution, without dilation.
            down_sampling = nn.Sequential(
                nn.Conv2d(self.in_planes, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_planes, out_channels, stride, down_sampling, dilation=self.current_dilation)]
        self.in_planes = out_channels * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.in_planes, out_channels, dilation=self.current_dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        if not self.remove_avg_pool_layer:
            x = self.avg_pool(x)

        if not self.fully_conv:
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def adjust_input_image_size_for_proper_feature_alignment(input_img_batch, output_stride=8):
    """Resizes the input image to allow proper feature alignment during the
    forward propagation.
    To get more details, read here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py#L159
    """
    input_spatial_dims = np.asarray(input_img_batch.shape[2:], dtype=np.float)
    new_spatial_dims = np.ceil(input_spatial_dims / output_stride).astype(np.int) * output_stride + 1
    # Converting the numpy to list, torch.nn.functional.upsample_bilinear accepts
    # size in the list representation.
    new_spatial_dims = list(new_spatial_dims)
    input_img_batch_new_size = nn.functional.upsample_bilinear(input=input_img_batch,
                                                               size=new_spatial_dims)
    return input_img_batch_new_size


class ResNet34_8s(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, pretrained=False):
        super(ResNet34_8s, self).__init__()
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        self.model = ResNet(block=BasicBlock,
                            num_blocks=[3, 4, 6, 3],
                            inp_ch=in_channels,
                            fully_conv=True,
                            output_stride=8,
                            remove_avg_pool_layer=True)
        if pretrained:
            # restore network weights
            pass
        # Randomly initialize the 1x1 Conv scoring layer
        self.model.fc = nn.Conv2d(self.model.in_planes, out_channels, 1)
        self._normal_initialization(self.resnet34_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]

        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)

        x = self.model(x)
        x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode='bilinear', align_corners=False)
        return x
