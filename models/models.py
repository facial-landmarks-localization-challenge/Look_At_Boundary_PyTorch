import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import HeatmapLoss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(inplanes, planes, padding=0,
                               kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, padding=1,
                               kernel_size=3, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, padding=0,
                               kernel_size=1, stride=1, bias=False)
        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.Conv2d(inplanes, planes * self.expansion, padding=0,
                                   kernel_size=1, stride=stride, bias=False)
        self.downsample = downsample

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            out = self.conv1(x)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv2(out)
            out = self.bn3(out)
            out = self.relu3(out)
            out = self.conv3(out)
        else:
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv2(out)
            out = self.bn3(out)
            out = self.relu3(out)
            out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class Hourglass(nn.Module):

    def __init__(self, block=Bottleneck, num_blocks=1, planes=64, depth=4):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.hg = self._make_hourglass(block, num_blocks, planes, depth)

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    @staticmethod
    def _make_residual(block, num_blocks, planes):
        layers = []
        for index in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hourglass(self, block, num_blocks, planes, depth):
        hourglass = []
        for index in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if index == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hourglass.append(nn.ModuleList(res))
        return nn.ModuleList(hourglass)

    def _hourglass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = self.maxpool(x)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hourglass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_forward(self.depth, x)


class FMFHourglass(nn.Module):

    def __init__(self, planes, depth):
        super(FMFHourglass, self).__init__()
        self.depth = depth
        self.maxpool = nn.MaxPool2d(2, stride=2)
        hourglass = []
        for index in range(depth):
            res = []
            for j in range(3):
                res.append(Bottleneck(planes * Bottleneck.expansion, planes))
            if index == depth - 1:
                del(res[-1])
            hourglass.append(nn.ModuleList(res))
        self.hg = nn.ModuleList(hourglass)

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    def _hourglass_forward(self, n, x):
        up1 = self.hg[n - 1][2](x)
        low1 = self.maxpool(x)
        low1 = self.hg[n - 1][0](low1)

        if n > 1:
            low2 = self._hourglass_forward(n - 1, low1)
            low2 = self.hg[n - 1][1](low2)
        else:
            low2 = self.hg[n - 1][1](low1)
        up2 = F.interpolate(low2, scale_factor=2, mode='bilinear', align_corners=True)
        out = up1 + up2
        return out

    def forward(self, x):
        out = self.maxpool(x)
        out = self.hg[self.depth-1][0](out)
        if self.depth > 1:
            out = self._hourglass_forward(self.depth-1, out)
        out = self.hg[self.depth-1][1](out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out


class MessagePassing(nn.Module):
    pass_order = {'A': ['1', '13', '12', '11', '10', '5', '4', '7', '9', '6', '8', '2', '3'],
                  'B': ['2', '3', '6', '8', '7', '9', '4', '5', '10', '11', '12', '13', '1']}
    boundary_relation = {'A': {'1': ['2', '3', '7', '9', '13'],
                               '2': [],
                               '3': [],
                               '4': ['7', '9'],
                               '5': ['4'],
                               '6': ['2'],
                               '7': ['6'],
                               '8': ['3'],
                               '9': ['8'],
                               '10': ['5'],
                               '11': ['10'],
                               '12': ['11'],
                               '13': ['12']},
                         'B': {'1': [],
                               '2': ['1', '6'],
                               '3': ['1', '8'],
                               '4': ['5'],
                               '5': ['10'],
                               '6': ['7'],
                               '7': ['1', '4'],
                               '8': ['9'],
                               '9': ['1', '4'],
                               '10': ['11'],
                               '11': ['12'],
                               '12': ['13'],
                               '13': ['1']}}

    def __init__(self, classes=13, step=2, inchannels=256, channels=16, first=0, last=0):
        super(MessagePassing, self).__init__()
        self.first = first       # 标识当前次message passing是否是第一次message passing, 1表示是第一次
        self.last = last         # 标识当前次message passing是否是最后一次message passing, 1表示是最后一次
        self.classes = classes   # boundary number: 13
        self.step = step         # message passing steps: 2
        prepare_conv, prepare_bn, prepare_relu = [], [], []
        after_bn, after_relu, after_conv = [], [], []
        inner_level_pass, inter_level_pass = [], []
        for index in range(2 * classes):
            prepare_conv.append(nn.Conv2d(inchannels, channels, padding=0,
                                          kernel_size=1, stride=1, bias=False))
            prepare_bn.append(nn.BatchNorm2d(channels))
            prepare_relu.append(nn.ReLU())
        for index in range(classes):
            after_bn.append(nn.BatchNorm2d(2*channels))
            after_relu.append(nn.ReLU())
            after_conv.append(nn.Conv2d(2*channels, 1, padding=0,
                                        kernel_size=1, stride=1, bias=False))
        for item in self.pass_order['A']:
            for index in range(len(self.boundary_relation['A'][item])):
                inner_level_pass.append(self._make_passing())
        for item in self.pass_order['B']:
            for index in range(len(self.boundary_relation['B'][item])):
                inner_level_pass.append(self._make_passing())
        if self.last == 0:
            for index in range(2*self.classes):
                inter_level_pass.append(self._make_passing())
        self.pre_conv = nn.ModuleList(prepare_conv)
        self.pre_bn = nn.ModuleList(prepare_bn)
        self.pre_relu = nn.ModuleList(prepare_relu)
        self.aft_bn = nn.ModuleList(after_bn)
        self.aft_relu = nn.ModuleList(after_relu)
        self.aft_conv = nn.ModuleList(after_conv)
        self.inner_pass = nn.ModuleList(inner_level_pass)
        self.inter_pass = nn.ModuleList(inter_level_pass)

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    def _make_passing(self, inplanes=16, planes=8, pad=3, ker_size=7, stride=1, bias=False):
        passing = []
        for pass_step in range(self.step):
            if pass_step == 0:
                passing.append(nn.Conv2d(inplanes, planes, padding=pad,
                                         kernel_size=ker_size, stride=stride, bias=bias))
                passing.append(nn.BatchNorm2d(planes))
                passing.append(nn.ReLU())
            elif pass_step == self.step - 1:
                passing.append(nn.Conv2d(planes, inplanes, padding=pad,
                                         kernel_size=ker_size, stride=stride, bias=bias))
            else:
                passing.append(nn.Conv2d(planes, planes, padding=pad,
                                         kernel_size=ker_size, stride=stride, bias=bias))
                passing.append(nn.BatchNorm2d(planes))
                passing.append(nn.ReLU())
        return nn.Sequential(*passing)

    def forward(self, x, ahead_msg):
        inner_msg_count = 0
        feature_map = []
        result = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                  '10': [], '11': [], '12': [], '13': []}
        result_a = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                    '10': [], '11': [], '12': [], '13': []}
        result_b = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                    '10': [], '11': [], '12': [], '13': []}
        msg_box_a = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                     '10': [], '11': [], '12': [], '13': []}
        msg_box_b = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                     '10': [], '11': [], '12': [], '13': []}
        inter_level_msg = {'A': {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [],
                                 '9': [], '10': [], '11': [], '12': [], '13': []},
                           'B': {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [],
                                 '9': [], '10': [], '11': [], '12': [], '13': []}}

        for index in range(self.classes):  # direction 'A'
            out = self.pre_conv[index](x)
            for get_msg_index in range(len(msg_box_a[self.pass_order['A'][index]])):  # get inner level msg
                out = out + msg_box_a[self.pass_order['A'][index]][get_msg_index]
            if self.first == 0:  # 即不是第一次message passing, get inter level msg
                out = out + ahead_msg['A'][self.pass_order['A'][index]][0]
            out = self.pre_bn[index](out)
            out = self.pre_relu[index](out)
            result_a[self.pass_order['A'][index]].append(out)  # save to be concatenated
            for send_msg_index in range(len(self.boundary_relation['A'][self.pass_order['A'][index]])):  # message pass
                temp = self.inner_pass[inner_msg_count](out)
                inner_msg_count = inner_msg_count + 1
                msg_box_a[self.boundary_relation['A'][self.pass_order['A'][index]][send_msg_index]].append(temp)
            if self.last == 0:  # 即不是最后一次message passing，则向下一个stack传递消息
                temp = self.inter_pass[index](out)
                inter_level_msg['A'][self.pass_order['A'][index]].append(temp)

        for index in range(self.classes):  # direction 'B'
            out = self.pre_conv[index + self.classes](x)
            for get_msg_index in range(len(msg_box_b[self.pass_order['B'][index]])):  # get inner level msg
                out = out + msg_box_b[self.pass_order['B'][index]][get_msg_index]
            if self.first == 0:  # 即不是第一次message passing, get inter level msg
                out = out + ahead_msg['B'][self.pass_order['B'][index]][0]
            out = self.pre_bn[index + self.classes](out)
            out = self.pre_relu[index + self.classes](out)
            result_b[self.pass_order['B'][index]].append(out)  # save to be concatenated
            for send_msg_index in range(len(self.boundary_relation['B'][self.pass_order['B'][index]])):  # message pass
                temp = self.inner_pass[inner_msg_count](out)
                inner_msg_count = inner_msg_count + 1
                msg_box_b[self.boundary_relation['B'][self.pass_order['B'][index]][send_msg_index]].append(temp)
            if self.last == 0:  # 即不是最后一次message passing，则向下一个stack传递消息
                temp = self.inter_pass[index + self.classes](out)
                inter_level_msg['B'][self.pass_order['B'][index]].append(temp)

        for index in range(self.classes):   # concatenation and conv to get feature_map
            result[str(index + 1)] = torch.cat((result_a[str(index + 1)][0],
                                                result_b[str(index + 1)][0]), 1)  # after concat: 1 32 64 64
            result[str(index + 1)] = self.aft_bn[index](result[str(index + 1)])
            result[str(index + 1)] = self.aft_relu[index](result[str(index + 1)])
            result[str(index + 1)] = self.aft_conv[index](result[str(index + 1)])

        feature_map.append(result['1'])
        for index in range(self.classes - 1):   # concat all 'classes' feature maps
            feature_map[0] = torch.cat((feature_map[0], result[str(index + 2)]), 1)

        if self.last == 0:  # 如果不是最后一个stack的message passing，则除了输出feature map外还输出层间消息
            return feature_map[0], inter_level_msg
        else:
            return feature_map[0]


class Estimator(nn.Module):

    def __init__(self, stacks=4, msg_pass=1):
        super(Estimator, self).__init__()
        self.stacks = stacks
        self.msg_pass = msg_pass
        self.hm_loss = HeatmapLoss()
        self.conv1 = nn.Conv2d(1, 64, padding=3, kernel_size=7,
                               stride=2, bias=False)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_relu = nn.ReLU(inplace=False)
        self.pre_res_1 = Bottleneck(64, 32)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)  # problem, need to see the source code of caffe
        self.pre_res_2 = Bottleneck(128, 32)
        self.pre_res_2_bn = nn.BatchNorm2d(128)
        self.pre_res_2_relu = nn.ReLU(inplace=False)
        self.hourglass_0 = Bottleneck(128, 64)
        hg, mp = [], []
        linear_1_res, linear_1_bn, linear_1_relu, linear_1_conv = [], [], [], []
        linear_2_bn, linear_2_relu, linear_2_conv = [], [], []
        linear_3 = []
        linear_mp_bn, linear_mp_relu, linear_mp_conv = [], [], []
        for index in range(self.stacks):
            hg.append(Hourglass())
            linear_1_res.append(Bottleneck(256, 64))
            linear_1_bn.append(nn.BatchNorm2d(256))
            linear_1_relu.append(nn.ReLU())
            linear_1_conv.append(nn.Conv2d(256, 256, padding=0, kernel_size=1,
                                           stride=1, bias=False))
            if msg_pass:
                if index == 0:
                    mp.append(MessagePassing(first=1))
                elif index == self.stacks - 1:
                    mp.append(MessagePassing(last=1))
                else:
                    mp.append(MessagePassing())
            else:
                linear_mp_bn.append(nn.BatchNorm2d(256))
                linear_mp_relu.append(nn.ReLU())
                linear_mp_conv.append(nn.Conv2d(256, 13, padding=0, kernel_size=1,
                                                stride=1, bias=False))
            if index != self.stacks - 1:
                linear_2_bn.append(nn.BatchNorm2d(256))
                linear_2_relu.append(nn.ReLU())
                linear_2_conv.append(nn.Conv2d(256, 256, padding=0, kernel_size=1,
                                     stride=1, bias=False))
                linear_3.append(nn.Conv2d(13, 256, padding=0, kernel_size=1,
                                          stride=1, bias=False))
        self.hg = nn.ModuleList(hg)
        self.linear_1_res = nn.ModuleList(linear_1_res)
        self.linear_1_bn = nn.ModuleList(linear_1_bn)
        self.linear_1_relu = nn.ModuleList(linear_1_relu)
        self.linear_1_conv = nn.ModuleList(linear_1_conv)
        self.mp = nn.ModuleList(mp)
        self.linear_2_bn = nn.ModuleList(linear_2_bn)
        self.linear_2_relu = nn.ModuleList(linear_2_relu)
        self.linear_2_conv = nn.ModuleList(linear_2_conv)
        self.linear_3 = nn.ModuleList(linear_3)
        self.linear_mp_bn = nn.ModuleList(linear_mp_bn)
        self.linear_mp_relu = nn.ModuleList(linear_mp_relu)
        self.linear_mp_conv = nn.ModuleList(linear_mp_conv)

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        heatmaps = []         # save all the stacks output feature maps
        inter_level_msg = []
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.conv1_relu(out)
        out = self.pre_res_1(out)
        out = self.pool1(out)
        out = self.pre_res_2(out)
        out = self.pre_res_2_bn(out)
        out = self.pre_res_2_relu(out)
        out = self.hourglass_0(out)
        for index in range(self.stacks):
            temp = self.hg[index](out)
            temp = self.linear_1_res[index](temp)
            temp = self.linear_1_bn[index](temp)
            temp = self.linear_1_relu[index](temp)
            temp = self.linear_1_conv[index](temp)
            if self.msg_pass:
                if index != self.stacks - 1:
                    heatmap, inter_level_msg = self.mp[index](temp, inter_level_msg)
                else:
                    heatmap = self.mp[index](temp, inter_level_msg)
            else:
                heatmap = self.linear_mp_bn[index](temp)
                heatmap = self.linear_mp_relu[index](heatmap)
                heatmap = self.linear_mp_conv[index](heatmap)
            heatmaps.append(heatmap)
            if index != self.stacks - 1:
                temp = self.linear_2_bn[index](temp)
                temp = self.linear_2_relu[index](temp)
                linear2_out = self.linear_2_conv[index](temp)
                linear3_out = self.linear_3[index](heatmap)
                out = out + linear2_out + linear3_out
        return heatmaps  # 每一个stack的输出heatmap经过append

    def calc_loss(self, pred_heatmaps, gt_heatmap):
        heatmap_loss = []
        for stack in range(self.stacks):
            heatmap_loss.append(self.hm_loss(pred_heatmaps[stack], gt_heatmap))
        heatmap_loss = torch.stack(heatmap_loss, dim=0)
        heatmap_loss = torch.sum(heatmap_loss)
        return heatmap_loss


class Regressor(nn.Module):

    def __init__(self, classes=13, fuse_stages=4, planes=16, output=196):
        super(Regressor, self).__init__()
        self.classes = classes
        self.FMF_stages = 3
        self.fuse_stages = fuse_stages
        self.planes = planes
        self.conv1 = nn.Conv2d(14, self.planes, padding=3, kernel_size=7, stride=2, bias=False) \
            if fuse_stages > 0 else nn.Conv2d(1, self.planes, padding=3, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.bn2 = nn.BatchNorm2d(256)  # regressor最后一个Batchnorm
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)           # regressor ip之前最后一个relu
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)  # problem, need to see the source code of caffe (solved)
        baseline_bn, baseline_relu, baseline_res_1, baseline_res_2 = [], [], [], []
        pre_fmf_bn, pre_fmf_relu, pre_fmf_conv = [], [], []
        aft_fmf_bn, aft_fmf_relu, aft_fmf_conv = [], [], []
        tanh = []
        fmfhourglass = []
        for index in range(self.FMF_stages + 1):
            if index == 0:
                baseline_bn.append(nn.BatchNorm2d(self.planes))
                baseline_relu.append(nn.ReLU())
                baseline_res_1.append(Bottleneck(self.planes, self.planes//2))
                baseline_res_2.append(Bottleneck(self.planes * 2, self.planes//2))
            else:
                baseline_bn.append(nn.BatchNorm2d(self.planes * pow(2, index)))
                baseline_relu.append(nn.ReLU())
                baseline_res_1.append(Bottleneck(self.planes * pow(2, index), self.planes * pow(2, index-1), stride=2))
                baseline_res_2.append(Bottleneck(self.planes * pow(2, index+1), self.planes * pow(2, index-1)))
        for index in range(self.FMF_stages):
            pre_fmf_bn.append(nn.BatchNorm2d(self.planes * pow(2, index+1) + self.classes))
            pre_fmf_relu.append(nn.ReLU())
            pre_fmf_conv.append(nn.Conv2d(self.planes*pow(2, index+1) + self.classes, self.planes*pow(2, index+1),
                                          padding=0, kernel_size=1, stride=1, bias=False))
        for index in range(self.FMF_stages):
            fmfhourglass.append(FMFHourglass(planes=8*pow(2, index), depth=3-index))
        for index in range(self.FMF_stages):
            aft_fmf_bn.append(nn.BatchNorm2d(self.planes * pow(2, index + 1)))
            aft_fmf_bn.append(nn.BatchNorm2d(self.planes * pow(2, index + 1)))
            aft_fmf_relu.append(nn.ReLU())
            aft_fmf_relu.append(nn.ReLU())
            aft_fmf_conv.append(nn.Conv2d(self.planes * pow(2, index + 1), self.planes * pow(2, index + 1),
                                          padding=0, kernel_size=1, stride=1, bias=False))
            aft_fmf_conv.append(nn.Conv2d(self.planes * pow(2, index + 1), self.planes * pow(2, index + 1),
                                          padding=0, kernel_size=1, stride=1, bias=False))
            tanh.append(nn.Tanh())
        self.bl_bn = nn.ModuleList(baseline_bn)
        self.bl_relu = nn.ModuleList(baseline_relu)
        self.bl_res_1 = nn.ModuleList(baseline_res_1)
        self.bl_res_2 = nn.ModuleList(baseline_res_2)
        self.pre_fmf_bn = nn.ModuleList(pre_fmf_bn)
        self.pre_fmf_relu = nn.ModuleList(pre_fmf_relu)
        self.pre_fmf_conv = nn.ModuleList(pre_fmf_conv)
        self.FMF_Hourglass = nn.ModuleList(fmfhourglass)
        self.aft_fmf_bn = nn.ModuleList(aft_fmf_bn)
        self.aft_fmf_relu = nn.ModuleList(aft_fmf_relu)
        self.aft_fmf_conv = nn.ModuleList(aft_fmf_conv)
        self.tanh = nn.ModuleList(tanh)
        self.fc1 = nn.Linear(256 * 8 * 8, 256)  # 目前的代码暂时不考虑通用性，很多数字暂时都强硬地固定下来了
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output)
        self.fc_relu1 = nn.ReLU(inplace=False)
        self.fc_relu2 = nn.ReLU(inplace=False)

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, input_img, heatmap):
        data_concat = []
        if self.fuse_stages > 0:
            out = F.interpolate(heatmap, scale_factor=4, mode='bilinear', align_corners=True)
            data_concat.append(input_img)
            for index in range(self.classes - 1):
                data_concat[0] = torch.cat((data_concat[0], input_img), 1)
            out = data_concat[0]*out
            out = torch.cat((out, input_img), 1)
        else:
            out = input_img
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.bl_bn[0](out)
        out = self.bl_relu[0](out)
        out = self.bl_res_1[0](out)
        out = self.bl_res_2[0](out)
        for index in range(self.FMF_stages):
            if index < self.fuse_stages - 1:
                temp = F.interpolate(heatmap, scale_factor=pow(2, -1*index), mode='bilinear', align_corners=True)
                temp_out = torch.cat((temp, out), 1)
                temp_out = self.pre_fmf_bn[index](temp_out)
                temp_out = self.pre_fmf_relu[index](temp_out)
                temp_out = self.pre_fmf_conv[index](temp_out)
                temp_out = self.FMF_Hourglass[index](temp_out)
                temp_out = self.aft_fmf_bn[2 * index](temp_out)
                temp_out = self.aft_fmf_relu[2 * index](temp_out)
                temp_out = self.aft_fmf_conv[2 * index](temp_out)
                temp_out = self.aft_fmf_bn[2 * index + 1](temp_out)
                temp_out = self.aft_fmf_relu[2 * index + 1](temp_out)
                temp_out = self.aft_fmf_conv[2 * index + 1](temp_out)
                temp_out = self.tanh[index](temp_out)
                temp_out = temp_out * out
                out = temp_out + out
            out = self.bl_bn[index+1](out)
            out = self.bl_relu[index + 1](out)
            out = self.bl_res_1[index + 1](out)
            out = self.bl_res_2[index + 1](out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc1(out)
        out = self.fc_relu1(out)
        out = self.fc2(out)
        out = self.fc_relu2(out)
        out = self.fc3(out)

        return out


class Discrim(nn.Module):
    channels, linear_n = [13, 64, 192, 384, 256, 256], [4096, 1024, 256, 13]
    ker_size, strd, pad = [2, 5, 3, 3, 3], [2, 1, 1, 1, 1], [0, 2, 1, 1, 1]
    maxpool_mask = [1, 1, 0, 0, 1]

    def __init__(self, conv_layers=5, linear_layers=3):
        super(Discrim, self).__init__()
        conv_features = []
        linear_classify = []
        for index in range(conv_layers):
            conv_features.append(nn.Conv2d(Discrim.channels[index], Discrim.channels[index + 1],
                                           kernel_size=Discrim.ker_size[index],
                                           stride=Discrim.strd[index],
                                           padding=Discrim.pad[index],
                                           bias=False))
            conv_features.append(nn.BatchNorm2d(Discrim.channels[index + 1]))
            conv_features.append(nn.ReLU(inplace=False))
            if Discrim.maxpool_mask[index] == 1:
                conv_features.append(nn.MaxPool2d(3, stride=2, padding=1))
            else:
                conv_features.append(nn.ReLU(inplace=False))
        for index in range(linear_layers):
            linear_classify.append(nn.Linear(Discrim.linear_n[index], Discrim.linear_n[index+1]))
            if index != linear_layers - 1:
                linear_classify.append(nn.ReLU(inplace=False))
            else:
                linear_classify.append(nn.Sigmoid())
        self.features = nn.Sequential(*conv_features)
        self.classifier = nn.Sequential(*linear_classify)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, self.num_flat_features(out))
        out = self.classifier(out)
        return out

