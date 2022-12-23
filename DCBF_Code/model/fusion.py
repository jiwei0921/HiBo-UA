import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HolisticAttention import HA

# Channel Shuffle
def Channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x



class BMF(nn.Module):
    # Boundary-aware Multimodal Fusion Strategy
    def __init__(self):
        super(BMF, self).__init__()

        bottleneck_planes = 32 + 32
        scales = 4
        # conv2 (kernel=3)
        self.conv2_1_13 = nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales, kernel_size=(1, 3),
                                      padding=(0, 1))
        self.conv2_1_31 = nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales, kernel_size=(3, 1),
                                      padding=(1, 0))
        # conv3 (kernel=5)
        self.conv3_1_15 = nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales, kernel_size=(1, 5),
                                      padding=(0, 2))
        self.conv3_1_51 = nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales, kernel_size=(5, 1),
                                      padding=(2, 0))
        # conv4 (kernel=7)
        self.conv4_1_17 = nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales, kernel_size=(1, 7),
                                      padding=(0, 3))
        self.conv4_1_71 = nn.Conv2d(bottleneck_planes // scales, bottleneck_planes // scales, kernel_size=(7, 1),
                                      padding=(3, 0))

        self.conv3 = nn.Conv2d(bottleneck_planes, bottleneck_planes, kernel_size=1)
        self.cross_conv = nn.Conv2d(32*2, 32, 1, padding=0)

        self.rgb_conv = nn.Conv2d(32*2, 32, 1, padding=0)
        self.dep_conv = nn.Conv2d(32*2, 32, 1, padding=0)

        self.B_conv_3x3 = nn.Conv2d(32, 32, 3, padding=1)
        self.B_conv1_Sal = nn.Conv2d(32, 1, 1)
        self.sig = nn.Sigmoid()
        self.B_conv1_Edge= nn.Conv2d(32, 1, 1)

        self.fusion_layer = nn.Conv2d(32*2 + 2, 32, 1, padding=0)
        self.conv1_sal = nn.Conv2d(32, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x3_r,x3_d):


        # RGB & Depth multi-modal feature
        CR_fea3 = torch.cat([x3_r,x3_d],dim=1)  # H,W,2C (32+32)
        # Channel Shuffle
        CR_fea3_cs = Channel_shuffle(CR_fea3,4)
        CR_fea = self.conv3(CR_fea3_cs)

        # Grouping Dilated Convolution
        Group_scale = 4
        xs = torch.chunk(CR_fea, Group_scale, 1)
        final = []
        for s in range(Group_scale):
            if s == 0:
                # identity
                final.append(xs[s])
            elif s == 1:
                fea2 = self.conv2_1_13(xs[s])
                fea2 = self.conv2_1_31(fea2)
                final.append(fea2)
            elif s == 2:
                fea3 = self.conv3_1_15(xs[s])
                fea3 = self.conv3_1_51(fea3)
                final.append(fea3)
            else:
                fea4 = self.conv4_1_17(xs[s])
                fea4 = self.conv4_1_71(fea4)
                final.append(fea4)

        out = torch.cat(final, 1)

        Cross_fea = self.cross_conv(out)  # H, W, C=32

        RGB_feature = torch.cat([Cross_fea,x3_r],1)
        RGB_fea = self.rgb_conv(RGB_feature)

        Depth_feature = torch.cat([Cross_fea, x3_d], 1)
        Depth_fea = self.dep_conv(Depth_feature)



        '''Boundary-aware Strategy'''
        Content_fea3 = self.B_conv_3x3(Depth_fea)
        Sal_main_pred= self.B_conv1_Sal(Content_fea3)

        # Content Erasing
        Edge_fea3= RGB_fea * (1 - self.sig(Sal_main_pred))

        import matplotlib.pyplot as plt
        plt.figure()
        # plt.imshow(Edge_fea3.detach().numpy()[0][1])
        for i in range(1, 64+1):
            plt.subplot(2*8, 4, i)
            if 1<= i<=32:
                plt.imshow(Content_fea3.detach().numpy()[0][i-1],cmap='gray')
                plt.axis('off')
                plt.subplots_adjust(wspace=0.02,hspace=0.02)
            else:
                plt.imshow(Edge_fea3.detach().numpy()[0][i - 1 -32], cmap='gray')
                plt.axis('off')
                plt.subplots_adjust(wspace=0.02, hspace=0.02)
        plt.show()

        Edge_pred = self.B_conv1_Edge(Edge_fea3)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(torch.sigmoid(Edge_pred).detach().numpy()[0][0],cmap='gray')
        # plt.show()
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(torch.sigmoid(Sal_main_pred).detach().numpy()[0][0], cmap='gray')
        # plt.show()

        Multimodal = torch.cat([Content_fea3,Sal_main_pred,
                                    Edge_fea3,Edge_pred],dim=1)
        Multimodal_fea = self.fusion_layer(Multimodal)
        med_sal = self.conv1_sal(Multimodal_fea)


        return Multimodal_fea, Sal_main_pred, Edge_pred, med_sal



class fusion(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(fusion, self).__init__()

        channel = in_channel
        self.rfb3_1 = RFB(channel, channel)
        self.rfb4_1 = RFB(channel, channel)
        self.rfb5_1 = RFB(channel, channel)
        self.agg1 = aggregation(channel)

        self.rfb3_2 = RFB(channel, channel)
        self.rfb4_2 = RFB(channel, channel)
        self.rfb5_2 = RFB(channel, channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

        self.HA = HA()

        self.BMF3 = BMF()
        self.BMF4 = BMF()
        self.BMF5 = BMF()

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self._init_weight()



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x3_r,x4_r,x5_r,x3_d,x4_d,x5_d):

        # Boundary-aware Multi-modal Fusion Module
        x3, sal_main3, edge_main3, med_sal3 = self.BMF3(x3_r, x3_d)          # b_size,32, 1/8.  1/8    (44, 44)
        x4, sal_main4, edge_main4, med_sal4 = self.BMF4(x4_r, x4_d)          # b_size,32, 1/16. 1/16   (22, 22)
        x5, sal_main5, edge_main5, med_sal5 = self.BMF5(x5_r, x5_d)          # b_size,32, 1/32. 1/32   (11, 11)


        # Decoder
        x3_1 = self.rfb3_1(x3)
        x4_1 = self.rfb4_1(x4)
        x5_1 = self.rfb5_1(x5)
        attention_map = self.agg1(x5_1, x4_1, x3_1)
        x3_2 = self.HA(attention_map.sigmoid(), x3)
        x4_2 = self.conv4(x3_2)
        x5_2 = self.conv5(x4_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection_map = self.agg2(x5_2, x4_2, x3_2)

        return self.upsample(attention_map), self.upsample(detection_map), \
               [self.up8(sal_main3),self.up8(edge_main3)], \
               [self.up16(sal_main4),self.up16(edge_main4)], \
               [self.up32(sal_main5), self.up32(edge_main5)], \
               [self.up8(med_sal3), self.up16(med_sal4), self.up32(med_sal5)]

