import torch.nn as nn
import MyResNet
import torch
import torch.nn.functional as F
from collections import OrderedDict


class Model(nn.Module):

    def __init__(self, margin, norm_degree, embedding_dimensionality):
        super(Model, self).__init__()

        self.c = {}
        self.c['margin'] = margin
        self.c['norm_degree'] = norm_degree
        self.c['embedding_dimensionality'] = embedding_dimensionality

        self.cnn = MyResNet.resnet50(pretrained=True, num_classes=embedding_dimensionality)
        self.attributes_net = AttNet(embedding_dimensionality)
        self.initialize_weights()

    def forward(self, img, att_p, att_n):

        img = self.cnn(img)
        att_p = self.attributes_net(att_p)
        att_n = self.attributes_net(att_n)

        # L2 norm
        norm = img.norm(p=2, dim=1, keepdim=True)
        img = img.div(norm)
        norm = att_p.norm(p=2, dim=1, keepdim=True)
        att_p = att_p.div(norm)
        norm = att_n.norm(p=2, dim=1, keepdim=True)
        att_n = att_n.div(norm)

        # Check if triplet is already correct (not used for the loss, just for monitoring)
        correct = torch.zeros([1], dtype=torch.int32).cuda()
        d_i_att_p = F.pairwise_distance(img, att_p, p=self.c['norm_degree'])
        d_i_att_n = F.pairwise_distance(img, att_n, p=self.c['norm_degree'])

        for i in range(0,len(d_i_att_p)):
            if (d_i_att_n[i] - d_i_att_p[i]) > self.c['margin']:
                correct[0] += 1

        return img, att_p, att_n, correct


    def initialize_weights(self):
        for l in self.attributes_net.modules(): # Initialize only attributes_net weights
            if isinstance(l, nn.Conv2d):
                n = l.kernel_size[0] * l.kernel_size[1] * l.out_channels
                l.weight.data.normal_(0, math.sqrt(2. / n))
                if l.bias is not None:
                    l.bias.data.zero_()
            elif isinstance(l, nn.BatchNorm2d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                l.weight.data.normal_(0, 0.01)
                l.bias.data.zero_()

class AttModelTest(nn.Module):

    def __init__(self, embedding_dimensionality):
        super(AttModelTest, self).__init__()
        self.attributes_net = AttNet(embedding_dimensionality)

    def forward(self, att):
        att = self.attributes_net(att)
        norm = att.norm(p=2, dim=1, keepdim=True)
        att = att.div(norm)
        return att

class AttNet(nn.Module):

    def __init__(self, embedding_dimensionality):
        super(AttNet, self).__init__()
        self.fc_1 = BasicFC_BN(40, embedding_dimensionality)
        self.fc_2 = nn.Linear(embedding_dimensionality, embedding_dimensionality)

    def forward(self, att):
        att = self.fc_1(att)
        att = self.fc_2(att)

        return att

class ImgModelTest(nn.Module):

    def __init__(self, embedding_dimensionality):
        super(ImgModelTest, self).__init__()
        self.cnn = MyResNet.resnet50(pretrained=False, num_classes=embedding_dimensionality)

    def forward(self, img):
        img = self.cnn(img)
        norm = img.norm(p=2, dim=1, keepdim=True)
        img = img.div(norm)
        return img

class BasicFC_BN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)