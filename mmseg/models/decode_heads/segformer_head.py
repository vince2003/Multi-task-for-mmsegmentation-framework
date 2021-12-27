# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed
import pdb

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvMixer(nn.Module):
    def __init__(self,no_of_op_channels , depth , kernal , patch_size):
        super().__init__()
        self.o = no_of_op_channels
        self.d = depth
        self.k = kernal
        self.p = patch_size
        #self.n = output
        self.bn = nn.BatchNorm2d(self.o)
        self.cnn1 = nn.Conv2d(6 , self.o , (self.p , self.p) , stride=self.p)
        self.bn1 = nn.BatchNorm2d(self.o)
        self.cnn2 = nn.Conv2d(self.o , self.o , (self.k , self.k) , groups=self.o , padding=int((self.k-1)/2))
        self.bn2 = nn.BatchNorm2d(self.o)
        self.cnn3 = nn.Conv2d(self.o , self.o , (1,1))
        self.cnn_f = nn.Conv2d(self.o , 2 , (1,1))

    def forward(self , x):
        #pdb.set_trace()
        x = self.bn(F.gelu(self.cnn1(x)))
        
        for i in range(self.d):
          x = self.bn1(F.gelu(self.cnn2(x)))+ x #residual step and depthwise convolution
          x = self.bn2(F.gelu(self.cnn3(x))) #pointwise convolution        
        x = self.cnn_f(x)
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)    
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h
    
class SAM(nn.Module):
    def __init__(self, num_in=64, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        #x_anchor1 = self.priors(x_mask)
        #x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out    

@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    
    def __init__(self, **kwargs):
        super(SegFormerHead, self).__init__(**kwargs)
        #pdb.set_trace()
        
        num_classes=kwargs['num_classes']
        params_seg=dict(input_transform='multiple_select', in_index=[0, 1, 2, 3],\
                        dropout_ratio=0.1, in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32],\
                        num_classes=num_classes, decoder_params=768)
        self.SegHead = SegHead(**params_seg)


        num_classes=kwargs['num_classes']
        params_seg=dict(input_transform='multiple_select', in_index=[0, 1, 2, 3],\
                        dropout_ratio=0.1, in_channels=[64, 128, 320, 512], feature_strides=[4, 8, 16, 32],\
                        num_classes=num_classes, decoder_params=768)
        self.Skeleton = Skeleton(**params_seg)
        
        
        #self.cm=ConvMixer(6,10,kernal=3,patch_size=1)
        self.linear_seg = ConvModule(
            in_channels=64,
            out_channels=2,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )        
        
        self.linear_fuse = ConvModule(
            in_channels=64,
            out_channels=2,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )       

        self.dropout = nn.Dropout2d(0.1)
        
        
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()
        

    def forward(self, inputs):
        #pdb.set_trace()
        mul_feat, detail_feat=self.SegHead(inputs) #([2, 64, 128, 128]), ([2, 64, 128, 128])

        #pdb.set_trace()
        
        skeleton_feat, out_skeleton=self.Skeleton(inputs)#([2, 64, 128, 128]), torch.Size([2, 2, 512, 512])

        
        glb_sup=mul_feat#+skeleton_feat.detach()
        detail_sup=detail_feat+skeleton_feat.detach()
        #pdb.set_trace()  
        
        #CIM
        #att = self.ca(detail_sup) * detail_sup # channel attention
        #att = self.sa(att) * att # spatial attention 
        #detail_sup = detail_sup+att
        #----
        
        #out=torch.cat((glb_sup,detail_sup),1)
        # SAM
        out = self.SAM(glb_sup, detail_sup)
        
        glb_sup = self.dropout(glb_sup)
        out_seg=self.linear_seg(glb_sup)
        
        out = self.dropout(out)
        out=self.linear_fuse(out)
        #pdb.set_trace()
        return out, out_seg, out_skeleton
    

class SegHead(nn.Module):
    def __init__(self, input_transform, in_index, dropout_ratio, in_channels, feature_strides, num_classes, decoder_params):  
        super().__init__()
    
        #pdb.set_trace()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        
        self.input_transform = input_transform
        self.in_index = in_index
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None         
        

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        embedding_dim = decoder_params

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, 64, kernel_size=1)
        
        #self.conv_input = nn.Conv2d(3, num_classes, kernel_size=1)
        
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs        
        

    def forward(self, inputs):
        #pdb.set_trace()
        detail_feat = inputs[4]
        inputs=inputs[:4]
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        #x = self.dropout(_c)
        x = self.linear_pred(_c)
        
        #x = resize(x, size=input_org.size()[2:], mode='bilinear',align_corners=False)
        #pdb.set_trace()
        #input_conv = self.conv_input(input_org)
        #x = x.append(input_org)
        return x, detail_feat
    
    
class Skeleton(nn.Module):
    def __init__(self, input_transform, in_index, dropout_ratio, in_channels, feature_strides, num_classes, decoder_params):  
        super().__init__()
    
        #pdb.set_trace()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        
        self.input_transform = input_transform
        self.in_index = in_index
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None         
        

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        embedding_dim = decoder_params

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_skeleton = nn.Conv2d(embedding_dim, 64, kernel_size=1)
        self.linear_pred = nn.Conv2d(64, num_classes, kernel_size=1)
        
        #self.conv_input = nn.Conv2d(3, num_classes, kernel_size=1)
        
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs        
        

    def forward(self, inputs):
        #pdb.set_trace()
        #input_org=inputs[4]
        inputs=inputs[:4]
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        
        x = self.linear_skeleton(_c) 
        feat_skeleton = x.clone()
        
        x = self.dropout(x)
        
        x = self.linear_pred(x)
        
        #x = resize(x, size=[input_org.size()[2]*4,input_org.size()[3]*4], mode='bilinear',align_corners=False)
        #pdb.set_trace()
        #input_conv = self.conv_input(input_org)
        #x = x+input_conv
        #feat_skeleton.append(x)

        return feat_skeleton, x #[]


