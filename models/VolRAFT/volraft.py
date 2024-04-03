import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .volcorr import CorrBlock, AlternateCorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class ModuleVolRAFT(nn.Module):
    def __init__(self, args):
        super(ModuleVolRAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64

            # RAFT
            # self.args.corr_levels = 4
            # self.args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.args.corr_levels = 4
            self.args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(num_channels = self.args.num_channels, 
                                     output_dim=128, 
                                     norm_fn='instance', 
                                     dropout=args.dropout)        
            self.cnet = SmallEncoder(num_channels = self.args.num_channels, 
                                     output_dim=hdim+cdim, 
                                     norm_fn='none', 
                                     dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def initialize_flow(self, img, num_levels=4):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W, D = img.shape
        # coords0 = coords_grid(N, H//8, W//8, D//8, device=img.device)
        # coords1 = coords_grid(N, H//8, W//8, D//8, device=img.device)

        ht = np.ceil(H / 8)
        wd = np.ceil(W / 8)
        dp = np.ceil(D / 8)
        coords0 = coords_grid(N, ht, wd, dp, device=img.device)
        coords1 = coords_grid(N, ht, wd, dp, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)
    
    # normalize the range across both tensors
    # tensor structure should be: (Batch, Channel, Height, Width, Depth)
    def normalization(self, tensor0, tensor1):
        b0, c0, h0, w0, d0 = tensor0.shape
        b1, c1, h1, w1, d1 = tensor1.shape

        assert b0 == b1, f'batch size is not matching between {tensor0.shape} and {tensor1.shape}'
        assert c0 == c1, f'channel size is not matching between {tensor0.shape} and {tensor1.shape}'
        assert (h0 == h1) & (w0 == w1) & (d0 == d1), f'volume size is not matching between {tensor0.shape} and {tensor1.shape}'

        # Stack tensors to the last dimension
        tensor = torch.stack((tensor0, tensor1), dim = 5)

        # Compute the sd and mean at (height, width, depth) and the last dimension
        tensor_min = torch.amin(tensor, dim=(2, 3, 4, 5), keepdim = True)
        tensor_max = torch.amax(tensor, dim=(2, 3, 4, 5), keepdim = True)

        # Squeeze out the last dimension
        # Repeat in the height, width and depth dimensions
        tensor_min = tensor_min.squeeze(-1).repeat([1, 1, h0, w0, d0])
        tensor_max = tensor_max.squeeze(-1).repeat([1, 1, h0, w0, d0])

        # Normalize both tensors by the min and max to [0, 1]
        tensor0_norm = (tensor0 - tensor_min)
        tensor1_norm = (tensor1 - tensor_min)
        mask = (tensor_max - tensor_min > 0.0)
        tensor0_norm[mask] = tensor0_norm[mask] / (tensor_max - tensor_min)[mask]
        tensor1_norm[mask] = tensor1_norm[mask] / (tensor_max - tensor_min)[mask]

        tensor0_norm = tensor0_norm.contiguous()
        tensor1_norm = tensor1_norm.contiguous()

        return tensor0_norm, tensor1_norm
    
    def normalization_RAFT(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        return image1, image2

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # Do normalization
        # image1, image2 = self.normalization_RAFT(image1, image2)

        # Do normalization
        if self.args.should_normalize:
            image1, image2 = self.normalization(image1, image2)

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1, 
                                                num_levels=self.args.corr_levels)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0, desired_flow_shape = self.args.flow_shape)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
