import os
import pickle
import hashlib
import math

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.jit.annotations import Tuple
from torchvision.ops.deform_conv import deform_conv2d

from torch.nn.common_types import _size_2_t
from typing import Union


def equi_conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                cache_dir: str = '/tmp/cache/pano_detect/equi_conv_offset', ):
    """
    type: (Tensor, Tensor, Tensor, Optional[Tensor], Tuple[int, int], Tuple[int, int], Tuple[int, int]) -> Tensor

    Performs Equirectangular Convolution, described in Corners for Layout : End to End Layout Recovery from 360 Images

    Arguments:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]):
            convolution weights, split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1

    Returns:
        output (Tensor[batch_sz, out_channels, out_h, out_w]): result of convolution


    Examples::
        >>> input = torch.rand(1, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(5, 2 * kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([1, 5, 8, 8])
    """
    weight = weight.to(input.device)
    out_channels = weight.shape[0]
    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)
    else:
        bias = bias.to(input.device)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    bs, n_in_channels, in_h, in_w = input.shape

    pano_W = int((in_w + 2 * pad_w - dil_w * (weights_w - 1) - 1) // stride_w + 1)
    pano_H = int((in_h + 2 * pad_h - dil_h * (weights_h - 1) - 1) // stride_h + 1)

    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = torch.as_tensor(axis, device=input.device, dtype=input.dtype)
        axis = axis / math.sqrt(torch.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        ROT = torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], device=input.device, dtype=input.dtype)
        return ROT

    def equi_coord(pano_W, pano_H, k_W, k_H, u, v):
        fov_w = k_W * math.radians(360. / float(pano_W))
        focal = (float(k_W) / 2) / math.tan(fov_w / 2)
        c_x = 0
        c_y = 0

        u_r, v_r = u, v
        u_r, v_r = u_r - float(pano_W) / 2., v_r - float(pano_H) / 2.
        phi, theta = u_r / (pano_W) * (math.pi) * 2, -v_r / (pano_H) * (math.pi)

        ROT = rotation_matrix((0, 1, 0), phi)
        ROT = torch.matmul(ROT, rotation_matrix((1, 0, 0), theta))  # np.eye(3)

        h_range = torch.tensor(range(k_H), device=input.device, dtype=input.dtype)
        w_range = torch.tensor(range(k_W, ), device=input.device, dtype=input.dtype)
        w_ones = (torch.ones(k_W, device=input.device, dtype=input.dtype))
        h_ones = (torch.ones(k_H, device=input.device, dtype=input.dtype))
        h_grid = torch.matmul(torch.unsqueeze(h_range, -1), torch.unsqueeze(w_ones, 0)) + 0.5 - float(k_H) / 2
        w_grid = torch.matmul(torch.unsqueeze(h_ones, -1), torch.unsqueeze(w_range, 0)) + 0.5 - float(k_W) / 2

        K = torch.tensor([[focal, 0, c_x], [0, focal, c_y], [0., 0., 1.]], device=input.device, dtype=input.dtype)
        # inv_K = torch.inverse(K)
        inv_K = torch.from_numpy(
            np.linalg.inv(K.cpu().numpy().astype(np.float32))
        ).to(input.device).type(input.dtype)
        rays = torch.stack([w_grid, h_grid, torch.ones(h_grid.shape, device=input.device, dtype=input.dtype)], 0)
        rays = torch.matmul(inv_K, rays.reshape(3, k_H * k_W))
        rays /= torch.norm(rays, dim=0, keepdim=True)
        rays = torch.matmul(ROT, rays)
        rays = rays.reshape(3, k_H, k_W)

        phi = torch.atan2(rays[0, ...], rays[2, ...])
        theta = torch.asin(torch.clamp(rays[1, ...], -1, 1))
        x = (pano_W) / (2. * math.pi) * phi + float(pano_W) / 2.
        y = (pano_H) / (math.pi) * theta + float(pano_H) / 2.

        roi_y = h_grid + v_r + float(pano_H) / 2.
        roi_x = w_grid + u_r + float(pano_W) / 2.

        new_roi_y = (y)
        new_roi_x = (x)

        offsets_x = (new_roi_x - roi_x)
        offsets_y = (new_roi_y - roi_y)

        del ROT

        return offsets_x, offsets_y

    def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width=1, s_height=1, bs=16):
        offset = torch.zeros(2 * k_H * k_W, pano_H, pano_W, device=input.device, dtype=input.dtype)

        for v in range(0, pano_H, s_height):
            print('computing', v)
            for u in range(0, pano_W, s_width):
                offsets_x, offsets_y = equi_coord(pano_W, pano_H, k_W, k_H, u, v)
                offsets = torch.cat((torch.unsqueeze(offsets_y, -1), torch.unsqueeze(offsets_x, -1)), dim=-1)
                total_offsets = offsets.flatten()
                offset[:, v, u] = total_offsets
                del offsets_x
                del offsets_y
                del offsets
                del total_offsets

        offset = torch.unsqueeze(offset, 0)
        offset = torch.cat([offset for _ in range(bs)], dim=0)
        offset.requires_grad_(False)
        return offset

    offset_index = f'{bs}, {in_h}, {in_w}, {weights_h}, {weights_w}, {stride_h}, {stride_w}, {pad_h}, {pad_w}, {dil_h}, {dil_w}'
    print(f'offset: {offset_index}')
    offset_index = hashlib.sha256(offset_index.encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, offset_index)
    print(f'offset: {cache_file}')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if os.path.exists(cache_file):
        print(f'load offset from cache: {cache_file}')
        with open(cache_file, 'rb') as f:
            offset = torch.from_numpy(pickle.load(f)).to(input.device).type(input.dtype)
    else:
        print(f'compute offset and save to cache: {cache_file}')
        offset = distortion_aware_map(pano_W, pano_H, weights_w, weights_h,
                                      s_width=stride_w, s_height=stride_h, bs=bs)
        pickle.dump(offset.cpu().numpy(), open(cache_file, 'wb'))
        offset = offset.to(input.device).type(input.dtype)
        print(f'compute offset finished: {cache_file}')

    ret = deform_conv2d(input, offset, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
    print(f'deform_conv2d finished: {cache_file}')
    del offset
    return ret


class EquiConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
    ):
        super(EquiConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups,
                                            self.kernel_size[0], self.kernel_size[1]))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Arguments:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        return equi_conv2d(input, self.weight, self.bias, stride=self.stride,
                           padding=self.padding, dilation=self.dilation)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


def demo1():
    input = torch.ones(1, 1, 8, 8)
    conv = nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
    )
    conv.weight.data.fill_(1)  # [2, 1, 3, 3]
    b = conv(input)
    print(b.shape)

    offset = torch.ones(1, 2 * 3 * 3, 8, 8)
    c = deform_conv2d(
        input=input,
        offset=offset,
        weight=conv.weight,
        stride=1,
        padding=1,
        dilation=1,
        bias=conv.bias,
    )
    print(c.shape)


def demo2():
    from torch.nn import ZeroPad2d as zeropad

    # testing the EquiConv2d function
    a = torch.randn(2, 64, 32, 32)
    convol = EquiConv2d(64, 128, 3, padding=1, bias=False)
    # offset = torch.randn(2,18,32,32)
    b = convol(a)
    print(b.shape)

    # testing the equi_conv2d function
    # (1,1,4,4)
    in1 = torch.tensor([[[[1., 1., 2., 2.], [1., 1., 2., 2.], [3., 3., 4., 4.], [3., 3., 4., 4.]]]])
    # print(in1)
    # offset = torch.randn(1,8,4,4)
    samepad = zeropad((2, 2, 2, 2))
    # (1,1,8,8)
    in2 = samepad(in1)
    weights = torch.ones((1, 1, 2, 2))
    # print(weights)
    out1 = equi_conv2d(in2, weights, stride=2)
    print(out1.shape)
    # print(out1)


if __name__ == '__main__':
    # x = torch.rand(1, 1, 256, 256)
    # M = EquiConv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    # out = M(x)
    # print(out.shape)



    print(a1 == a2)
    print(hashlib.md5(a1).hexdigest())
    print(hashlib.md5(a2).hexdigest())
    # a = (torch.tensor(1), torch.tensor(40), torch.tensor(40), torch.tensor(3), torch.tensor(3), 2, 2, 1, 1, 1, 1,
    #      'project_dis: 1.000000, project_size: 3.000000, theta_rotate: 0.000000', 1.0)
    # for i in range(10):
    #     offset_index = pickle.dumps(a)
    #     print(offset_index)
    #     offset_index = hashlib.md5(offset_index).hexdigest()
    #     print(offset_index)
