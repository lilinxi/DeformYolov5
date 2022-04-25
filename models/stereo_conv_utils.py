import os
import logging
import math
import pickle
import hashlib

import cv2
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.jit.annotations import Optional, Tuple
from torchvision.ops.deform_conv import deform_conv2d

import proto_gen.detect_pb2
import proj.transform
import proj.stereo_proj
import proj.stereo_conv_util

from torch.nn.common_types import _size_2_t
from typing import Union

"""
采用 αw  和 αh 参数化感受野。因此，直接定义视场上的卷积。核沿着球体旋转并施加，其位置由其中心的球坐标 (图中为φ和θ)定义。

与标准的内核不同，标准内核是由它们的大小 kw × kh 参数化的，而在 EquiConvs，定义了角度大小 (αw × αh) 和 分辨率 (rw × rh)。在实际操作中，我们保留长宽比，αw / rw = αh / rh，并使用正方形 kernel，所以从现在开始我们将视场表示为α (αw = αh)，分辨率表示为 r (rw = rh) 。
"""


def compute_kernel_offset(
        output_X_normal, output_Y_normal,
        out_width, out_height,
        kernel_height, kernel_width,
        dil_h, dil_w,
        proj_params,
        thetaRate,
        device, dtype,
        absolute=False,
):
    proj_params.theta_rotate = 0  # 不考虑 theta_rotate
    assert kernel_height == kernel_width, 'kernel_height must be equal to kernel_width'
    assert dil_h == dil_w, 'dil_h must be equal to dil_w'

    theta_min, phi_min, theta_range, phi_range = proj.stereo_conv_util.get_theta_phi_range(proj_params)
    delta_theta = theta_range / out_width * thetaRate
    delta_phi = phi_range / out_height * thetaRate

    kernel_center = ((kernel_height - 1) * dil_h + 1) // 2
    # (kernel_height=0, kernel_width=2) 是右上角
    kernel_offset_x = torch.zeros(kernel_height, kernel_width, device=device, dtype=dtype)
    kernel_offset_y = torch.zeros(kernel_height, kernel_width, device=device, dtype=dtype)

    # 计算卷积核中心的位置
    raw_x, raw_y, raw_z = proj.stereo_proj._projXY2panoXYZ(output_X_normal, output_Y_normal, proj_params=proj_params)
    center_theta, center_phi = proj.transform.xyz2theta_phi(raw_x, raw_y, raw_z)

    # 计算整个卷积核的偏移量
    for i in range(kernel_height):
        for j in range(kernel_width):
            raw_kernel_h = i * dil_h - kernel_center
            raw_kernel_w = j * dil_w - kernel_center

            offset_theta = center_theta + raw_kernel_w * delta_theta
            offset_phi = center_phi + raw_kernel_h * delta_phi

            offset_x, offset_y, offset_z = proj.transform.theta_phi2xyz(offset_theta, offset_phi)
            offset_X, offset_Y, _ = proj.stereo_proj._panoXYZ2projXY(offset_x, offset_y, offset_z, proj_params=proj_params)

            if absolute:  # 绝对偏移量，相对于中心
                kernel_offset_x[i, j] = (offset_X - output_X_normal) * out_width
                kernel_offset_y[i, j] = (offset_Y - output_Y_normal) * out_height
            else:  # 相对于标准卷积核的偏移量
                kernel_offset_x[i, j] = (offset_X - output_X_normal) * out_width - raw_kernel_w
                kernel_offset_y[i, j] = (offset_Y - output_Y_normal) * out_height - raw_kernel_h

    return kernel_offset_x, kernel_offset_y


# 可视化立体卷积核
def plot_kernel_offset(proj_req: proto_gen.detect_pb2.StereoProjectRequest):
    im_raw = cv2.imread(proj_req.project_request.pano_dataset_model.image_path)  # BGR
    im = proj.stereo_proj.stereo_proj(im_raw, proj_req.project_params_list[0],
                                      proj_req.project_request.project_width, proj_req.project_request.project_height)
    X = proj_req.project_params_list[0].project_size * 1 / 9
    Y = proj_req.project_params_list[0].project_size * 1 / 9

    pX = X * proj_req.project_request.project_width / proj_req.project_params_list[0].project_size
    pY = Y * proj_req.project_request.project_height / proj_req.project_params_list[0].project_size

    demo_kernel_x, demo_kernel_y = compute_kernel_offset(
        output_X_normal=X, output_Y_normal=Y,
        out_width=proj_req.project_request.project_width, out_height=proj_req.project_request.project_height,
        kernel_height=7, kernel_width=7,
        dil_h=1, dil_w=1,
        proj_params=proj_req.project_params_list[0],
        thetaRate=5.0,
        device='cpu', dtype=torch.float32,
        absolute=True,
    )
    for i in range(49):
        x = demo_kernel_x.flatten()[i]
        y = demo_kernel_y.flatten()[i]
        px = round(pX + x.item())
        py = round(pY + y.item())
        print(pX, '+', x.item(), '=', px, '\t', pY, '+', y.item(), '=', py)
        cv2.circle(im, (px, py), 3, (0, 255, 0), -1)
    print("===========================================")

    demo_kernel_x, demo_kernel_y = compute_kernel_offset(
        output_X_normal=X, output_Y_normal=Y,
        out_width=proj_req.project_request.project_width, out_height=proj_req.project_request.project_height,
        kernel_height=3, kernel_width=3,
        dil_h=2, dil_w=2,
        proj_params=proj_req.project_params_list[0],
        thetaRate=1.0,
        device='cpu', dtype=torch.float32,
        absolute=True,
    )
    for i in range(3 * 3):
        x = demo_kernel_x.flatten()[i]
        y = demo_kernel_y.flatten()[i]
        px = round(pX + x.item())
        py = round(pY + y.item())
        print(pX, '+', x.item(), '=', px, '\t', pY, '+', y.item(), '=', py)
        cv2.circle(im, (px, py), 3, (0, 0, 255), -1)
    print("===========================================")

    demo_kernel_x, demo_kernel_y = compute_kernel_offset(
        output_X_normal=X, output_Y_normal=Y,
        out_width=proj_req.project_request.project_width, out_height=proj_req.project_request.project_height,
        kernel_height=3, kernel_width=3,
        dil_h=3, dil_w=3,
        proj_params=proj_req.project_params_list[0],
        thetaRate=1.0,
        device='cpu', dtype=torch.float32,
        absolute=True,
    )
    for i in range(3 * 3):
        x = demo_kernel_x.flatten()[i]
        y = demo_kernel_y.flatten()[i]
        px = round(pX + x.item())
        py = round(pY + y.item())
        print(pX, '+', x.item(), '=', px, '\t', pY, '+', y.item(), '=', py)
        cv2.circle(im, (px, py), 3, (255, 0, 0), -1)
    print("===========================================")

    cv2.imshow('im_raw', im_raw)
    cv2.imshow('delta_theta_phi', im)
    cv2.waitKey(0)

@profile
def stereo_conv2d(
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        mask: Optional[Tensor] = None,
        proj_params: proto_gen.detect_pb2.StereoProjectParams = None,
        thetaRate: float = 1.0,  # 默认 detTheta = proj_theta / proj_size, 可调整 detTheta = detTheta * thetaRate
        cache_dir: str = '/tmp/cache/pano_detect/stereo_conv_offset',
):
    out_channels = weight.shape[0]

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)
    else:
        bias = bias.to(input.device)

    stride_h, stride_w = _pair(stride)
    # if stride_h != stride_w or stride_h != 1:
    #     raise ValueError("stride_h must be 1 and equals stride_w, not support yet")
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    # if dil_h != dil_w or dil_h != 1:
    #     raise ValueError("dil_h must be 1 and equals dil_w, not support yet")
    kernel_height, kernel_width = weight.shape[-2:]  # [output_channels, input_channels, kernel_h, kernel_w]
    # if kernel_height != kernel_width or kernel_height != 3:
    #     raise ValueError("kernel_height must be 3 and equals kernel_width, not support yet")
    bs, n_in_channels, in_h, in_w = input.shape  # [batch_size, input_channels, input_height, input_width]

    out_width = int((in_w + 2 * pad_w - dil_w * (kernel_width - 1) - 1) // stride_w + 1)
    out_height = int((in_h + 2 * pad_h - dil_h * (kernel_height - 1) - 1) // stride_h + 1)

    # offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width])
    offset = torch.zeros(2 * kernel_height * kernel_width, out_height, out_width,
                         device=input.device, dtype=input.dtype)

    def compute_offset(offset, bs):
        for dh in range(0, out_height, stride_h):
            for dw in range(0, out_width, stride_w):
                logging.debug(f'{dh}, {dw}')
                offsets_x, offsets_y = compute_kernel_offset(
                    output_X_normal=dw / in_w * proj_params.project_size,
                    output_Y_normal=dh / in_h * proj_params.project_size,
                    out_width=out_width, out_height=out_height,
                    kernel_height=kernel_height, kernel_width=kernel_width,
                    dil_h=dil_h, dil_w=dil_w,
                    proj_params=proj_params,
                    thetaRate=thetaRate,
                    device=input.device, dtype=input.dtype,
                )
                offsets = torch.cat((torch.unsqueeze(offsets_y, -1), torch.unsqueeze(offsets_x, -1)), dim=-1)
                total_offsets = offsets.flatten()
                offset[:, dh, dw] = total_offsets
        offset = torch.unsqueeze(offset, 0)
        offset = torch.cat([offset for _ in range(bs)], dim=0)
        offset.requires_grad_(False)
        return offset

    offset_index_tuple = (
        bs,
        in_h, in_w,
        kernel_height, kernel_width,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        proj_params.SerializeToString(),
        thetaRate,
    )
    offset_index = pickle.dumps(offset_index_tuple)
    offset_index = hashlib.sha256(offset_index).hexdigest()
    cache_file = os.path.join(cache_dir, offset_index)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if os.path.exists(cache_file):
        print(f'load offset from cache: {cache_file}')
        with open(cache_file, 'rb') as f:
            offset = torch.from_numpy(pickle.load(f)).to(input.device).type(input.dtype)
    else:
        print(f'compute offset and save to cache: {cache_file}')
        offset = compute_offset(offset, bs)
        pickle.dump(offset.cpu().numpy(), open(cache_file, 'wb'))
        offset = offset.to(input.device).type(input.dtype)

    return deform_conv2d(input, offset, weight, bias, stride, padding, dilation, mask)


class StereoConv2d(nn.Module):
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
            proj_params: proto_gen.detect_pb2.StereoProjectParams = proto_gen.detect_pb2.StereoProjectParams(
                project_dis=1, project_size=3),
            thetaRate: float = 1.0,  # 默认 detTheta = proj_theta / proj_size, 可调整 detTheta = detTheta * thetaRate
    ):
        super(StereoConv2d, self).__init__()

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

        self.proj_params = proj_params
        self.thetaRate = thetaRate

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return stereo_conv2d(input, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=None,
                             proj_params=self.proj_params,
                             thetaRate=self.thetaRate)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO,
    )

    x = torch.rand(1, 1, 256, 256)
    M = StereoConv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
    out = M(x)
    print(out.shape)
