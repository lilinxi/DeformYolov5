import os
import logging
import math
import pickle
import hashlib

import cv2
import numpy as np
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

from models.stereo_conv_utils import compute_kernel_offset, get_theta_phi_range


# 可视化立体卷积核
def plot_kernel_offset():
    # im_raw = cv2.imread(proj_req.project_request.pano_dataset_model.image_path)  # BGR
    # im = proj.stereo_proj.stereo_proj(im_raw, project_params,
    #                                   project_width, project_height)
    im = cv2.imread("/Users/bytedance/Desktop/proj_image_12.png")  # BGR
    project_params = proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2)
    project_size = 2
    project_width = 640
    project_height = 640
    X = project_size * 7 / 9
    Y = project_size * 4 / 9

    pX = X * project_width / project_size
    pY = Y * project_height / project_size

    kernel_size = 9

    demo_kernel_x, demo_kernel_y = compute_kernel_offset(
        output_X_normal=X, output_Y_normal=Y,
        out_width=project_width, out_height=project_height,
        kernel_height=kernel_size, kernel_width=kernel_size,
        dil_h=1, dil_w=1,
        proj_params=project_params,
        thetaRate=10.0,
        device='cpu', dtype=torch.float32,
        absolute=True,
    )
    for i in range(kernel_size * kernel_size):
        x = demo_kernel_x.flatten()[i]
        y = demo_kernel_y.flatten()[i]
        px = round(pX + x.item())
        py = round(pY + y.item())
        print(pX, '+', x.item(), '=', px, '\t', pY, '+', y.item(), '=', py)
        cv2.circle(im, (px, py), 3, (0, 255,0 ), -1)
    print("===========================================")

    cv2.imshow('delta_theta_phi', im)
    cv2.imwrite('/Users/bytedance/Desktop/delta_theta_phi_12.png', im)
    cv2.waitKey(0)


def plot_delta_theta_phi(
        image_path: str,
        proj_params: proto_gen.detect_pb2.StereoProjectParams,
        delta_size=10) -> np.ndarray:
    im = cv2.imread(image_path)  # BGR
    theta_min, phi_min, theta_range, phi_range = get_theta_phi_range(proj_params)
    for delta_theta in np.linspace(0, theta_range, delta_size):
        for delta_phi in np.linspace(0, phi_range, delta_size):
            theta = theta_min + delta_theta
            phi = phi_min + delta_phi
            # for delta_theta in np.linspace(0, 2 * np.pi, delta_size):
            #     for delta_phi in np.linspace(0, np.pi, delta_size):
            #         heta = delta_theta
            #         phi = delta_phi
            x, y, z = proj.transform.theta_phi2xyz(theta, phi, theta_rotate=proj_params.theta_rotate)
            X, Y, _ = proj.stereo_proj._panoXYZ2projXY(x, y, z, proj_params)
            projX = int(X / proj_params.project_size * 640)
            projY = int(Y / proj_params.project_size * 640)
            cv2.circle(im, (projX, projY), 10, (0, 255,0), -1)
    return im


if __name__ == '__main__':
    # plot_kernel_offset()
    cv2.imshow('plot_delta_theta_phi', plot_delta_theta_phi(
        '/Users/bytedance/Desktop/proj_image_12.png',
        proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2)))
    cv2.imwrite('/Users/bytedance/Desktop/plot_delta_theta_phi_2.png', plot_delta_theta_phi(
        '/Users/bytedance/Desktop/proj_image_2.png',
        proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2)))
    cv2.imwrite('/Users/bytedance/Desktop/plot_delta_theta_phi_5.png', plot_delta_theta_phi(
        '/Users/bytedance/Desktop/proj_image_5.png',
        proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2)))
    cv2.imwrite('/Users/bytedance/Desktop/plot_delta_theta_phi_12.png', plot_delta_theta_phi(
        '/Users/bytedance/Desktop/proj_image_12.png',
        proto_gen.detect_pb2.StereoProjectParams(project_dis=1, project_size=2)))
    cv2.waitKey(0)
