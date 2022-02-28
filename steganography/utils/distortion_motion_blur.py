import math

import torch
import numpy as np
from skimage.draw import line
from torch.nn.functional import conv2d
from steganography.utils.distortion_line_dictionary import LineDictionary

'''
任意方向上的运动模糊
'''

class Motion_Blur():
    def __init__(self, img:torch.Tensor, angle, kernel_size, linetype="full"):
        '''
        kernel_size 必须是一个odd
        '''
        self.img = img
        self.angle = angle
        self.kernel_size = kernel_size
        self.lintype = linetype
        self.kernel = self.get_kernel()

    def nearestValue(self,theta, validAngles):
        idx = (np.abs(validAngles - theta)).argmin()
        return validAngles[idx]

    def LineKernel(self,dim, angle, linetype="full"):
        kernelwidth = dim
        kernelCenter = int(math.floor(dim / 2))
        angle = self.SanitizeAngleValue(kernelCenter, angle)
        kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
        lineDict = LineDictionary(kernelwidth)
        lineAnchors = lineDict.lines[dim][angle]

        if linetype == "right":
            lineAnchors[0] = kernelCenter
            lineAnchors[1] = kernelCenter
        if linetype == "left":
            lineAnchors[2] = kernelCenter
            lineAnchors[3] = kernelCenter

        rr, cc = line(lineAnchors[0], lineAnchors[1], lineAnchors[2], lineAnchors[3])
        kernel[rr, cc] = 1
        normalizationFactor = np.count_nonzero(kernel)
        kernel = kernel / normalizationFactor
        return kernel

    def get_kernel(self):
        self.kernel = self.LineKernel(dim=self.kernel_size, angle=self.angle, linetype=self.lintype)
        self.kernel = torch.from_numpy(self.kernel).expand(self.img.shape[-3],1,self.kernel.shape[-2],self.kernel.shape[-1])
        return self.kernel

    def motion_blur(self):
        return conv2d(self.img,self.kernel,groups=self.img.shape[-3],padding=self.kernel_size//2)



    def SanitizeAngleValue(self,kernelCenter, angle):
        numDistinctLines = kernelCenter * 4
        angle = math.fmod(angle, 180.0)
        validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)
        angle = self.nearestValue(angle, validLineAngles)
        return angle