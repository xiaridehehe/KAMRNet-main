import os
import random
import math
import numpy as np

import torch as th

class RandomCrop(object):

    def __init__(self, size):
        """
        Randomly crop a torch tensor
        Arguments
        --------
        size : tuple or list
            dimensions of the crop
        """
        self.size = size

    def __call__(self, *inputs):
        h_idx = random.randint(0,inputs[0].size(1)-self.size[0])
        w_idx = random.randint(0,inputs[0].size(2)-self.size[1])
        outputs = []
        for idx, _input in enumerate(inputs):
            _input = _input[:, h_idx:(h_idx+self.size[0]),w_idx:(w_idx+self.size[1])]
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]