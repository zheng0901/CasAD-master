
import copy
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
EPSILON = 1e-7



class MyDataset(Dataset):
    def __init__(self,input_tslices, input_global,input_fnode,input_newnodes,label, max_length):
        self.input_global,self.y = input_global,label
        self.max_cascade_length = max_length
        self.len = len(input_global)
        self.input_tslices,self.input_fnode,self.input_newnodes = input_tslices,input_fnode,input_newnodes
        # self.pattern_labels = pattern_labels

    def __len__(self):
        return self.len

    def __getitem__(self, x_idx):
        b_tslines=  self.input_tslices[x_idx]
        b_fnode = self.input_fnode[x_idx]
        b_newnode = self.input_newnodes[x_idx]
        b_global = self.input_global[x_idx]
        b_y = self.y[x_idx]
        # b_pattern_labels = self.pattern_labels[x_idx]

        b_time = np.array(b_tslines)
        time2 = np.insert(b_time, 0, 0)
        time3 = time2[0 : -1]
        temp = b_time - time3
        idx = np.where(temp == 0)


        if len(idx[0]):
            idx = np.array(idx)
            idx = np.squeeze(idx, 0)
            idx_next = idx + 1
            idx = idx.tolist()
            idx_next = idx_next.tolist()

            if idx_next[-1] == len(b_time):
                b_time[idx[-1]] = b_time[idx[-1]] - 0.001
                b_time[idx[0:-1]] = (b_time[idx[0:-1]] + b_time[idx_next[0:-1]]) / 2.0
            else:
                b_time[idx[0:]] = (b_time[idx[0:]] + b_time[idx_next[0:]]) / 2.0

            # for i in range(0, len(idx)):
            #     a = idx[i]
            #     if a == len(b_time)-1:
            #         b_time[a] = b_time[a]- 0.001
            #     else:
            #         b_time[a] = (b_time[a] + b_time[a+1]) / 2.0

            # b_time[0] = (b_time[0] + b_time[1]) / 2.0


        b_time = b_time.tolist()

        while len(b_global) < self.max_cascade_length:
            b_global.append(np.zeros(shape=len(b_global[0])))

        while len(b_time) < self.max_cascade_length:
            b_time.append(b_time[-1]-0.001)

        while len(b_newnode) < self.max_cascade_length:
            b_newnode.append(np.zeros(shape=len(b_newnode[0])))


        # return np.array(b_global),np.array(b_y), np.array(b_time),np.array(b_fnode),np.array(b_newnode),np.array(b_pattern_labels)
        return np.array(b_global),np.array(b_y), np.array(b_time),np.array(b_fnode),np.array(b_newnode)
