import torch
import os
import torch.nn as nn
import numpy as np

import sys
sys.path.append('./temporal_network')
import EvoSkeleton.libs.model as libm
from EvoSkeleton.libs.dataset.h36m.data_utils import unNormalizeData

class EvoNet(nn.Module):
    
    def __init__(self, root = './EvoSkeleton/examples'):
        super(EvoNet, self).__init__()
        self.num_joints = 16
        # 16 out of 17 key-points are used as inputs in this examplar model
        self.re_order_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]

        self.model_path = root + '/example_model.th'
        self.stats = np.load(root + '/stats.npy', allow_pickle = True).item()
        self.model = self.load_model() 

    def forward(self, x):
        num_stages = len(self.model)
        # for legacy code that does not have the num_blocks attribute
        for i in range(num_stages):
            self.model[i].num_blocks = len(self.model[i].res_blocks)
        prediction = self.model[0](x)
        # prediction for later stages
        for stage_idx in range(1, num_stages):
            prediction += self.model[stage_idx](x)
    
        return prediction


    def load_model(self):
        # load the checkpoint and statistics
        ckpt = torch.load(self.model_path)
        # initialize the model
        cascade = libm.get_cascade()
        input_size = 32
        output_size = 48
        for stage_id in range(2):
            # initialize a single deep learner
            stage_model = libm.get_model(stage_id + 1,
                                        refine_3d=False,
                                        norm_twoD=False, 
                                        num_blocks=2,
                                        input_size=input_size,
                                        output_size=output_size,
                                        linear_size=1024,
                                        dropout=0.5,
                                        leaky=False)
            cascade.append(stage_model)
        
        cascade.load_state_dict(ckpt)

        return cascade


    def normalize(self, skeleton):
        re_order= [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
        norm_skel = skeleton
        bs = norm_skel.shape[0]
        norm_skel = norm_skel[:,re_order].reshape(bs, 32)

        norm_skel = norm_skel.reshape(bs, 16, 2)
        mean_x = torch.mean(norm_skel[:,:,0], dim=1).unsqueeze(1)
        std_x = torch.std(norm_skel[:,:,0], dim=1).unsqueeze(1)
        mean_y = torch.mean(norm_skel[:,:,1], dim=1).unsqueeze(1)
        std_y = torch.std(norm_skel[:,:,1], dim=1).unsqueeze(1)
        denominator = (0.5*(std_x + std_y))

        norm_skel[:,:,0] = (norm_skel[:,:,0] - mean_x)/denominator
        norm_skel[:,:,1] = (norm_skel[:,:,1] - mean_y)/denominator

        norm_skel = norm_skel.reshape(bs, 32)         
        return norm_skel


    def afterprocessing(self, x):
        pred_for_show = unNormalizeData(x.data.cpu().numpy(),
            self.stats['mean_3d'],
            self.stats['std_3d'],
            self.stats['dim_ignore_3d']
        )

        bs = pred_for_show.shape[0]
        skeleton = pred_for_show.reshape(bs, -1, 3)
        skeleton[:, :, [0,1,2]] = skeleton[:, :, [0,2,1]]
        skeleton = skeleton.reshape(bs, 96)
        vals = np.reshape(skeleton, (bs, 32, -1))
        data_to_save = vals[:, [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27], :]

        return data_to_save
