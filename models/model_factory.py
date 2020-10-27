import os
import shutil

import torch
import torch.nn as nn
from torch.optim import Adam
from models import convlstm, stconvlstm

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)

        networks_map = {
            'convlstm': convlstm.RNN,
            'stconvlstm': stconvlstm.RNN
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.MSE_criterion = nn.MSELoss().to(configs.device)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        stats['optimizer_param'] = self.optimizer.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])
        self.optimizer.load_state_dict(stats['optimizer_param'])

    # frames.shape : [batch, seq, height, width, channel]
    # that is : (batch_size, seq_length, height / patch, width / patch, patch_size * patch_size * num_channels)
    def train(self, frames, mask, itr=None):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames = self.network(frames_tensor, mask_tensor, itr)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask, itr=None):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor, itr)
        return next_frames.detach().cpu().numpy()