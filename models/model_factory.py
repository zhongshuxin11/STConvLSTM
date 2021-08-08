import os
import torch
import torch.nn as nn
from torch.optim import Adam
from models import stconvlstm


class MSEL1Loss(nn.Module):
    def __init__(self, reduction='mean', alpha=0.5):
        super(MSEL1Loss, self).__init__()
        self.alpha = alpha

        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.l1_criterion = nn.L1Loss(reduction=reduction)

    def __call__(self, inputs, targets):
        mse_loss = self.mse_criterion(inputs, targets)
        l1_loss = self.l1_criterion(inputs, targets)
        return mse_loss + self.alpha * l1_loss


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)

        networks_map = {
            'stconvlstm': stconvlstm.RNN
        }
        criterion_map = {
            'MSE': nn.MSELoss,
            'L1': nn.L1Loss,
            'MSE+L1': MSEL1Loss,
            'SmoothL1': nn.SmoothL1Loss
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        if configs.criterion in criterion_map:
            Criterion = criterion_map[configs.criterion]
            self.criterion = Criterion(reduction='sum').to(configs.device)
            print('Using %s as criterion' % configs.criterion)
        else:
            raise ValueError('Name of criterion unknown %s' % configs.criterion)


        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        stats['optimizer_param'] = self.optimizer.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
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
        loss = self.criterion(next_frames, frames_tensor[:, 1:])
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask, itr=None):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor, itr)
        return next_frames.detach().cpu().numpy()
