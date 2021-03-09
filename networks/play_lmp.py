from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.distributions as td
import pytorch_lightning as pl

from networks.networks import VisionNet, PlanRecognitionNet, PlanProposalNet, PolicyNet
import utils.constants as constants


class PlayLMPNet(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-4,
        beta: float = 0.01,
        num_mix = 5,
    ):
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.num_mix = num_mix
        self.vision_net = VisionNet()
        self.pr_net = PlanRecognitionNet()
        self.pp_net = PlanProposalNet()
        self.policy_net = PolicyNet()

    def forward(self, x_visual, x_proprio):
        '''
        x_visual: (bs, seq_len, c, h, w)
        x_proprio: (bs, seq_len, proprio_size)
        '''
        bs, seq_len, c, h, w = x_visual.shape

        # vision_net
        x_visual = x_visual.reshape(-1, c, h, w) # reshape to parallelize vision network application
        x_visual_encoded = self.vision_net(x_visual) # (bs*seq_len, out_features_visual)
        x_visual_encoded = x_visual_encoded.reshape(bs, seq_len, -1) # reshape it back: (bs, seq_len, out_features_visual)
            
        # pp_net
        pp_input = torch.cat([x_visual_encoded[:,-1,:], x_visual_encoded[:,0,:]], dim=1) # (bs, out_features_visual) -> [goal, current]
        y_pp_mean, y_pp_scale = self.pp_net(pp_input) # (bs, out_features_plan)

        # policy_net
        sampled_plan = td.Normal(y_pp_mean, y_pp_scale).sample() # (bs, out_features_plan)
        policy_input = torch.cat([sampled_plan, pp_input], dim=-1).unsqueeze(1) # (bs, seq_len, out_features_plan + out_features_visual) -> unsqueeze to add seq_len
        pi, mean, scale = self.policy_net(policy_input)
        action_output = self.policy_net.sample(pi, mean, scale)

        return action_output # action prediction: (bs, seq_len, out_features_action)

    def training_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def _shared_eval(self, batch, batch_idx, prefix):
        x_visual = batch['O_seq']
        x_proprio = batch['a_seq']
        actions = batch['a_seq']
        bs, seq_len, c, h, w = x_visual.shape

        # vision_net
        x_visual = x_visual.reshape(-1, c, h, w) # reshape to parallelize vision network application
        x_visual_encoded = self.vision_net(x_visual) # (bs*seq_len, out_features_visual)
        x_visual_encoded = x_visual_encoded.reshape(bs, seq_len, -1) # reshape it back: (bs, seq_len, out_features_visual)
        
        # pr_net
        pr_input = torch.cat([x_visual_encoded, x_proprio], dim=-1) # (bs, seq_len, out_features_visual, out_features_action)
        y_pr_mean, y_pr_scale = self.pr_net(pr_input) # (bs, out_features_plan) for both mean & std
        pr_dist = td.Normal(y_pr_mean, y_pr_scale)

        # pp_net
        pp_input = torch.cat([x_visual_encoded[:,-1,:], x_proprio[:,0,:], x_visual_encoded[:,0,:]], dim=-1) # (bs, out_features_visual) -> [goal, current]
        y_pp_mean, y_pp_scale = self.pp_net(pp_input) # (bs, out_features_plan) for both mean & std
        pp_dist = td.Normal(y_pp_mean, y_pp_scale)

        # policy_net
        sampled_plan = pr_dist.rsample()
        goal_plan = torch.cat([sampled_plan, x_visual_encoded[:,-1,:]], dim=-1) # (bs, out_features_plan + out_features_visual)
        goal_plan = goal_plan.unsqueeze(1).expand(-1, seq_len, -1) # unsqueeze to add seq_len dim and then expand seq_len size to match intended seq_len
        policy_input = torch.cat([goal_plan, pr_input], dim=-1) # [sampled_plan, vision_goal (i.e. visual only), vision_current (i.e. visuo-proprio)]
        pi, mean, scale = self.policy_net(policy_input)
        
        # kl loss
        kl_loss = td.kl_divergence(pr_dist, pp_dist).mean()

        # reconstruction loss (discretized mixture of logistics loss)
        recon_loss = self.policy_net.loss(pi, mean, scale, actions)

        # total loss
        loss = recon_loss + self.beta * kl_loss

        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        # TODO: include all params from all networks
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer