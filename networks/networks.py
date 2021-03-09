from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.distributions as td
import pytorch_lightning as pl

import utils.functions as CF
import utils.constants as constants


class SpatialSoftmax(nn.Module):
    def __init__(self, channel, height, width):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height, device=self.device),
            torch.linspace(-1.0, 1.0, self.width, device=self.device),
        )
        pos_x = pos_x.reshape(self.height * self.width)
        pos_y = pos_y.reshape(self.height * self.width)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        """
        x: (bs, c, h, w)
        """
        x = x.view(-1, self.height * self.width)
        softmax_c = F.softmax(x, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_c, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_c, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class VisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        out_features = constants.OUT_FEATURES_VISUAL
        w, h = calc_out_wh(300, 300, 8, 0, 5)
        w, h = calc_out_wh(w, h, 4, 0, 2)
        w, h = calc_out_wh(w, h, 3, 0, 1)

        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.spatial_softmax = SpatialSoftmax(64, w, h)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, out_features)

    def forward(self, x_visual):
        """
        x_visual: (bs, c, h, w)
        """
        x_visual = F.relu(self.conv1(x_visual))
        x_visual = F.relu(self.conv2(x_visual))
        x_visual = F.relu(self.conv3(x_visual))
        x_visual = self.spatial_softmax(x_visual)
        x_visual = F.relu(self.fc1(x_visual))
        x_visual = self.fc2(x_visual)

        return x_visual  # (bs, out_features)

    def calc_out_wh(self, w, h, kernel_size, padding, stride):
        width = (w - kernel_size + 2 * padding) // stride + 1
        height = (h - kernel_size + 2 * padding) // stride + 1
        return width, height


class PlanRecognitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = constants.OUT_FEATURES_VISUAL + constants.OUT_FEATURES_ACTION
        self.out_features = constants.OUT_FEATURES_PLAN
        self.bi_rnn = nn.LSTM(
            input_size=self.in_features,
            hidden_size=2048,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(4096, self.out_features)

    def forward(self, x):
        """
        x: (bs, seq_len, input_size) -> input_size: out_features_visual + out_features_action
        """
        x, _ = self.bi_rnn(x)
        x = x[:, -1]  # only use the last item of the output seq
        mean = self.fc(x)
        var = F.softplus(self.fc(x))
        return mean, var  # (bs, out_features_plan)


class PlanProposalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = constants.OUT_FEATURES_VISUAL + (
            constants.OUT_FEATURES_VISUAL + constants.OUT_FEATURES_ACTION
        )
        self.out_features = constants.OUT_FEATURES_PLAN

        self.fc1 = nn.Linear(self.in_features, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, self.out_features)

    def forward(self, x):
        """
        x: (bs, input_size) -> input_size: goal (vision only) + current (visuo-proprio)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mean = self.fc5(x)
        var = F.softplus(self.fc5(x))

        return mean, var  # (bs, out_features_plan)


class PolicyNet(nn.Module):
    def __init__(self, num_mix=5, log_scale_min=-7.0):
        super().__init__()
        self.num_mix = constants.NUM_MIX
        self.log_scale_min = log_scale_min

        hidden_size = 2048
        self.in_features = (
            constants.OUT_FEATURES_PLAN
            + constants.OUT_FEATURES_VISUAL
            + (constants.OUT_FEATURES_ACTION + constants.OUT_FEATURES_VISUAL)
        )
        self.out_features = constants.OUT_FEATURES_PLAN

        self.rnn = nn.RNN(
            self.in_features,
            hidden_size=hidden_size,
            num_layers=2,
            nonlinearity="relu",
            bidirectional=False,
            batch_first=True,
        )

        self.fc_logit_probs = nn.Linear(
            hidden_size, self.out_features * self.num_mix
        )  # need mixtures for each feature
        self.fc_means = nn.Linear(hidden_size, self.out_features * self.num_mix)
        self.fc_log_scales = nn.Linear(hidden_size, self.out_features * self.num_mix)

    def forward(self, x):
        """
        x: (bs, seq_len, input_size)
        """
        bs, seq_len, _ = x.shape
        x, _ = self.rnn(x)  # take the RNN output
        print(f"x.shape={x.shape}")

        logit_probs = self.fc_logit_probs(x)
        means = self.fc_means(x)
        log_scales = self.fc_log_scales(x)
        log_scales = torch.clamp(log_scales, min=self.log_scale_min)

        # rescale to correct dimensions
        logit_probs = logit_probs.view(bs, seq_len, self.out_features, self.num_mix)
        means = means.view(bs, seq_len, self.out_features, self.num_mix)
        log_scales = log_scales.view(bs, seq_len, self.out_features, self.num_mix)

        logit_probs = F.softmax(
            logit_probs, dim=-1
        )  # convert logit_probs param to probability

        return logit_probs, means, log_scales

    def loss(self, logit_probs, means, log_scales, y, num_bins=256):
        """
        log-likelihood for mixture of discretized logistics; assumes the data has been rescaled to [-1,1] interval
        y: target action (bs, input_size, 1) - i.e. (bs, 8, 1)
        """
        # Make sure log scale doesn't go to infinity
        log_scales = torch.clamp(log_scales, min=self.log_scale_min)

        # Broadcast y -> (bs, input_size, mix_num)
        y = y * torch.ones([1, 1, self.num_mix], device=self.device)

        # Discretize (different cases to prevent overflow)
        centered_y = y - means
        inv_std = torch.exp(
            -log_scales
        )  # e^(-log(σ)) = 1/σ, i.e. inverted std to normalize scale to 1

        upper_in = inv_std * (centered_y + (1.0 / (num_bins - 1)))
        cdf_upper = F.sigmoid(upper_in)
        lower_in = inv_std * (centered_y - (1.0 / (num_bins - 1)))
        cdf_lower = F.sigmoid(lower_in)

        log_cdf_upper = upper_in - F.softplus(
            upper_in
        )  # log prob for edge case of 0 (before scaling)
        log_one_minus_cdf_lower = -F.softplus(
            lower_in
        )  # log prob for edge case of 255 (before scaling)
        cdf_delta = (
            cdf_upper - cdf_lower
        )  # probability for all other cases - i.e. area under pdf between upper_in and lower_in

        mid_in = inv_std * centered_y
        log_pdf_mid = (
            mid_in - log_scales - 2 * F.softplus(mid_in)
        )  # log prob in the center of the bin, used in extreme cases (i.e. approximate integral under pdf)

        # now select the right output: left edge case, right edge case, normal case, extremely low prob case
        # log_probs is the probability of the sample feature belonging to a bin (i.e. sample feature having some value between 0-255, normalized to [-1,1])
        # log_probs -> (bs, input_size, mix_num)
        log_probs = torch.where(
            x < -0.999,
            log_cdf_upper,
            torch.where(
                x > 0.999,
                log_one_minus_cdf_lower,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                    log_pdf_mid - np.log((num_bins - 1) / 2),
                ),
            ),
        )
        print(f"log_probs.shape={log_probs.shape}")

        log_probs = log_probs + F.log_softmax(
            logit_probs, dim=-1
        )  # log(xy) = log(x) + log(y) -> i.e. log(probs * logit_probs) -> mixture weighted log probs

        return -torch.sum(
            torch.logsumexp(log_probs), dim=-1
        ).mean()  # logsumexp = smooth approx of max; prevents overflow compared to softmax during normalization

    def sample(self, logit_probs, means, log_scales):
        # sample mixture using gumbel-max trick (which mimics softmax)
        max_val, min_val = 1e-5, -1e-5
        temp = (max_val - min_val) * torch.rand(
            means.shape, device=self.device
        ) + min_val  # torch.rand -> uniform distribution
        temp = logit_probs - torch.log(-torch.log(temp))
        argmax = torch.argmax(temp, -1)
        print(f"argmax.shape={argmax.shape}")
        dist = CF.one_hot_encoding(argmax, self.num_mix)

        # select mixture params
        log_scales = (dist * log_scales).sum(dim=-1)
        means = (dist * means).sum(dim=-1)

        # inversion sampling (sample from mixture distribution using cdf instead of pdf)
        scales = torch.exp(log_scales)
        u = (max_val - min_val) * torch.rand(means.shape, device=self.device) + min_val
        actions = means + scales * (torch.log(u) - torch.log(1.0 - u))

        # clipping actions within range
        actions = actions.clamp(-0.999, 0.999)
        print(f"actions.shape={actions.shape}")

        return actions
