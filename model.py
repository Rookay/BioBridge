# -*- coding: UTF-8 -*-
# @Project ：BioEEG-VisDec 
# @File    ：model.py
# @IDE     ：PyCharm 
# @Date    ：2025/11/21 12:22
import torch.nn.functional as F
import torch
import torch.nn as nn
from EEG_model import EEG_BaseModel, EEGProjectLayer, EEGnet, Deepnet, Shallownet, TSconv
from Img_model import Img_BaseModel, ResNet50FeatureExtractor, ResNet18FeatureExtractor


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=1024, num_heads=8, dropout=0.1):
        super().__init__()

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj   = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, eeg_feat, aux_feat):
        q = self.query_proj(eeg_feat).unsqueeze(1)   # (B, 1, 1024)
        k = self.key_proj(aux_feat).unsqueeze(1)     # (B, 1, 1024)
        v = self.value_proj(aux_feat).unsqueeze(1)   # (B, 1, 1024)

        attn_out, _ = self.attn(q, k, v) 
        out = self.norm(q + attn_out)
        out = self.fc(out).squeeze(1)     # (B, 1024)

        return out
class ChannelSpatialAttention(nn.Module):
    """
    Soft channel-wise attention for EEG signals.
    Input : [B, C, T]
    Output: [B, C, T]
    """

    def __init__(self,num_channels,use_softmax,prior=None,learnable_prior = True):
        super().__init__()

        self.num_channels = num_channels
        self.use_softmax = use_softmax

        self.channel_logits = nn.Parameter(torch.zeros(num_channels))
        prior = torch.tensor(prior, dtype=torch.float32)
        self.prior = nn.Parameter(prior)

        if  prior is not None:
            assert prior.shape[0] == num_channels
            if learnable_prior:
                self.prior = nn.Parameter(prior)
            else:
                self.register_buffer("prior", prior)
        else:
            self.prior = None

    def forward(self, x):
        """
        x: EEG tensor [B, C, T]
        """
        assert x.dim() == 3, "Input must be [B, C, T]"

        logits = self.channel_logits

        if self.prior is not None:
            logits = logits + self.prior

        if self.use_softmax:
            weights = F.softmax(logits, dim=0)
        else:
            weights = torch.sigmoid(logits)

        weights = weights.view(1, -1, 1)

        x_weighted = x * weights

        return x_weighted, weights.squeeze()
class TemporalStepAttention(nn.Module):

    def __init__(
        self,
        T = 250,
        step = 10,
        use_softmax  = True,
        prior = None,
    ):

        super().__init__()

        assert T % step == 0, "T 必须能被 step 整除"

        self.T = T
        self.step = step
        self.K = T // step
        self.use_softmax = use_softmax

        self.weight = nn.Parameter(torch.zeros(self.K))

        if prior is not None:
            if prior.numel() == T:
                prior = prior.view(self.K, step).mean(dim=1)
            assert prior.numel() == self.K
            self.register_buffer("prior", prior)
        else:
            self.prior = None

    def forward(self, x):
        """
        x: (B, C, T)
        """
        B, C, T = x.shape
        assert T == self.T

        w = self.weight

        if self.prior is not None:
            w = w + self.prior

        if self.use_softmax:
            w = F.softmax(w, dim=0)
        else:
            w = torch.sigmoid(w)

        w_full = w.repeat_interleave(self.step)  # (T,)

        x_weighted = x * w_full.view(1, 1, T)

        return x_weighted, w_full
class Simple_Model(nn.Module):
    def __init__(self, config, drop_proj=0.3):
        super(Simple_Model, self).__init__()
        z_dim = config['models']['z_dim']
        c_num = config['models']['c_num']
        timesteps = config['timesteps']
        self.low_prior = [
            0.3,  # P7
            0.3,  # P5
            0.3,  # P3
            0.3,  # P1
            0.3,  # Pz
            0.3,  # P2
            0.3,  # P4
            0.3,  # P6
            0.3,  # P8

            0.7,  # PO7
            0.7,  # PO3
            0.8,  # POz
            0.7,  # PO4
            0.7,  # PO8

            1.0,  # O1
            1.2,  # Oz
            1.0  # O2
        ]
        self.high_prior = [
            1.0,  # P7
            0.9,  # P5
            0.9,  # P3
            0.8,  # P1
            1.1,  # Pz
            0.8,  # P2
            0.9,  # P4
            0.9,  # P6
            1.0,  # P8

            0.6,  # PO7
            0.6,  # PO3
            0.7,  # POz
            0.6,  # PO4
            0.6,  # PO8

            0.2,  # O1
            0.1,  # Oz
            0.2  # O2
        ]
        self.low_process = ChannelSpatialAttention(c_num, use_softmax=True, prior=self.low_prior)
        self.high_process = ChannelSpatialAttention(c_num, use_softmax=True, prior=self.high_prior)

        self.low_eeg_encoder = EEG_BaseModel(z_dim,c_num,timesteps)
        self.high_eeg_encoder = EEG_BaseModel(z_dim,c_num,timesteps)

        self.low_eeg_encoder.backbone = EEGnet(z_dim,c_num,timesteps)
        self.high_eeg_encoder.backbone = EEGnet(z_dim,c_num,timesteps)

        self.low_img_encoder = Img_BaseModel(backbone_name="resnet50")
        self.high_img_encoder = ResNet50FeatureExtractor()
        self.cross_attention = CrossAttentionFusion(num_heads=8, dropout=0.1)


    def forward(self, eeg, img):
        eeg_low,_ = self.low_process(eeg)
        eeg_high,_ = self.high_process(eeg)
        low_eeg_feat = self.low_eeg_encoder(eeg_low)
        high_eeg_feat = self.high_eeg_encoder(eeg_high)
        high_eeg_feat = self.cross_attention(high_eeg_feat,low_eeg_feat)
        # with torch.no_grad():
        low_img_feat,blureed_img = self.low_img_encoder(img)
        high_img_feat = self.high_img_encoder(blureed_img)
        return low_eeg_feat, low_img_feat, high_eeg_feat, high_img_feat


