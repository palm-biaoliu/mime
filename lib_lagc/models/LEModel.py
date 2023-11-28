import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math

from .backbone import build_backbone
from .encoder import build_encoder
from ..utils.helper import clean_state_dict

NUM_CHANNEL = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
}

NUM_CLASS = {
    'pascal': 20,
    'coco': 80,
    'nuswide': 81,
    'cub': 312,
}

class GroupWiseLinear(nn.Module):

    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class LEModel(nn.Module):
    def __init__(self, backbone, encoder, num_class, feat_dim, is_proj):
        """[summary]
    
        Args:
            backbone : backbone model.
            encoder : encoder model.
            num_class : number of classes. (80 for MSCOCO).
            feat_dim : dimension of features.
            is_proj : open/close a projector.
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.is_proj = is_proj
        
        hidden_dim = encoder.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)
        if is_proj:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feat_dim)
            )


    def forward(self, input):
        # import ipdb; ipdb.set_trace()
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        query_input = self.query_embed.weight
        hs = self.encoder(self.input_proj(src), query_input, pos)[0]    # B,K,d

        out = self.fc(hs[-1])


        if self.is_proj:
            feat = hs[-1]
            batch_size = feat.shape[0]
            feat = torch.cat(torch.unbind(feat, dim=0))                     #  == (batch_size, num_class, ...)->(batch_size * num_class, ...)
            feat = F.normalize(self.proj(feat), dim=1)                      #  == (..., hidden_dim)->(..., feat_dim)
            feat = torch.stack(torch.chunk(feat, batch_size, dim=0), dim=0) #  == (batch_size * num_class, ...)->(batch_size, num_class, ...)
            return out, feat
        else:
            return out


    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


class LEModel_VIB(nn.Module):
    def __init__(self, backbone, encoder, num_class, feat_dim, is_proj=True):
        """[summary]

        Args:
            backbone : backbone model.
            encoder : encoder model.
            num_class : number of classes. (80 for MSCOCO).
            feat_dim : dimension of features.
            is_proj : open/close a projector.
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.proj = is_proj

        hidden_dim = encoder.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, feat_dim, bias=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * feat_dim)
        )

    def forward(self, input, is_train=True):
        # import ipdb; ipdb.set_trace()
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        query_input = self.query_embed.weight
        hs = self.encoder(self.input_proj(src), query_input, pos)[0]  # B,K,d
        statistics = hs[-1]
        if self.proj:
            statistics = self.proj(statistics)
            z_mu = statistics[:, :, :self.feat_dim]
            z_std = torch.exp(torch.clamp(statistics[:,:,self.feat_dim:]/2, min = -6, max = 6))
        else:
            z_dim = int(statistics.shape[2] / 2)
            z_mu = statistics[:, :, :z_dim]
            z_std = F.softplus(statistics[:, :, z_dim:])
        normal_sample_machine = torch.distributions.normal.Normal(z_mu, z_std)
        # z = normal_sample_machine.rsample()
        z = normal_sample_machine.rsample((10,)).mean(dim=0)
        out = self.fc(z)

        return out, z_mu, z_std

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))


class LEModel_smile(nn.Module):
    def __init__(self, backbone, encoder, num_class, feat_dim, is_proj=True):
        """[summary]

        Args:
            backbone : backbone model.
            encoder : encoder model.
            num_class : number of classes. (80 for MSCOCO).
            feat_dim : dimension of features.
            is_proj : open/close a projector.
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.proj = is_proj

        hidden_dim = encoder.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.d_encoder_alpha = GroupWiseLinear(num_class, hidden_dim, bias=True)
        self.d_encoder_beta = GroupWiseLinear(num_class, hidden_dim, bias=True)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * feat_dim)
        )

    def forward(self, input, is_train=True):
        # import ipdb; ipdb.set_trace()
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        query_input = self.query_embed.weight
        hs = self.encoder(self.input_proj(src), query_input, pos)[0]  # B,K,d
        features, statistics = hs[-1], hs[-1]

        statistics = self.proj(statistics)
        z_mu = statistics[:, :, :self.feat_dim]
        z_std = F.softplus(statistics[:, :, self.feat_dim:])

        # normal_sample_machine = torch.distributions.normal.Normal(z_mu, z_std)
        # z = normal_sample_machine.rsample()
        # z = normal_sample_machine.rsample((10,)).mean(dim=0)
        out = self.fc(features)

        # distribution_statics = self.d_encoder(z)
        # alpha = distribution_statics[:, :self.num_classes]
        # beta = distribution_statics[:, self.num_classes:]
        alpha = self.d_encoder_alpha(features)
        beta = self.d_encoder_beta(features)
        alpha, beta = F.softplus(alpha) + 1e-6, F.softplus(beta) + 1e-6
        # alpha, beta = F.relu(alpha) + 1e-6, F.relu(beta) + 1e-6
        beta_sample_machine = torch.distributions.beta.Beta(alpha, beta)
        d = beta_sample_machine.rsample()
        return out, d, z_mu, z_std, alpha, beta

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))


def build_LEModel(args):
    args.num_class = NUM_CLASS[args.dataset_name]
    args.hidden_dim = NUM_CHANNEL[args.backbone]

    backbone = build_backbone(args)
    encoder = build_encoder(args)

    model = LEModel(
        backbone = backbone,
        encoder = encoder,
        num_class = args.num_class,
        feat_dim = args.feat_dim,
        is_proj = args.is_proj,
    )
    
    model.input_proj = nn.Identity()
    print("set model.input_proj to Indentify!")

    return model


def build_LEModel_VIB(args, is_proj=True):
    args.num_class = NUM_CLASS[args.dataset_name]
    args.hidden_dim = NUM_CHANNEL[args.backbone]

    backbone = build_backbone(args)
    encoder = build_encoder(args)

    model = LEModel_VIB(
        backbone=backbone,
        encoder=encoder,
        num_class=args.num_class,
        feat_dim=args.feat_dim,
        is_proj=is_proj,
    )

    model.input_proj = nn.Identity()
    print("set model.input_proj to Indentify!")

    return model


def build_LEModel_smile(args, is_proj=True):
    args.num_class = NUM_CLASS[args.dataset_name]
    args.hidden_dim = NUM_CHANNEL[args.backbone]

    backbone = build_backbone(args)
    encoder = build_encoder(args)

    model = LEModel_smile(
        backbone=backbone,
        encoder=encoder,
        num_class=args.num_class,
        feat_dim=args.feat_dim,
        is_proj=is_proj,
    )

    model.input_proj = nn.Identity()
    print("set model.input_proj to Indentify!")

    return model