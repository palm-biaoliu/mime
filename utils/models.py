from types import SimpleNamespace

import torch
import numpy as np
import copy

from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F
from collections import OrderedDict

from lib_lagc.models.LEModel import build_LEModel

'''
utility functions
'''


def inverse_sigmoid(p):
    epsilon = 1e-5
    p = np.minimum(p, 1 - epsilon)
    p = np.maximum(p, epsilon)
    return np.log(p / (1 - p))


'''
model definitions
'''


class FCNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(FCNet, self).__init__()
        self.fc = torch.nn.Linear(num_feats, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class ImageClassifier(torch.nn.Module):

    def __init__(self, P, model_feature_extractor=None, model_linear_classifier=None):

        super(ImageClassifier, self).__init__()
        # print('initializing image classifier')

        model_feature_extractor_in = copy.deepcopy(model_feature_extractor)
        model_linear_classifier_in = copy.deepcopy(model_linear_classifier)

        self.arch = P['arch']

        if self.arch == 'resnet50':
            # configure feature extractor:
            if model_feature_extractor_in is not None:
                # print('feature extractor: specified by user')
                feature_extractor = model_feature_extractor_in
            else:
                if P['use_pretrained']:
                    # print('feature extractor: imagenet pretrained')
                    feature_extractor = resnet50(pretrained=True)
                else:
                    # print('feature extractor: randomly initialized')
                    feature_extractor = resnet50(pretrained=False)
                feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
            if P['freeze_feature_extractor']:
                # print('feature extractor frozen')
                for param in feature_extractor.parameters():
                    param.requires_grad = False
            else:
                # print('feature extractor trainable')
                for param in feature_extractor.parameters():
                    param.requires_grad = True
            feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.feature_extractor = feature_extractor

            # configure final fully connected layer:
            if model_linear_classifier_in is not None:
                # print('linear classifier layer: specified by user')
                linear_classifier = model_linear_classifier_in
            else:
                # print('linear classifier layer: randomly initialized')
                linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
            self.linear_classifier = linear_classifier

        elif self.arch == 'linear':
            # print('training a linear classifier only')
            self.feature_extractor = None
            self.linear_classifier = FCNet(P['feat_dim'], P['num_classes'])

        else:
            raise ValueError('Architecture not implemented.')

    def forward(self, x):
        if self.arch == 'linear':
            # x is a batch of feature vectors
            logits = self.linear_classifier(x)
        else:
            # x is a batch of images
            feats = self.feature_extractor(x)
            logits = self.linear_classifier(torch.squeeze(feats))
        return logits


class LabelEstimator(torch.nn.Module):

    def __init__(self, P, observed_label_matrix, estimated_labels):

        super(LabelEstimator, self).__init__()
        # print('initializing label estimator')

        # Note: observed_label_matrix is assumed to have values in {-1, 0, 1} indicating 
        # observed negative, unknown, and observed positive labels, resp.

        num_examples = int(np.shape(observed_label_matrix)[0])
        observed_label_matrix = np.array(observed_label_matrix).astype(np.int8)
        total_pos = np.sum(observed_label_matrix == 1)
        total_neg = np.sum(observed_label_matrix == -1)
        # print('observed positives: {} total, {:.1f} per example on average'.format(total_pos, total_pos / num_examples))
        # print('observed negatives: {} total, {:.1f} per example on average'.format(total_neg, total_neg / num_examples))

        if estimated_labels is None:
            # initialize unobserved labels:
            w = 0.1
            q = inverse_sigmoid(0.5 + w)
            param_mtx = q * (2 * torch.rand(num_examples, P['num_classes']) - 1)

            # initialize observed positive labels:
            init_logit_pos = inverse_sigmoid(0.995)
            idx_pos = torch.from_numpy((observed_label_matrix == 1).astype(np.bool_))
            param_mtx[idx_pos] = init_logit_pos

            # initialize observed negative labels:
            init_logit_neg = inverse_sigmoid(0.005)
            idx_neg = torch.from_numpy((observed_label_matrix == -1).astype(np.bool_))
            param_mtx[idx_neg] = init_logit_neg
        else:
            param_mtx = inverse_sigmoid(torch.FloatTensor(estimated_labels))

        self.logits = torch.nn.Parameter(param_mtx)

    def get_estimated_labels(self):
        with torch.set_grad_enabled(False):
            estimated_labels = torch.sigmoid(self.logits)
        estimated_labels = estimated_labels.clone().detach().cpu().numpy()
        return estimated_labels

    def forward(self, indices):
        x = self.logits[indices, :]
        x = torch.sigmoid(x)
        return x


class MultilabelModel(torch.nn.Module):
    def __init__(self, P, feature_extractor, linear_classifier, observed_label_matrix, estimated_labels=None, LEM=False):
        super(MultilabelModel, self).__init__()

        if LEM:
            args = SimpleNamespace()
            args.dataset_name = P['dataset']
            args.backbone = P['arch']
            args.is_proj = False
            args.img_size = 448
            args.feat_dim = 128
            self.f = build_LEModel(args)
        else:
            self.f = ImageClassifier(P, feature_extractor, linear_classifier)

        self.g = LabelEstimator(P, observed_label_matrix, estimated_labels)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        # g_preds = self.g(batch['idx'])  # oops, we had a sigmoid here in addition to
        return f_logits


class MultilabelModel_EM(torch.nn.Module):
    def __init__(self, P, feature_extractor, linear_classifier, LEM=False):
        super(MultilabelModel_EM, self).__init__()
        if LEM:
            args = SimpleNamespace()
            args.dataset_name = P['dataset']
            args.backbone = P['arch']
            args.is_proj = False
            args.img_size = 448
            args.feat_dim = 128
            self.f = build_LEModel(args)
        else:
            self.f = ImageClassifier(P, feature_extractor, linear_classifier)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        return f_logits


class MultilabelModel_smile(torch.nn.Module):
    def __init__(self, P, feature_extractor):
        super(MultilabelModel_smile, self).__init__()
        model_feature_extractor_in = copy.deepcopy(feature_extractor)

        self.arch = P['arch']

        if self.arch == 'resnet50':
            # configure feature extractor:
            if model_feature_extractor_in is not None:
                feature_extractor = model_feature_extractor_in
            else:
                if P['use_pretrained']:
                    feature_extractor = resnet50(pretrained=True)
                else:
                    feature_extractor = resnet50(pretrained=False)
                feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
            if P['freeze_feature_extractor']:
                for param in feature_extractor.parameters():
                    param.requires_grad = False
            else:
                for param in feature_extractor.parameters():
                    param.requires_grad = True
            feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.feature_extractor = feature_extractor
        self.feat_dim = P['feat_dim']
        self.num_classes = int(P['num_classes'])
        self.z_dim = P['z_dim']
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, 2 * self.z_dim)
        )
        self.decoder_D = nn.Linear(self.z_dim, 2 * self.num_classes, bias=True)
        self.linear_classifier = nn.Linear(self.z_dim, self.num_classes, bias=True)

    def forward(self, x):
        extracted_feat_statics = torch.squeeze(self.feature_extractor(x))
        # encoder for z
        extracted_feat_statics = self.proj(extracted_feat_statics)
        z_mu = extracted_feat_statics[:, :self.z_dim]
        z_std = F.softplus(extracted_feat_statics[:, self.z_dim:])
        # z_std = torch.exp(extracted_feat_statics[:, self.z_dim:] / 2)
        normal_sample_machine = torch.distributions.normal.Normal(z_mu, z_std)
        z = normal_sample_machine.rsample()
        # encoder for d
        distribution_statics = self.decoder_D(z)
        alpha = distribution_statics[:, :self.num_classes]
        beta = distribution_statics[:, self.num_classes:]
        alpha, beta = F.softplus(alpha), F.softplus(beta)
        # alpha, beta = F.relu(alpha) + 1e-6, F.relu(beta) + 1e-6
        beta_sample_machine = torch.distributions.beta.Beta(alpha, beta)
        d = beta_sample_machine.rsample()
        # prediction
        out = self.linear_classifier(z)
        return out, d, z_mu, z_std, alpha, beta


class MultilabelModel_LAGC(torch.nn.Module):
    def __init__(self, P, is_proj=False):
        super(MultilabelModel_LAGC, self).__init__()
        args = SimpleNamespace()
        args.dataset_name = P['dataset']
        args.backbone = P['arch']
        args.is_proj = is_proj
        args.img_size = 448
        args.feat_dim = P['z_dim']
        self.f = build_LEModel(args)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        return f_logits


class MultilabelModel_VIB(torch.nn.Module):
    def __init__(self, P, model_feature_extractor):
        super(MultilabelModel_VIB, self).__init__()
        model_feature_extractor_in = copy.deepcopy(model_feature_extractor)

        self.arch = P['arch']

        if self.arch == 'resnet50':
            # configure feature extractor:
            if model_feature_extractor_in is not None:
                feature_extractor = model_feature_extractor_in
            else:
                if P['use_pretrained']:
                    feature_extractor = resnet50(pretrained=True)
                else:
                    feature_extractor = resnet50(pretrained=False)
                feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
            if P['freeze_feature_extractor']:
                for param in feature_extractor.parameters():
                    param.requires_grad = False
            else:
                for param in feature_extractor.parameters():
                    param.requires_grad = True
            feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.feature_extractor = feature_extractor
        self.encoder_z = nn.ModuleList([])
        self.num_classes = P['num_classes']
        self.z_dim = P['z_dim']
        for _ in range(P['num_classes']):
            self.encoder_z.append(self._get_encoder_z(P['feat_dim'], P['z_dim']))
        self.linear_classifier = nn.Linear(P['z_dim'], 1, bias=True)

    def forward(self, x):
        extracted_feat = self.feature_extractor(x)
        output_list = []
        mu_list = []
        std_list = []
        for enc in self.encoder_z:
            statistics = enc(torch.squeeze(extracted_feat))
            z_mu = statistics[:, :self.z_dim]
            # z_std = F.softplus(statistics[:, self.z_dim:])
            z_std = torch.exp(torch.clamp(statistics[:,self.z_dim:] / 2, min = -6, max = 6))
            mu_list.append(z_mu)
            std_list.append(z_std)
            normal_sample_machine = torch.distributions.normal.Normal(z_mu, z_std)
            z = normal_sample_machine.rsample()
            out = self.linear_classifier(z)
            output_list.append(out)
        return torch.cat(output_list, dim=1), torch.cat(mu_list, dim=1), torch.cat(std_list, dim=1)

    def _get_encoder_z(self, feat_dim, z_dim):
        encode_z = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(True),
            # nn.Linear(1024, 1024),
            # nn.ReLU(True),
            nn.Linear(1024, 2 * z_dim))
        return encode_z
