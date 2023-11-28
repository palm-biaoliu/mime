import numpy as np
import torch
import torch.nn.functional as F

from utils.models import LabelEstimator
from utils.countExpected import count_estimate_

LOG_EPSILON = 1e-5

'''
helper functions
'''


def neg_log(x):
    return - torch.log(x + LOG_EPSILON)


def log_loss(preds, targs):
    return targs * neg_log(preds)


def expected_positive_regularizer(preds, expected_num_pos, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos) ** 2
    else:
        raise NotImplementedError
    return reg


def weighted_bce_loss(preds, obs, weights, bias=False):
    loss_mtx = torch.zeros_like(obs)
    weights[obs == 1.0] = 1.0
    loss_mtx[obs == 1.0] = (weights * neg_log(preds) + (1 - weights) * neg_log(1 - preds))[obs == 1.0]
    if bias:
        loss_mtx[obs == 0.0] = \
            ((1 / (obs.shape[-1] - 1)) * (weights * neg_log(preds) + (1 - weights) * neg_log(1 - preds)))[obs == 0.0]
    else:
        loss_mtx[obs == 0.0] = (weights * neg_log(preds) + (1 - weights) * neg_log(1 - preds))[obs == 0.0]
    return loss_mtx.mean()


def loss_an(preds, Y_obs):
    # input validation:
    assert torch.min(Y_obs) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(Y_obs)
    loss_mtx[Y_obs == 1] = neg_log(preds[Y_obs == 1])
    loss_mtx[Y_obs == 0] = neg_log(1.0 - preds[Y_obs == 0])
    return loss_mtx.mean()


def loss_anlw(preds, Y_obs):
    # input validation:
    assert torch.min(Y_obs) >= -1
    # compute loss:
    loss_mtx = torch.zeros_like(Y_obs)
    loss_mtx[Y_obs == 1] = neg_log(preds[Y_obs == 1])
    loss_mtx[Y_obs == 0] = neg_log(1.0 - preds[Y_obs == 0])
    return loss_mtx.mean()


def loss_anls(preds, Y_obs, ls_coef=0.1):
    # input validation:
    assert torch.min(Y_obs) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(Y_obs)
    loss_mtx[Y_obs == 1] = (1.0 - ls_coef) * neg_log(preds[Y_obs == 1]) + ls_coef * neg_log(1.0 - preds[Y_obs == 1])
    loss_mtx[Y_obs == 0] = (1.0 - ls_coef) * neg_log(1.0 - preds[Y_obs == 0]) + ls_coef * neg_log(preds[Y_obs == 0])
    return loss_mtx.mean()


def loss_wan(preds, Y_obs):
    # input validation:
    assert torch.min(Y_obs) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(Y_obs)
    loss_mtx[Y_obs == 1] = neg_log(preds[Y_obs == 1])
    loss_mtx[Y_obs == 0] = neg_log(1.0 - preds[Y_obs == 0]) / (Y_obs.shape[-1] - 1)
    return loss_mtx.mean()


def loss_epr(preds, Y_obs, expected_num_pos):
    loss_mtx = torch.zeros_like(Y_obs)
    loss_mtx[Y_obs == 1] = neg_log(preds[Y_obs == 1])
    # compute regularizer:
    reg_loss = expected_positive_regularizer(preds, expected_num_pos, norm='2') / (Y_obs.size(1) ** 2)
    return loss_mtx.mean() + reg_loss


class role:
    def __init__(self, Y) -> None:
        self.label_est = LabelEstimator(Y, None).cuda()

    def __call__(self, preds, Y_obs, expected_num_pos, idx):
        # unpack:
        estimated_labels = self.label_est(idx)
        # input validation:
        # assert torch.min(Y_obs) >= 0
        # (image classifier) compute loss w.r.t. observed positives:
        loss_mtx_pos_1 = torch.zeros_like(Y_obs)
        loss_mtx_pos_1[Y_obs == 1] = neg_log(preds[Y_obs == 1])
        # (image classifier) compute loss w.r.t. label estimator outputs:
        estimated_labels_detached = estimated_labels.detach()
        loss_mtx_cross_1 = torch.zeros_like(Y_obs)
        loss_mtx_cross_1[Y_obs == 0] = \
            (estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(1.0 - preds))[
                Y_obs == 0]
        # (image classifier) compute regularizer:
        reg_1 = expected_positive_regularizer(preds, expected_num_pos, norm='2') / (Y_obs.size(1) ** 2)
        # (label estimator) compute loss w.r.t. observed positives:
        loss_mtx_pos_2 = torch.zeros_like(Y_obs)
        loss_mtx_pos_2[Y_obs == 1] = neg_log(estimated_labels[Y_obs == 1])
        # (label estimator) compute loss w.r.t. image classifier outputs:
        preds_detached = preds.detach()
        loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(
            1.0 - estimated_labels)
        # (label estimator) compute regularizer:
        reg_2 = expected_positive_regularizer(estimated_labels, expected_num_pos, norm='2') / (Y_obs.size(1) ** 2)
        # compute final loss matrix:
        reg_loss = 0.5 * (reg_1 + reg_2)
        # reg_loss = None
        loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
        loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)

        return loss_mtx.mean() + reg_loss


def loss_EM(preds, Y_obs, args):
    # unpack:
    observed_labels = Y_obs

    # input validation:
    assert torch.min(observed_labels) >= 0

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -args.alpha * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
    )

    return loss_mtx.mean()


def loss_EM_APL(preds, Y_obs, args):
    # unpack:
    observed_labels = Y_obs

    # input validation:
    assert torch.min(observed_labels) >= -1

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -args.alpha * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
    )

    soft_label = -observed_labels[observed_labels < 0]
    loss_mtx[observed_labels < 0] = args.beta * (
            soft_label * neg_log(preds[observed_labels < 0]) +
            (1 - soft_label) * neg_log(1 - preds[observed_labels < 0])
    )
    return loss_mtx.mean()


def loss_VIB(preds, Y_obs, z_mu, z_std, args):
    observed_labels = Y_obs
    # input validation:
    assert torch.min(observed_labels) >= -1
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = -0.5 * (1 + 2 * z_std.log() - z_mu.pow(2) - z_std.pow(2)).mean() * args.b
    return loss_mtx.mean() + reg_loss


def pseudo_labeling_vib(model, train_loader, epoch, ulabel_num, args):
    model.eval()
    P = {}
    batch = {}
    total_preds = None
    total_idx = None
    # P['steps_per_epoch'] = len(Z['dataloaders'][phase])
    P['steps_per_epoch'] = len(train_loader)

    for i, (batch_X, batch_Y_obs, batch_Y, batch_idx) in enumerate(train_loader):

        P['batch'] = i

        # move data to GPU:
        # batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['image'] = batch_X.cuda()
        batch['idx'] = batch_idx

        # forward pass:
        with torch.set_grad_enabled(False):
            batch['logits'] = model(batch['image'])[0]
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)

        # gather:
        if P['batch'] == 0:
            total_preds = batch['preds'].detach().cpu().numpy()
            total_idx = batch['idx'].cpu().numpy()
        else:
            total_preds = np.vstack((batch['preds'].detach().cpu().numpy(), total_preds))
            total_idx = np.hstack((batch['idx'].cpu().numpy(), total_idx))
    label_mtx = np.zeros_like(total_preds)
    label_mtx[total_preds > args.t] = 1
    train_loader.dataset.label_matrix_obs[total_idx] += label_mtx
    train_loader.dataset.label_matrix_obs[train_loader.dataset.label_matrix_obs > 1] = 1


# combined with em+apl
def aysmmetric_pseudo_labeling(model, train_loader, epoch, ulabel_num, args):
    model.eval()
    P = {}
    batch = {}
    total_preds = None
    total_idx = None
    # P['steps_per_epoch'] = len(Z['dataloaders'][phase])
    P['steps_per_epoch'] = len(train_loader)

    for i, (batch_X, batch_Y_obs, batch_Y, batch_idx) in enumerate(train_loader):

        P['batch'] = i

        # move data to GPU:
        # batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['image'] = batch_X.cuda()
        batch['idx'] = batch_idx

        # forward pass:
        with torch.set_grad_enabled(False):
            batch['logits'] = model(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)

        # gather:
        if P['batch'] == 0:
            total_preds = batch['preds'].detach().cpu().numpy()
            total_idx = batch['idx'].cpu().numpy()
        else:
            total_preds = np.vstack((batch['preds'].detach().cpu().numpy(), total_preds))
            total_idx = np.hstack((batch['idx'].cpu().numpy(), total_idx))

            # pseudo-label:
            if P['batch'] >= P['steps_per_epoch'] - 1:

                for i in range(total_preds.shape[1]):  # class-wise

                    class_preds = total_preds[:, i]
                    class_labels_obs = train_loader.dataset.label_matrix_obs[:, i]
                    class_labels_obs = class_labels_obs[total_idx]

                    # select unlabel data:
                    unlabel_class_preds = class_preds[class_labels_obs == 0]
                    unlabel_class_idx = total_idx[class_labels_obs == 0]

                    # select samples:
                    neg_PL_num = int(args.neg_proportion * ulabel_num[i] / (epoch - args.warm_up))
                    sorted_idx_loc = np.argsort(unlabel_class_preds)  # ascending
                    selected_idx_loc = sorted_idx_loc[:neg_PL_num]  # select indices

                    # assgin soft labels:
                    for loc in selected_idx_loc:
                        train_loader.dataset.label_matrix_obs[unlabel_class_idx[loc], i] = -unlabel_class_preds[loc]


def set_pseudo_labeling(model, train_loader, epoch, args, tmp_obs, pseudo_label_matrix, train_Y):
    # assert phase == 'train'
    model.eval()
    P = {}
    batch = {}
    total_preds = None
    total_idx = None
    P['steps_per_epoch'] = len(train_loader)

    for i, (batch_X, batch_Y_obs, batch_Y, batch_idx) in enumerate(train_loader):

        P['batch'] = i
        # print(i)
        # move data to GPU:
        # batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['image'] = batch_X.cuda()
        batch['idx'] = batch_idx

        # forward pass:
        with torch.set_grad_enabled(False):
            batch['logits'] = model(batch['image'])[0]   #pl:model(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)

        # gather:
        if P['batch'] == 0:
            total_preds = batch['preds'].detach().cpu().numpy()
            total_idx = batch['idx'].cpu().numpy()
        else:
            total_preds = np.vstack((batch['preds'].detach().cpu().numpy(), total_preds))
            total_idx = np.hstack((batch['idx'].cpu().numpy(), total_idx))

            # pseudo-label:
            if P['batch'] >= P['steps_per_epoch'] - 1:
                total_right = 0
                total_estimate = count_estimate_(args)

                # pseudo_label_matrix = np.zeros_like(train_loader.dataset.label_matrix_obs)
                pseudo_label_matrix = np.zeros_like(train_loader.dataset.label_matrix_obs)
                for i in range(total_preds.shape[1]):  # class-wise

                    class_preds = total_preds[:, i]  # 某个类别的预测置信度
                    class_labels_obs = train_loader.dataset.label_matrix_obs[:, i]
                    class_labels_obs = class_labels_obs[total_idx]

                    # select unlabel data:
                    unlabel_class_preds = class_preds[class_labels_obs == 0]  # 筛选出未被标记的标记置信度
                    unlabel_class_idx = total_idx[class_labels_obs == 0]

                    class_estimate = total_estimate[i]

                    pseudo_positive_num = int(class_estimate * total_preds.shape[0])

                    # select samples:
                    sorted_idx_loc = np.flipud(np.argsort(unlabel_class_preds))
                    selected_idx_loc = sorted_idx_loc[:pseudo_positive_num]  # select indices

                    # label_num = 0
                    for j in range(0, total_preds.shape[0]):
                        if (tmp_obs[j, i] == 1):
                            pseudo_label_matrix[j, i] = 1
                            # label_num += 1
                    # print(label_num)

                    for loc in selected_idx_loc:
                        pseudo_label_matrix[unlabel_class_idx[loc], i] = -1

                    for k in range(0,total_preds.shape[0]):
                        if(pseudo_label_matrix[k, i]==train_Y[k,i]):
                            total_right += 1
                    # print("bp")

                print(total_right / (total_preds.shape[0] * total_preds.shape[1]))

    train_loader.dataset.label_matrix_obs = pseudo_label_matrix


def beta_kl_loss(alpha, beta, prior_value, eps=1e-6):
    prior = torch.ones_like(alpha) * prior_value
    alpha = alpha + eps
    beta = beta + eps
    KL = (torch.lgamma(alpha) + torch.lgamma(beta)) + \
         ((alpha - prior) * (torch.digamma(alpha)) + (beta - prior) * (torch.digamma(beta)))
    return KL.mean()


def vae_loss(args, batch_X, batch_Y_obs, batch_Y_score, batch_D_score_1, batch_D_score_2, batch_G_A, batch_X_rec,
             batch_Y_rec, batch_A_rec):
    # for encoder and decoder
    loss_align = weighted_bce_loss(batch_D_score_1, batch_Y_obs.clone().detach(), batch_D_score_2.clone().detach(),
                                   True)
    loss_recx = F.mse_loss(batch_X_rec, batch_X.clone().detach())
    loss_recy = weighted_bce_loss(batch_Y_rec, batch_Y_obs.clone().detach(), batch_Y_obs.clone().detach())
    loss_recA = F.mse_loss(batch_A_rec, batch_G_A.detach())
    loss_kl = beta_kl_loss(batch_D_score_1, 1 - batch_D_score_1, 1)
    loss_1 = loss_align + args.gamma * (loss_recx + loss_recy + loss_recA) + args.delta * loss_kl
    # for label estimator
    loss_base = weighted_bce_loss(batch_D_score_2, batch_Y_obs.clone().detach(), batch_D_score_1.clone().detach(), True)

    return loss_1 + args.base * loss_base


def loss_smile(args, preds, observed_labels, z_mu, z_std, distributions, d_alpha, d_beta):
    loss_mtx_D = torch.zeros_like(observed_labels)
    loss_mtx_D[observed_labels == 1] = neg_log(distributions[observed_labels == 1])
    loss_mtx_D[observed_labels == 0] = neg_log(1.0 - distributions[observed_labels == 0])
    loss_D = loss_mtx_D.mean()
    loss_mtx_pred = torch.zeros_like(observed_labels)
    loss_mtx_pred[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx_pred[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    loss_pred = loss_mtx_pred.mean()
    reg_z = -0.5 * (1 + 2 * z_std.log() - z_mu.pow(2) - z_std.pow(2)).mean()
    prior_alpha, prior_beta = torch.ones_like(d_alpha), torch.ones_like(d_alpha)
    reg_d = (torch.lgamma(d_alpha + d_beta) + torch.lgamma(prior_alpha) + torch.lgamma(prior_beta)
             - (torch.lgamma(prior_alpha + prior_beta) + torch.lgamma(d_alpha) + torch.lgamma(d_beta))
             + (d_alpha - prior_alpha) * torch.digamma(d_alpha)
             + (d_beta - prior_beta) * torch.digamma(d_beta)
             - (d_alpha - prior_alpha + d_beta - prior_beta) * torch.digamma(d_alpha + d_beta)).mean()
    loss_mtx_align = distributions * neg_log(preds) + (1 - distributions) * neg_log(1 - preds)
    loss_mtx_align[observed_labels == 0] = loss_mtx_align[observed_labels == 0] * 1
    loss_align = loss_mtx_align.mean()
    final_loss = \
        args.theta * loss_align + \
        args.alpha * loss_pred + \
        args.beta * loss_D + \
        args.rz * reg_z + args.rd * reg_d
    return final_loss


class LACLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(LACLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # left valid labels and features
        labels = labels[labels != 0]
        batch_size = labels.shape[0]
        features = features[:batch_size]

        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(torch.clamp(logits, -50, 50)) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
