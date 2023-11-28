import torch
import numpy as np
from utils.countExpected import count_estimate_
import torch.nn.functional as F
import math


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


'''
loss functions
'''


def loss_bce(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss


def loss_bce_ls(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - P['ls_coef']) * neg_log(preds[observed_labels == 1]) + P[
        'ls_coef'] * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - P['ls_coef']) * neg_log(1.0 - preds[observed_labels == 0]) + P[
        'ls_coef'] * neg_log(preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss


def loss_iun(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[true_labels == 0] = neg_log(
        1.0 - preds[true_labels == 0])  # This loss gets unrealistic access to true negatives.
    reg_loss = None
    return loss_mtx, reg_loss


def loss_iu(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.any(observed_labels == 1)  # must have at least one observed positive
    assert torch.any(observed_labels == -1)  # must have at least one observed negative
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == -1] = neg_log(1.0 - preds[observed_labels == -1])
    reg_loss = None
    return loss_mtx, reg_loss


def loss_pr(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    batch_size = int(batch['label_vec_obs'].size(0))
    num_classes = int(batch['label_vec_obs'].size(1))
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    for n in range(batch_size):
        preds_neg = preds[n, :][observed_labels[n, :] == 0]
        for i in range(num_classes):
            if observed_labels[n, i] == 1:
                torch.nonzero(observed_labels[n, :])
                loss_mtx[n, i] = torch.sum(torch.clamp(1.0 - preds[n, i] + preds_neg, min=0))
    reg_loss = None
    return loss_mtx, reg_loss


def loss_an(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    # assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss

def loss_an_ls(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = (1.0 - P['ls_coef']) * neg_log(preds[observed_labels == 1]) + P[
        'ls_coef'] * neg_log(1.0 - preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = (1.0 - P['ls_coef']) * neg_log(1.0 - preds[observed_labels == 0]) + P[
        'ls_coef'] * neg_log(preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss


def loss_wan(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation: 
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0]) / float(P['num_classes'] - 1)
    reg_loss = None

    return loss_mtx, reg_loss


def loss_epr(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss w.r.t. observed positives:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # compute regularizer: 
    reg_loss = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    return loss_mtx, reg_loss


def loss_role(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    estimated_labels = batch['label_vec_est']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # (image classifier) compute loss w.r.t. observed positives:
    loss_mtx_pos_1 = torch.zeros_like(observed_labels)
    loss_mtx_pos_1[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # (image classifier) compute loss w.r.t. label estimator outputs:
    estimated_labels_detached = estimated_labels.detach()
    loss_mtx_cross_1 = estimated_labels_detached * neg_log(preds) + (1.0 - estimated_labels_detached) * neg_log(
        1.0 - preds)
    # (image classifier) compute regularizer: 
    reg_1 = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # (label estimator) compute loss w.r.t. observed positives:
    loss_mtx_pos_2 = torch.zeros_like(observed_labels)
    loss_mtx_pos_2[observed_labels == 1] = neg_log(estimated_labels[observed_labels == 1])
    # (label estimator) compute loss w.r.t. image classifier outputs:
    preds_detached = preds.detach()
    loss_mtx_cross_2 = preds_detached * neg_log(estimated_labels) + (1.0 - preds_detached) * neg_log(
        1.0 - estimated_labels)
    # (label estimator) compute regularizer:
    reg_2 = expected_positive_regularizer(estimated_labels, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    # compute final loss matrix:
    reg_loss = 0.5 * (reg_1 + reg_2)
    loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
    loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)

    return loss_mtx, reg_loss


def loss_EM(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true'].to(Z['device'])

    # input validation:
    assert torch.min(observed_labels) >= 0

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -P['alpha'] * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
    )

    return loss_mtx, None


def loss_EM_APL(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']

    # input validation:
    assert torch.min(observed_labels) >= -1

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -P['alpha'] * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
    )

    soft_label = -observed_labels[observed_labels < 0]
    loss_mtx[observed_labels < 0] = P['beta'] * (
            soft_label * neg_log(preds[observed_labels < 0]) +
            (1 - soft_label) * neg_log(1 - preds[observed_labels < 0])
    )
    return loss_mtx, None


def loss_VIB(batch, P, Z):
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    z_mu = batch['mus']
    z_std = batch['stds']
    # input validation:
    # assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = -0.5 * (1 + 2 * z_std.log() - z_mu.pow(2) - z_std.pow(2)).mean() * P['beta']
    return loss_mtx, reg_loss

def loss_warm(batch, P, Z):
    preds = batch['preds']
    distributions = batch['distributions']
    observed_labels = batch['label_vec_obs']
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    batch['loss_tensor'] = loss_mtx.mean()

    loss_mtx_D = torch.zeros_like(observed_labels)
    loss_mtx_D[observed_labels == 1] = neg_log(distributions[observed_labels == 1])
    loss_mtx_D[observed_labels == 0] = neg_log(1.0 - distributions[observed_labels == 0])
    batch['loss_tensor_D'] = loss_mtx_D.mean()

    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()
    batch['reg_loss_np'] = 0.0
    return batch


def loss_smile(batch, P, Z):
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    z_mu = batch['mus']
    z_std = batch['stds']
    distributions = batch['distributions']
    d_alpha = batch['alphas']
    d_beta = batch['betas']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx_D = torch.zeros_like(observed_labels)
    # loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    loss_mtx_D[observed_labels == 1] = neg_log(distributions[observed_labels == 1])
    loss_mtx_D[observed_labels == 0] = neg_log(1.0 - distributions[observed_labels == 0])
    # loss_mtx_D[observed_labels == 1] = (1 - 0.1) * neg_log(distributions[observed_labels == 1]) + \
    #                                    0.1 * neg_log(1.0 - distributions[observed_labels == 1])
    # loss_mtx_D[observed_labels == 0] = (1 - 0.1) * neg_log(1.0 - distributions[observed_labels == 0]) + \
    #                                    0.1 * neg_log(distributions[observed_labels == 0])
    loss_D = loss_mtx_D.mean()
    loss_mtx_pred = torch.zeros_like(observed_labels)
    loss_mtx_pred[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx_pred[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    # loss_mtx_pred[observed_labels == 1] = (1 - 0.1) * neg_log(preds[observed_labels == 1]) + \
    #                                       0.1 * neg_log(1.0 - preds[observed_labels == 1])
    # loss_mtx_pred[observed_labels == 0] = (1 - 0.1) * neg_log(1.0 - preds[observed_labels == 0]) + \
    #                                       0.1 * neg_log(preds[observed_labels == 0])
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
    # loss_align = (preds * neg_log(distributions) + (1 - preds) * neg_log(1 - distributions)).mean()
    batch['loss_tensor'] = \
        P['theta'] * loss_align + \
        P['alpha'] * loss_pred + \
        P['beta'] * loss_D + \
        1e-5 * reg_z + 1e-5 * reg_d
    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()
    batch['reg_loss_np'] = 0.0
    return batch


def loss_ll(batch, P, Z):
    preds = batch['logits']
    label_vec = batch['label_vec_obs']

    assert preds.dim() == 2

    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))

    unobserved_mask = (label_vec == 0)

    # compute loss for each image and class:
    loss_matrix = F.binary_cross_entropy_with_logits(preds, label_vec.clip(0), reduction='none')
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(preds, torch.logical_not(label_vec.clip(0)).float(),
                                                               reduction='none')

    correction_idx = None

    if P['clean_rate'] == 1:  # if epoch is 1, do not modify losses
        final_loss_matrix = loss_matrix
    else:
        if P['mod_scheme'] == 'LL-Cp':
            k = math.ceil(batch_size * num_classes * P['delta_rel'])
        else:
            k = math.ceil(batch_size * num_classes * (1 - P['clean_rate']))

        unobserved_loss = unobserved_mask.bool() * loss_matrix
        topk = torch.topk(unobserved_loss.flatten(), k)
        topk_lossvalue = topk.values[-1]
        correction_idx = torch.where(unobserved_loss > topk_lossvalue)
        if P['mod_scheme'] in ['LL-Ct', 'LL-Cp']:
            final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
        else:
            zero_loss_matrix = torch.zeros_like(loss_matrix)
            final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)

    main_loss = final_loss_matrix.mean()
    batch['loss_tensor'] = main_loss
    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()
    batch['reg_loss_np'] = 0.0

    return batch, correction_idx

loss_functions = {
    'bce': loss_bce,
    'bce_ls': loss_bce_ls,
    'iun': loss_iun,
    'iu': loss_iu,
    'pr': loss_pr,
    'an': loss_an,
    'an_ls': loss_an_ls,
    'wan': loss_wan,
    'epr': loss_epr,
    'role': loss_role,
    'EM': loss_EM,
    'EM_APL': loss_EM_APL,
    'VIB': loss_VIB,
}

'''
top-level wrapper
'''


def compute_batch_loss(batch, P, Z):
    assert batch['preds'].dim() == 2

    batch_size = int(batch['preds'].size(0))
    num_classes = int(batch['preds'].size(1))

    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(batch['preds'])
    # print(batch['label_vec_obs'].dtype)
    # print(torch.max(batch['label_vec_obs']))
    # input validation:
    assert torch.max(batch['label_vec_obs']) <= 1
    assert torch.min(batch['label_vec_obs']) >= -1
    assert batch['preds'].size() == batch['label_vec_obs'].size()
    assert P['loss'] in loss_functions

    # validate predictions:
    assert torch.max(batch['preds']) <= 1
    assert torch.min(batch['preds']) >= 0

    # compute loss for each image and class:
    loss_mtx, reg_loss = loss_functions[P['loss']](batch, P, Z)
    main_loss = (loss_mtx / loss_denom_mtx).sum()

    if reg_loss is not None:
        batch['loss_tensor'] = main_loss + reg_loss
        batch['reg_loss_np'] = reg_loss.clone().detach().cpu().numpy()
    else:
        batch['loss_tensor'] = main_loss
        batch['reg_loss_np'] = 0.0
    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()

    return batch
