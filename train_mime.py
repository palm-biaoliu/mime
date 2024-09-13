import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from lib_lagc.models.LEModel import build_LEModel_VIB
from utils import datasets
from utils.losses import compute_batch_loss
from utils.instrumentation import train_logger
from utils.thelog import initLogger
from utils.countExpected import count_estimate_

parser = argparse.ArgumentParser(description='ICML-2023')
parser.add_argument('-g', '--gpu', default=0, type=int)
parser.add_argument('-d', '--dataset', default='pascal', choices=['pascal', 'coco', 'nuswide', 'cub'], type=str)
parser.add_argument('-l', '--loss', default='VIB',
                    choices=['bce', 'bce_ls', 'iun', 'iu', 'pr', 'an', 'an_ls', 'wan', 'epr', 'role', 'VIB'], type=str)
parser.add_argument('-t', '--theta', default=0.95, type=float)
parser.add_argument('-b', '--beta', default=1e-4, type=float)
parser.add_argument('-z', '--z_dim', default=256, type=int)
parser.add_argument('-lr', default=1e-4, type=float)
parser.add_argument('-wd', default=1e-4, type=float)
parser.add_argument('-bs', default=32, type=int)
parser.add_argument('-T', default=1.0, type=float)


args = parser.parse_args()

# global logger
sys.stdout = open(os.devnull, 'w')
gb_logger, save_dir = initLogger(args, save_dir='results_mime/')


def run_train_phase(model, P, Z, logger, epoch, phase):
    '''
    Run one training phase.
    
    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training. 
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''

    assert phase == 'train'
    model.train()

    if epoch > P['warmup_epoch']:
        Z['datasets']['train'].label_matrix_obs = Z['pseudo_label_matrix_obs']
    for batch in Z['dataloaders'][phase]:
        # move data to GPU: 
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass: 
        Z['optimizer'].zero_grad()
        with torch.set_grad_enabled(True):
            # batch['logits'], batch['label_vec_est'] = model(batch)
            batch['logits'], batch['mus'], batch['stds'] = model(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'] / P['T'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch = compute_batch_loss(batch, P, Z)
        # backward pass:
        batch['loss_tensor'].backward()
        Z['optimizer'].step()
        # save current batch data:
        logger.update_phase_data(batch)
    if epoch > P['warmup_epoch']:
        Z['datasets']['train'].label_matrix_obs = Z['tmp_obs']


def run_eval_phase(model, P, Z, logger, epoch, phase):
    '''
    Run one evaluation phase.
    
    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training. 
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''

    assert phase in ['val', 'test']
    model.eval()
    for batch in Z['dataloaders'][phase]:
        # move data to GPU: 
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass:
        with torch.set_grad_enabled(False):
            batch['logits'], _, _ = model(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch['loss_np'] = -1
            batch['reg_loss_np'] = -1
        # save current batch data:
        logger.update_phase_data(batch)

def set_pseudo_labeling(model, P, Z, logger, epoch, phase):

    assert phase == 'train'
    model.train()

    total_preds = None
    total_idx = None
    P['steps_per_epoch'] = len(Z['dataloaders'][phase])

    for i, batch in enumerate(Z['dataloaders'][phase]):

        P['batch'] = i

        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)

        # forward pass:
        with torch.set_grad_enabled(False):
            batch['logits'],_,_ = model(batch['image'])
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
                total_estimate = count_estimate_(args)

                Z['pseudo_label_matrix_obs'] = np.zeros_like(Z['datasets']['train'].label_matrix_obs)
                for i in range(total_preds.shape[1]):  # class-wise

                    class_preds = total_preds[:, i]

                    # else:
                    class_labels_obs = Z['datasets']['train'].label_matrix_obs[:, i]
                    class_labels_obs = class_labels_obs[total_idx]

                    # # # select unlabel data:
                    unlabel_class_preds = class_preds[class_labels_obs == 0]
                    unlabel_class_idx = total_idx[class_labels_obs == 0]

                    class_estimate = total_estimate[i]

                    pseudo_positive_num = int(class_estimate * total_preds.shape[0])

                    # select samples:
                    sorted_idx_loc = np.flipud(np.argsort(unlabel_class_preds))
                    selected_idx_loc = sorted_idx_loc[:pseudo_positive_num]  # select indices

                    for j in range(0,total_preds.shape[0]):
                        if(Z['tmp_obs'][j,i]==1):
                            Z['pseudo_label_matrix_obs'][j, i] = 1

                    for loc in selected_idx_loc:
                        Z['pseudo_label_matrix_obs'][unlabel_class_idx[loc], i] = -1

def train(model, P, Z):
    '''
    Train the model.
    
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    '''

    # best_weights_f = copy.deepcopy(model.feature_extractor.state_dict())
    logger = train_logger(P)  # initialize logger

    for epoch in range(P['num_epochs']):
        gb_logger.info('Epoch {}/{}'.format(epoch, P['num_epochs'] - 1))
        P['epoch'] = epoch + 1
        for phase in ['train', 'val', 'test']:
            # reset phase metrics:
            logger.reset_phase_data()

            # run one phase:
            t_init = time.time()
            if phase == 'train':
                run_train_phase(model, P, Z, logger, P['epoch'], phase)
                if P['epoch'] >= P['warmup_epoch']:
                    set_pseudo_labeling(model, P, Z, logger, P['epoch'], phase)
            else:
                run_eval_phase(model, P, Z, logger, P['epoch'], phase)

            # save end-of-phase metrics:
            logger.compute_phase_metrics(phase, P['epoch'])

            # print epoch status:
            logger.report(t_init, time.time(), phase, P['epoch'], gb_logger)

            # update best epoch, if applicable:
            new_best = logger.update_best_results(phase, P['epoch'], P['val_set_variant'])
            if new_best:
                gb_logger.info('*** new best weights ***')

    gb_logger.info('')
    gb_logger.info('*** TRAINING COMPLETE ***')
    gb_logger.info('Best epoch: {}'.format(logger.best_epoch))
    gb_logger.info('Best epoch validation score: {:.2f}'.format(
        logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant'])))
    gb_logger.info(
        'Best epoch test score:       {:.2f}'.format(logger.get_stop_metric('test', logger.best_epoch, 'clean')))
    if 'best_weights_feature_extractor' not in locals():
        best_weights_feature_extractor = None
        best_weights_encoder_z = None
        best_weights_linear_classifier = None

    # return P, model, logger, best_weights_feature_extractor, best_weights_encoder_z, best_weights_linear_classifier
    return P, model, logger, None, None, None


def initialize_training_run(P, feature_extractor, linear_classifier, estimated_labels):
        '''
        Set up for model training.

        Parameters
        P: Dictionary of parameters, which completely specify the training procedure.
        feature_extractor: Feature extractor model to start from.
        linear_classifier: Linear classifier model to start from.
        estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
        '''

        np.random.seed(P['seed'])

        Z = {}

        # accelerator:
        Z['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data:
        Z['datasets'] = datasets.get_data(P)

        # observed label matrix:
        label_matrix = Z['datasets']['train'].label_matrix
        num_examples = int(np.shape(label_matrix)[0])
        mtx = np.array(label_matrix).astype(np.int8)
        total_pos = np.sum(mtx == 1)
        total_neg = np.sum(mtx == 0)
        gb_logger.info('training samples: {} total'.format(num_examples))
        gb_logger.info(
            'true positives: {} total, {:.2f} per example on average.'.format(total_pos, total_pos / num_examples))
        gb_logger.info(
            'true negatives: {} total, {:.2f} per example on average.'.format(total_neg, total_neg / num_examples))
        observed_label_matrix = Z['datasets']['train'].label_matrix_obs
        num_examples = int(np.shape(observed_label_matrix)[0])
        obs_mtx = np.array(observed_label_matrix).astype(np.int8)
        obs_total_pos = np.sum(obs_mtx == 1)
        obs_total_neg = np.sum(obs_mtx == 0)
        gb_logger.info('observed positives: {} total, {:.2f} per example on average.'.format(obs_total_pos,
                                                                                             obs_total_pos / num_examples))
        gb_logger.info('observed negatives: {} total, {:.2f} per example on average.'.format(obs_total_neg,
                                                                                             obs_total_neg / num_examples))

        Z['pseudo_label_matrix_obs'] = np.zeros_like(Z['datasets']['train'].label_matrix_obs)
        Z['tmp_obs'] = Z['datasets']['train'].label_matrix_obs

        # save dataset-specific parameters:
        P['num_classes'] = Z['datasets']['train'].num_classes


        # dataloaders:
        Z['dataloaders'] = {}
        for phase in ['train', 'val', 'test']:
            Z['dataloaders'][phase] = torch.utils.data.DataLoader(
                Z['datasets'][phase],
                batch_size=P['bsize'],
                shuffle=phase == 'train',
                sampler=None,
                num_workers=P['num_workers'],
                drop_last=True
            )

        # model:
        args = SimpleNamespace()
        args.dataset_name = P['dataset']
        args.backbone = P['arch']
        args.img_size = 448
        args.feat_dim = P['z_dim']
        model = build_LEModel_VIB(args)

        f_params = [param for param in list(model.backbone.parameters()) if param.requires_grad]
        g_params = [param for param in
                    list(model.encoder.parameters()) +
                    list(model.query_embed.parameters()) +
                    list(model.fc.parameters()) +
                    list(model.proj.parameters())
                    if param.requires_grad]
        opt_params = [
            {'params': f_params, 'lr': P['lr']},
            {'params': g_params, 'lr': 10 * P['lr']},
        ]

        Z['optimizer'] = getattr(torch.optim, 'Adamax')(
            opt_params,
            P['lr'],
            # betas=(0.9, 0.999), eps=1e-08,
            # weight_decay=P['wd']
        )

        return P, Z, model


def execute_training_run(P, feature_extractor, encoder_z_init, linear_classifier):
    '''
    Initialize, run the training process, and save the results.
    
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    print("initialize training run")
    P, Z, model = initialize_training_run(P, feature_extractor, encoder_z_init, linear_classifier)
    model.to(Z['device'])

    print("train")
    P, model, logger, best_weights_feature_extractor, best_weights_encoder_z, best_weights_linear_classifier = train(
        model, P, Z)

    final_logs = logger.get_logs()

    # return model.feature_extractor, model.encoder_z, model.linear_classifier, final_logs
    return None, None, None, final_logs


if __name__ == '__main__':
    print("start!")
    lookup = {
        'feat_dim': {
            'resnet50': 2048
        },
        'expected_num_pos': {
            'pascal': 1.5,
            'coco': 2.9,
            'nuswide': 1.9,
            'cub': 31.4
        },
    }

    P = {}

    # Top-level parameters:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    P['GPU'] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = P['GPU']
    P['dataset'] = args.dataset  # pascal, coco, nuswide, cub
    P['loss'] = args.loss  # bce, bce_ls, iun, iu, pr, an, an_ls, wan, epr, role, VIB
    P['train_mode'] = 'end_to_end'  # linear_fixed_features, end_to_end, linear_init
    P['val_set_variant'] = 'clean'  # clean, observed

    # Optimization parameters:
    P['lr_mult'] = 10.0  # learning rate multiplier for the parameters of g
    P['stop_metric'] = 'map'  # metric used to select the best epoch

    # Loss-specific parameters:
    P['ls_coef'] = 0.1  # label smoothing coefficient

    # Additional parameters:
    P['seed'] = 1200  # overall numpy seed
    P['use_pretrained'] = True  # True, False
    P['num_workers'] = 4

    # Dataset parameters:
    P['split_seed'] = 1200  # seed for train / val splitting
    P['val_frac'] = 0.2  # fraction of train set to split off for val
    P['ss_seed'] = 999  # seed for subsampling
    P['ss_frac_train'] = 1.0  # fraction of training set to subsample
    P['ss_frac_val'] = 1.0  # fraction of val set to subsample

    # Optimization parameters:
    P['sample_times'] = 5
    P['warmup_epoch'] = 2
    P['z_dim'] = 256
    if P['dataset'] == 'pascal':
        P['bsize'] = 32
        P['lr'] = 1e-4
        P['wd'] = 1e-4
        P['beta'] = 1e-4
        P['theta'] = 0.95
        P['z_dim'] = 256
        P['warmup_epoch'] = 2
    elif P['dataset'] == 'cub':
        P['bsize'] = 16
        P['lr'] = 1e-4
        P['wd'] = 1e-4
        P['warmup_epoch'] = 5
        P['beta'] = 1e-4
        P['theta'] = 0.95
        P['z_dim'] = 256
    elif P['dataset'] == 'nuswide':
        P['bsize'] = 16
        P['lr'] = 1e-4
        P['wd'] = 1e-4
        P['beta'] = 1e-4
        P['theta'] = 0.95
        P['z_dim'] = 256
    elif P['dataset'] == 'coco':
        P['bsize'] = 32
        P['lr'] = 1e-4
        P['wd'] = 1e-4
        P['beta'] = 1e-4
        P['theta'] = 0.95
        P['z_dim'] = 256
    P['T'] = args.T

    # Dependent parameters:
    if P['loss'] in ['bce', 'bce_ls']:
        P['train_set_variant'] = 'clean'
    else:
        P['train_set_variant'] = 'observed'

    P['num_epochs'] = 10
    P['freeze_feature_extractor'] = False
    P['use_feats'] = False
    P['arch'] = 'resnet50'

    P['feature_extractor_arch'] = 'resnet50'
    P['feat_dim'] = lookup['feat_dim'][P['feature_extractor_arch']]
    P['expected_num_pos'] = lookup['expected_num_pos'][P['dataset']]
    P['train_feats_file'] = './data/{}/train_features_imagenet_{}.npy'.format(P['dataset'], P['feature_extractor_arch'])
    P['val_feats_file'] = './data/{}/val_features_imagenet_{}.npy'.format(P['dataset'], P['feature_extractor_arch'])

    (feature_extractor, encoder_z, linear_classifier, logs) = execute_training_run(P,
                                                                                   feature_extractor=None,
                                                                                   encoder_z_init=None,
                                                                                   linear_classifier=None
                                                                                   )
