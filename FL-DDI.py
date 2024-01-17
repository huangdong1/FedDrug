from datetime import datetime
import time
import argparse

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
# import crypten

from models.ssi_ddi import SSI_DDI
from src import custom_loss
from data_preprocessing_ddi import DrugDataset, DrugDataLoader, TOTAL_ATOM_FEATS


def do_compute(model, batch, device, training=True):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)

    acc = metrics.accuracy_score(target, pred)
    auc_roc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)

    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    auc_prc = metrics.auc(r, p)

    return acc, auc_roc, auc_prc


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device):
    print('Starting training at', datetime.today())
    best_epoch, best_acc, best_roc, best_prc = 0, 0, 0, 0
    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(model, batch, device)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_auc_prc = do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(model, batch, device)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_auc_prc = do_compute_metrics(val_probas_pred, val_ground_truth)

        if val_acc > best_acc:
            best_epoch = i
            best_acc = val_acc
            best_roc = val_auc_roc
            best_prc = val_auc_prc

        # if i % 10 == 0:
        print(f'\tepoch: {i}, best_epoch: {best_epoch}, best_acc: {best_acc:.4f}, best_roc: {best_roc:.4f}, best_prc: {best_prc:.4f}')


def fl_train(fl_models, train_data_loaders, val_data_loader, loss_fn, fl_optimizers, epoch, device, params, cli_num):
    new_params = list()
    for k in range(cli_num):
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []
        train_loss = 0
        val_loss = 0
        fl_models[k].train()
        for batch in train_data_loaders[k]:
            p_score, n_score, probas_pred, ground_truth = do_compute(fl_models[k], batch, device)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            fl_optimizers[k].zero_grad()
            loss.backward()
            fl_optimizers[k].step()
            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data_loaders[k])

        train_probas_pred = np.concatenate(train_probas_pred)
        train_ground_truth = np.concatenate(train_ground_truth)
        train_acc, train_auc_roc, train_auc_prc = do_compute_metrics(train_probas_pred, train_ground_truth)

        print(f'\tclient: {k+1}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc_roc: {train_auc_roc:.4f}, train_auc_prc: {train_auc_prc:.4f}')
        # with torch.no_grad():
        #     train_probas_pred = np.concatenate(train_probas_pred)
        #     train_ground_truth = np.concatenate(train_ground_truth)
        #
        #     train_acc, train_auc_roc, train_auc_prc = do_compute_metrics(train_probas_pred, train_ground_truth)
        #
        #     for batch in val_data_loader:
        #         fl_models[k].eval()
        #         p_score, n_score, probas_pred, ground_truth = do_compute(fl_models[k], batch, device)
        #         val_probas_pred.append(probas_pred)
        #         val_ground_truth.append(ground_truth)
        #         loss, loss_p, loss_n = loss_fn(p_score, n_score)
        #         val_loss += loss.item() * len(p_score)
        #
        #     val_loss /= len(val_data)
        #     val_probas_pred = np.concatenate(val_probas_pred)
        #     val_ground_truth = np.concatenate(val_ground_truth)
        #     val_acc, val_auc_roc, val_auc_prc = do_compute_metrics(val_probas_pred, val_ground_truth)
        # print(f'Epoch: {epoch}, client: {k}, train_loss: {train_loss:.4f}, val_loss  : {val_loss:.4f}')
        # print(f'\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f}, train_auprc: {train_auc_prc:.4f}')
        # print(f'\tval_acc  : {val_acc:.4f}, val_roc  : {val_auc_roc:.4f}, val_auprc  : {val_auc_prc:.4f}')

    for param_i in range(len(params[0])):
        fl_params = list()
        for remote_index in range(cli_num):
            clone_param = params[remote_index][param_i].clone().cpu()
            # fl_params.append(crypten.cryptensor(torch.as_tensor(clone_param)))
            fl_params.append(torch.as_tensor(clone_param))
            # fl_params.append(crypten.cryptensor(clone_param.clone().detach().requires_grad_(True)))
        sign = 0
        for i in fl_params:
            if sign == 0:
                fl_param = i
                sign = 1
            else:
                fl_param = fl_param + i

        # new_param = (fl_param / cli_num).get_plain_text()
        new_param = fl_param / cli_num
        new_params.append(new_param)

    with torch.no_grad():
        for model_para in params:
            for param in model_para:
                param *= 0

        for remote_index in range(cli_num):
            for param_index in range(len(params[remote_index])):
                new_params[param_index] = new_params[param_index].to(device)
                params[remote_index][param_index].set_(new_params[param_index])
    return fl_models


def predicting(model, val_data_loader, device, loss_fn):
    model.eval()
    val_probas_pred = []
    val_ground_truth = []
    val_loss = 0
    with torch.no_grad():
        for batch in val_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(model, batch, device)
            val_probas_pred.append(probas_pred)
            val_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            val_loss += loss.item() * len(p_score)

    val_loss /= len(val_data_loader)
    val_probas_pred = np.concatenate(val_probas_pred)
    val_ground_truth = np.concatenate(val_ground_truth)
    val_acc, val_auc_roc, val_auc_prc = do_compute_metrics(val_probas_pred, val_ground_truth)
    return val_acc, val_auc_roc, val_auc_prc


def define_network(cli_num, device, lr_=0.05, momentum_=0.9, weight_decay_=0.0001):
    createVar = locals()
    optimizers = []
    models = []
    params = []
    for i in range(cli_num):
        k = str(i + 1)
        model_name = 'model_' + k
        opti_name = 'optimizer_' + k

        createVar[model_name] = SSI_DDI(TOTAL_ATOM_FEATS, 64, 64, 86, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2]).to(device)
        createVar[opti_name] = torch.optim.SGD(locals()[model_name].parameters(), lr=lr_, momentum=momentum_, weight_decay=weight_decay_)
        models.append(locals()[model_name])
        params.append(list(locals()[model_name].parameters()))
        optimizers.append(locals()[opti_name])
    return models, optimizers, params


parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=TOTAL_ATOM_FEATS, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=64, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=2000, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=64, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--client_number', type=int, default=8)
parser.add_argument('--ensemble_number', type=int, default=8)

args = parser.parse_args()
# crypten.init()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
cli_num = args.client_number
ens_num = args.ensemble_number
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# Loading dataset
df_ddi_train = pd.read_csv('data/ddi_training.csv')
df_ddi_val = pd.read_csv('data/ddi_validation.csv')
df_ddi_test = pd.read_csv('data/ddi_test.csv')

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

# Divide into 32 parts
cli_count = len(train_tup)//32

val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size)

# print(f"Training with {len(train_tup)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

loss = custom_loss.SigmoidLoss()

# Local learning
model = SSI_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2])
# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model.to(device=device)
for cli in range(32):
    print(f'Local learning, Client Num: {cli+1}/32')
    train_data = DrugDataset(train_tup[cli_count * cli:cli_count * (cli + 1)], ratio=data_size_ratio, neg_ent=neg_samples)
    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True)
    train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device)

# Centralized learning
ens_count = cli_count*ens_num
model = SSI_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2])
# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model.to(device=device)
for cen in range(32//ens_num):
    print(f'Centralized learning, ensemble_number: {ens_num}, Centralized Num: {cen+1}/{32//ens_num}')
    train_data = DrugDataset(train_tup[ens_count * cen:ens_count * (cen + 1)], ratio=data_size_ratio, neg_ent=neg_samples)
    train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True)
    train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device)

# Federated learning
train_data_loaders = []
for cli in range(cli_num):
    if cli < 31:
        train_data = DrugDataset(train_tup[cli_count*cli:cli_count*(cli+1)], ratio=data_size_ratio, neg_ent=neg_samples)
    else:
        train_data = DrugDataset(train_tup[cli_count * cli:], ratio=data_size_ratio, neg_ent=neg_samples)
    train_data_loaders.append(DrugDataLoader(train_data, batch_size=batch_size, shuffle=True))

fl_models, fl_optimizers, params = define_network(cli_num, device=device, lr_=lr)
best_epoch, best_acc, best_roc, best_prc = 0, 0, 0, 0
for ep in range(n_epochs):
    print(f'Federated learning, Epoch: {ep+1}, Client Num: {cli_num}')
    fl_models = fl_train(fl_models, train_data_loaders, val_data_loader, loss, fl_optimizers, ep, device, params, cli_num)
    val_acc, val_auc_roc, val_auc_prc = predicting(fl_models[0], test_data_loader, device, loss)
    if val_acc > best_acc:
        best_epoch = ep+1
        best_acc = val_acc
        best_roc = val_auc_roc
        best_prc = val_auc_prc
    print(f'Testing. best_epoch: {best_epoch}, best_acc: {best_acc:.4f}, best_roc: {best_roc:.4f}, best_prc: {best_prc:.4f}')
