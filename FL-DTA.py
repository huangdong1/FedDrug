import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from src.utils import *
import crypten
import csv
import time


def initialize_model(model, global_params):
    """
    Initialize the model's parameters with the global parameters.
    """
    for local_param, global_param in zip(model.parameters(), global_params):
        local_param.data = global_param.clone().detach().to(local_param.device)


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()


# training function at each epoch
def fl_train(train_sets, fl_models, fl_optimizers, params, epoch, cli_num):
    new_params = list()
    for k in range(len(train_sets)):
        fl_models[k].train()
        for batch_idx, data in enumerate(train_sets[k]):
            data, target = data.to(device), data.y.view(-1, 1).float().to(device)
            fl_optimizers[k].zero_grad()
            output = fl_models[k](data)
            loss = loss_fn(output, target)
            loss.backward()
            fl_optimizers[k].step()
        # print('Train epoch: {}, Client: {}, Loss: {:.6f}'.format(epoch, k+1, loss.item()))

    for param_i in range(len(params[0])):
        fl_params = list()
        for remote_index in range(cli_num):
            clone_param = params[remote_index][param_i].clone().cpu()
            fl_params.append(crypten.cryptensor(torch.as_tensor(clone_param)))
            # fl_params.append(torch.as_tensor(clone_param))
            # fl_params.append(crypten.cryptensor(clone_param.clone().detach().requires_grad_(True)))
        sign = 0
        for i in fl_params:
            if sign == 0:
                fl_param = i
                sign = 1
            else:
                fl_param = fl_param + i

        new_param = (fl_param / cli_num).get_plain_text()
        # new_param = fl_param / cli_num
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


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def define_network(cli_num, model, device, lr_=0.05, momentum_=0.9, weight_decay_=0.0001):
    createVar = locals()
    optimizers = []
    models = []
    params = []
    for i in range(cli_num):
        k = str(i + 1)
        model_name = 'model_' + k
        opti_name = 'optimizer_' + k

        createVar[model_name] = model().to(device)
        # createVar[model_name] = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
        createVar[opti_name] = torch.optim.SGD(locals()[model_name].parameters(),
                                               lr=lr_, momentum=momentum_, weight_decay=weight_decay_)
        # createVar[opti_name] = torch.optim.Adam(locals()[model_name].parameters(), lr=lr_)
        # createVar[opti_name] = torch.optim.SGD(locals()[model_name].parameters(), lr=lr_)
        models.append(locals()[model_name])
        params.append(list(locals()[model_name].parameters()))
        optimizers.append(locals()[opti_name])
    return models, optimizers, params


datasets = ['davis', 'kiba']
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][0]
model_st = modeling.__name__
crypten.init()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=3000, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--client_number', type=int, default=8)
parser.add_argument('--ensemble_number', type=int, default=8)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])

args = parser.parse_args()

cuda_name = "cuda:1"
NORMALISATION = 'bn'
TRAIN_BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.n_epochs
NUM_CLIENTS = args.client_number
LR = args.lr



train_loaders = []
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    root = './data/' + dataset

    # Local learning
    processed_data_file_test = './data/processed/' + dataset + '_test.pt'
    test_data = TestbedDataset(root='data', dataset=dataset + '_test', path=processed_data_file_test)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    for cli in range(NUM_CLIENTS):
        processed_data_file_train = './data/processed/' + dataset + '_train_' + str(cli + 1) + '.pt'
        train_data = TestbedDataset(root='data', dataset=dataset + '_train', path=processed_data_file_train)
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0001)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset + '.model'
        result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)
            G, P = predicting(model, device, test_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
            if ret[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_ci = ret[-1]
                if epoch >= 0:
                    print('client: ', cli + 1, 'rmse improved at epoch: ', best_epoch, ', best_mse: ', best_mse,
                          ', best_ci: ', best_ci)
            # else:
            #     print('client: ', cli+1, 'No improvement since epoch: ', best_epoch, ', best_mse: ', best_mse, ', best_ci: ', best_ci)


    # Federated learning
    # loading training data
    for cli in range(NUM_CLIENTS):
        processed_data_file_train = './data/processed/' + dataset + '_train_' + str(cli+1) + '.pt'
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', path=processed_data_file_train)
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
        train_loaders.append(train_loader)

    # loading test data
    processed_data_file_test = './data/processed/' + dataset + '_test.pt'
    test_data = TestbedDataset(root='data', dataset=dataset+'_test', path=processed_data_file_test)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
    fl_models, fl_optimizers, params = define_network(NUM_CLIENTS, model=modeling, device=device, lr_=LR)
    print("start training...")
    for ep in range(NUM_EPOCHS):
        start = time.time()
        fl_models = fl_train(train_loaders, fl_models, fl_optimizers, params, ep, NUM_CLIENTS)
        flag = 0
        for i in range(NUM_CLIENTS):
            G, P = predicting(fl_models[i], device, test_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
            if ret[4] > best_ci:
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = ep + 1
                best_mse = ret[1]
                best_ci = ret[-1]
                flag = 1
        end = time.time()
        print('time: ', (end - start))
        if flag == 1:
            print('Epoch: ', ep, ', C-index improved at epoch: ', best_epoch, ', best_mse: ', best_mse, ', best_ci: ', best_ci)
        else:
            print('Epoch: ', ep, ', No improvement since epoch: ', best_epoch, ', best_mse: ', best_mse, ', best_ci: ', best_ci)

        with open('result_fl_davis_nocrypten_' + str(NUM_CLIENTS) + '.csv', 'a+') as csvfile:
            # f.write(str(ep) + ',' + str(ret[1]) + ',' + str(ret[4]) + '\n')
            writer = csv.writer(csvfile)
            writer.writerow([ep+1, round(ret[1], 4), round(ret[4], 4)])

