import os
import sys
import time

from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch_geometric_signed_directed.utils import link_class_split, in_out_degree
from torch_geometric_signed_directed.data import load_signed_real_data, SignedData
from torch_geometric_signed_directed.nn.signed import SGCN, SDGNN, SiGAT, SNEA

from link_sign_direction_prediction_logistic_function import link_sign_direction_prediction_logistic_function

from MSGNN import MSGNN_link_prediction,GeneEmbeddingModel

from SSSNET_link_prediction import SSSNET_link_prediction

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'SigMaNet'))
from Signum import SigMaNet_link_prediction_one_laplacian
import laplacian
from parser_link import parameter_parser

args = parameter_parser()

from sklearn.preprocessing import label_binarize


def gaussian_enhance(adj_matrix, mean=0, std=0.1, lower_bound=-1, upper_bound=1):
    enhanced_matrix = adj_matrix.copy()
    zero_indices = np.where(enhanced_matrix == 0)
    gaussian_noise = np.random.normal(mean, std, size=len(zero_indices[0]))
    enhanced_noise = np.clip(gaussian_noise, lower_bound, upper_bound)
    enhanced_matrix[zero_indices] = enhanced_noise
    return enhanced_matrix


def calculate_metrics(test_y, pred, out, num_classes):
    test_y_bin = label_binarize(test_y, classes=range(num_classes))
    prob = torch.softmax(out, dim=1).cpu().numpy()
    auc = metrics.roc_auc_score(test_y_bin, prob, average='weighted', multi_class='ovr')
    precision = metrics.precision_score(test_y, pred, average='weighted')
    f1 = metrics.f1_score(test_y, pred, average='weighted')
    mcc = metrics.matthews_corrcoef(test_y, pred)
    recall = metrics.recall_score(test_y, pred, average='weighted')
    return f1, auc,  precision, mcc, recall


def train_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges, all_embeddings):
    model.train()
    out = model(X_real, X_img, edge_index=edge_index,
                query_edges=query_edges,
                edge_weight=edge_weight,all_embeddings=all_embeddings)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc





def test_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges, all_embeddings,num_classes):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, edge_index=edge_index,
                    query_edges=query_edges,
                    edge_weight=edge_weight,all_embeddings=all_embeddings)
    test_y = y.cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()

    test_acc = metrics.accuracy_score(test_y, pred)
    f1_macro = metrics.f1_score(test_y, pred, average='macro')
    f1_micro = metrics.f1_score(test_y, pred, average='micro')
    f1, auc,  precision, mcc, recall= calculate_metrics(test_y, pred, out, num_classes)
    return test_acc, f1_macro, f1_micro, f1, auc,  precision, mcc, recall

def train_SSSNET(features, edge_index_p, edge_weight_p,
                 edge_index_n, edge_weight_n, query_edges, y):
    model.train()
    out = model(edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, features, query_edges)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc


def test_SSSNET(features, edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, query_edges, y):
    model.eval()
    with torch.no_grad():
        out = model(edge_index_p, edge_weight_p,
                    edge_index_n, edge_weight_n, features, query_edges)
    test_y = y.cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()
    test_acc = metrics.accuracy_score(test_y, pred)
    f1_macro = metrics.f1_score(test_y, pred, average='macro')
    f1_micro = metrics.f1_score(test_y, pred, average='micro')
    return test_acc, f1_macro, f1_micro


def test(train_X, test_X, train_y, test_y):
    model.eval()
    with torch.no_grad():
        z = model()

    embeddings = z.cpu().numpy()
    accuracy, f1_macro, f1_micro = link_sign_direction_prediction_logistic_function(embeddings, train_X, train_y,test_X, test_y)
    return accuracy, f1_macro, f1_micro


def train():
    model.train()
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_SigMaNet(X_real, X_img, y, query_edges):
    model.train()
    with torch.no_grad():
        out = model(X_real, X_img, query_edges)
    loss = criterion(out, y)
    loss.requires_grad = True
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc


def test_SigMaNet(X_real, X_img, y, query_edges):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, query_edges)
    test_y = y.cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()
    test_acc = metrics.accuracy_score(test_y, pred)
    f1_macro = metrics.f1_score(test_y, pred, average='macro')
    f1_micro = metrics.f1_score(test_y, pred, average='micro')
    return test_acc, f1_macro, f1_micro


device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
if args.dataset not in ['pvCLCL', 'OPCL']:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.dataset)
    data = load_signed_real_data(dataset=args.dataset, root=path).to(device)
else:
    args.dataset += str(args.year)
    save_path = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/Lead_Lag', args.dataset + '_10p2.npz')
    A = sp.load_npz(save_path)
    data = SignedData(A=A)

sub_dir_name = 'runs' + str(args.runs) + 'epochs' + str(args.epochs) + \
               '100train_ratio' + str(int(100 * args.train_ratio)) + '100val_ratio' + str(int(100 * args.val_ratio)) + \
               '1000lr' + str(int(1000 * args.lr)) + '1000weight_decay' + str(
    int(1000 * args.weight_decay)) + '100dropout' + str(int(100 * args.dropout))
if args.seed != 0:
    sub_dir_name += 'seed' + str(args.seed)
if args.method == 'MSGNN':
    suffix = 'K' + str(args.K) + '100q' + str(int(100 * args.q)) + 'trainable_q' + str(args.trainable_q) + \
             'hidden' + str(args.hidden)
elif args.method == 'SigMaNet':
    suffix = 'K1_netflow'
elif args.method == 'SSSNET':
    suffix = 'hidden' + str(args.hidden) + 'hop' + str(args.hop) + '100tau' + str(int(100 * args.tau))
else:
    suffix = 'in_dim' + str(args.in_dim) + 'out_dim' + str(args.out_dim)

if args.method in ['SSSNET', 'SigMaNet', 'MSGNN']:
    num_input_feat = 2
    if args.sd_input_feat:
        suffix += 'SdInput'
        num_input_feat = 4
    if args.weighted_input_feat:
        suffix += 'WeightedInput'
        if args.weighted_nonnegative_input_feat:
            suffix += 'nonnegative'
    if args.input_unweighted:
        suffix += 'InputUnweighted'
if args.method == 'MSGNN':
    if args.normalization == 'None':
        args.normalization = None
        suffix += 'no_norm'
    if args.num_layers != 2:
        suffix += 'num_layers' + str(args.num_layers)

logs_folder_name = 'runs'
if args.debug:
    args.runs = 2
    args.epochs = 2
    logs_folder_name = 'debug_runs'
log_dir = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '../' + logs_folder_name, args.dataset, args.method, sub_dir_name)
writer = SummaryWriter(log_dir=log_dir + '/' + suffix)

if args.num_classes == 4:
    task = "four_class_signed_digraph"
elif args.num_classes == 5:
    task = "five_class_signed_digraph"

save_data_path_dir = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), '../data', args.dataset)
if args.direction_only_task:
    save_data_path = os.path.join(save_data_path_dir,
                                  task + '_direction_only' + str(device) + 'seed' + str(args.seed) + 'split' + str(
                                      args.runs) + '100val' + str(int(100 * args.val_ratio)) + '100train' + str(
                                      int(100 * args.train_ratio)) + '.pt')
else:
    save_data_path = os.path.join(save_data_path_dir, task + str(device) + 'seed' + str(args.seed) + 'split' + str(
        args.runs) + '100val' + str(int(100 * args.val_ratio)) + '100train' + str(int(100 * args.train_ratio)) + '.pt')
if os.path.exists(save_data_path):
    print('Loading existing data splits!')
    link_data = torch.load(open(save_data_path, 'rb'))
else:
    link_data = link_class_split(data, splits=args.runs, task=task, prob_val=args.val_ratio,
                                 prob_test=1 - args.train_ratio - args.val_ratio, seed=args.seed, device=device)
    if os.path.isdir(save_data_path_dir) == False:
        try:
            os.makedirs(save_data_path_dir)
        except FileExistsError:
            print('Folder exists for {}!'.format(save_data_path_dir))
    torch.save(link_data, save_data_path)
if args.direction_only_task:
    task += '_direction_only'
    args.num_classes -= 2

nodes_num = data.num_nodes
in_dim = args.in_dim
out_dim = args.out_dim

start = time.time()
criterion = nn.NLLLoss()
res_array = np.zeros((args.runs, 8))
for split in list(link_data.keys()):
    edge_index = link_data[split]['graph']
    edge_weight = link_data[split]['weights']


    edge_weight = gaussian_enhance(edge_weight.numpy(), mean=0, std=0.1, lower_bound=-1, upper_bound=1)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32, device=device)

    if args.input_unweighted:
        edge_weight = torch.where(edge_weight > 0, 1, -1)
    if args.q in [11, 22, 33, 44, 55]:
        curr_graph = SignedData(edge_index=edge_index, edge_weight=edge_weight)
        args.q = args.q / 55 * 0.5 / (curr_graph.A - curr_graph.A.transpose()).max()

    edge_i_list = edge_index.t().cpu().numpy().tolist()
    edge_weight_s = torch.where(edge_weight > 0, 1, -1)
    edge_s_list = edge_weight_s.long().cpu().numpy().tolist()
    edge_index_s = torch.LongTensor([[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)]).to(device)

    query_edges = link_data[split]['train']['edges']
    y = link_data[split]['train']['label']
    if args.direction_only_task:
        y = torch.div(y, 2, rounding_mode='floor').to(device)
    if args.weighted_input_feat:
        if args.weighted_nonnegative_input_feat:
            X_real = in_out_degree(edge_index, size=int(edge_index.max() - edge_index.min()) + 1,
                                   signed=args.sd_input_feat, \
                                   edge_weight=torch.abs(edge_weight)).to(device)
        else:
            X_real = in_out_degree(edge_index, size=int(edge_index.max() - edge_index.min()) + 1,
                                   signed=args.sd_input_feat, \
                                   edge_weight=edge_weight).to(device)
    else:
        if args.sd_input_feat:
            data1 = SignedData(edge_index=edge_index, edge_weight=edge_weight).to(device)
            data1.separate_positive_negative()
            x1 = in_out_degree(data1.edge_index_p, size=int(data1.edge_index.max() - data1.edge_index.min()) + 1).to(
                device)
            x2 = in_out_degree(data1.edge_index_n, size=int(data1.edge_index.max() - data1.edge_index.min()) + 1).to(
                device)
            X_real = torch.concat((x1, x2), 1)
        else:
            X_real = in_out_degree(edge_index, size=int(edge_index.max() - edge_index.min()) + 1,
                                   signed=args.sd_input_feat).to(device)

    X_img = X_real.clone()
    if args.method == 'SGCN':
        model = SGCN(nodes_num, edge_index_s, in_dim, out_dim, layer_num=2, lamb=5).to(device)
    elif args.method == 'SNEA':
        model = SNEA(nodes_num, edge_index_s, in_dim, out_dim, layer_num=2, lamb=5).to(device)
    elif args.method == 'SiGAT':
        model = SiGAT(nodes_num, edge_index_s, in_dim, out_dim).to(device)
    elif args.method == 'SDGNN':
        model = SDGNN(nodes_num, edge_index_s, in_dim, out_dim).to(device)
    elif args.method == 'MSGNN':
        model = MSGNN_link_prediction(q=args.q, K=args.K, num_features=num_input_feat, hidden=args.hidden,
                                      label_dim=args.num_classes, \
                                      trainable_q=args.trainable_q, layer=args.num_layers, dropout=args.dropout,
                                      normalization=args.normalization, cached=(not args.trainable_q)).to(device)
    elif args.method == 'SSSNET':
        model = SSSNET_link_prediction(nfeat=num_input_feat, hidden=args.hidden, nclass=args.num_classes,
                                       dropout=args.dropout,
                                       hop=args.hop, fill_value=args.tau, directed=data.is_directed).to(device)
        data1 = SignedData(edge_index=edge_index, edge_weight=edge_weight).to(device)
        data1.separate_positive_negative()
    elif args.method == 'SigMaNet':
        edge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(edge_index=edge_index, gcn=False,
                                                                                net_flow=True, x_real=X_real,
                                                                                edge_weight=edge_weight, \
                                                                                normalization='sym',
                                                                                return_lambda_max=False)
        model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=num_input_feat, hidden=4,
                                                       label_dim=args.num_classes,
                                                       i_complex=False, layer=2, follow_math=False, gcn=False,
                                                       net_flow=True, unwind=True, edge_index=edge_index, \
                                                       norm_real=norm_real, norm_imag=norm_imag).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    query_test_edges = link_data[split]['test']['edges']
    y_test = link_data[split]['test']['label']
    if args.direction_only_task:
        y_test = torch.div(y_test, 2, rounding_mode='floor').to(device)


    if args.method == 'MSGNN':
        data_path = 'Datasets/E/E-ExpressionData.csv'
        data = pd.read_csv(data_path, header=0, index_col=0).T
        data = data.transform(lambda x: np.log(x + 1))
        print("data的维度为:", data.shape)


        num_genes = data.shape[1]
        print("num_genes的个数为", num_genes)
        num_cells = data.shape[0]
        print("num_cells的个数为", num_cells)


        input_data = np.zeros((num_genes, 1, num_cells), dtype=np.float32)
        for i in range(num_genes):
            input_data[i, 0, :] = data.iloc[:, i].values

        input_data = torch.tensor(input_data, dtype=torch.float32)
        model_Fcorr = GeneEmbeddingModel(input_shape=(num_cells, 1))
        model_Fcorr.eval()
        with torch.no_grad():
             gene_embeddings = model_Fcorr(input_data)


        all_embeddings = []
        for i in range(num_genes):
            for j in range(num_genes):
                all_embeddings.append(gene_embeddings[i] * gene_embeddings[j])
        all_embeddings = torch.stack(all_embeddings, dim=0).detach().numpy()

        for epoch in range(args.epochs):
            train_loss, train_acc = train_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges,all_embeddings)
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}')
            writer.add_scalar('train_loss_' + str(split), train_loss, epoch)
        accuracy, f1_macro, f1_micro, f1, auc,  precision, mcc, recall = test_MSGNN(X_real, X_img,y_test,edge_index,edge_weight,query_test_edges, all_embeddings,num_classes=args.num_classes)

        print(f'Split: {split:02d}, Test_Acc: {accuracy:.4f}, F1 macro: {f1_macro:.4f}, F1 micro: {f1_micro:.4f}, \
                   F1: {f1:.4f}, AUC: {auc:.4f},  Precision: {precision:.4f}, \
                   MCC: {mcc:.4f}, Recall: {recall:.4f}.')


    elif args.method == 'SSSNET':
        for epoch in range(args.epochs):
            train_loss, train_acc = train_SSSNET(X_real, data1.edge_index_p, data1.edge_weight_p,
                                                 data1.edge_index_n, data1.edge_weight_n, query_edges, y)
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}')
            writer.add_scalar('train_loss_' + str(split), train_loss, epoch)

        accuracy, f1_macro, f1_micro = test_SSSNET(X_real, data1.edge_index_p, data1.edge_weight_p,
                                                   data1.edge_index_n, data1.edge_weight_n, query_test_edges, y_test)
        print(f'Split: {split:02d}, Test_Acc: {accuracy:.4f}, F1 macro: {f1_macro:.4f}, \
            F1 micro: {f1_micro:.4f}.')
    elif args.method == 'SigMaNet':
        for epoch in range(args.epochs):
            train_loss, train_acc = train_SigMaNet(X_real, X_img, y, query_edges)
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}')
            writer.add_scalar('train_loss_' + str(split), train_loss, epoch)

        accuracy, f1_macro, f1_micro = test_SigMaNet(X_real, X_img, y_test, query_test_edges)
        print(f'Split: {split:02d}, Test_Acc: {accuracy:.4f}, F1 macro: {f1_macro:.4f}, \
            F1 micro: {f1_micro:.4f}.')
    else:
        for epoch in range(args.epochs):
            loss = train()
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, Loss: {loss:.4f}.')
            writer.add_scalar('train_loss_' + str(split), loss, epoch)
        accuracy, f1_macro, f1_micro = test(query_edges.cpu(), query_test_edges.cpu(), y.cpu(), y_test.cpu())
        print(f'Split: {split:02d}, '
              f'Accuracy: {accuracy:.4f}, MacroF1: {f1_macro:.4f}, MicroF1: {f1_micro:.4f}')

    res_array[split] = [accuracy, f1_macro, f1_micro, f1, auc,  precision, mcc, recall]

end = time.time()
memory_usage = torch.cuda.max_memory_allocated(device) * 1e-6

print("{}'s average Accuracy, MacroF1 ,MicroF1 ,F1, AUC,  Precision, MCC, Recall: {}".format(args.method, res_array.mean(0)))

print("{}'s total training and testing time: {}s, memory usage: {}M.".format(args.method, end - start, memory_usage))

if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../link_results/', task, 'debug', args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../link_results/', task, args.dataset)

if os.path.isdir(os.path.join(dir_name, sub_dir_name, args.method)) == False:
    try:
        os.makedirs(os.path.join(dir_name, sub_dir_name, args.method))
    except FileExistsError:
        print('Folder exists for {}!'.format(sub_dir_name, args.method))

np.save(os.path.join(dir_name, sub_dir_name, args.method, suffix), res_array)
np.save(os.path.join(dir_name, sub_dir_name, args.method, 'runtime_memory_' + suffix),
        np.array([end - start, memory_usage]))

