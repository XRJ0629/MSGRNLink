from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

from complex_relu import complex_relu_layer
from MSConv import MSConv


class GeneEmbeddingModel(nn.Module):
    def __init__(self, input_shape):
        super(GeneEmbeddingModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2, padding=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(2, padding=1)

        self.shortcut_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, padding=0)
        self.shortcut_batch_norm = nn.BatchNorm1d(16)
        self.shortcut_pool1 = nn.MaxPool1d(2, padding=1)
        self.shortcut_pool2 = nn.MaxPool1d(2, padding=1)
        self.shortcut_conv_final = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, padding=0)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 64)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.batch_norm1(x1)
        x1 = self.pool1(x1)
        x1 = self.relu(x1)

        x1 = self.conv2(x1)
        x1 = self.batch_norm2(x1)
        x1 = self.pool2(x1)
        x1 = self.relu(x1)

        # shortcut分支
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_batch_norm(shortcut)
        shortcut = self.shortcut_pool1(shortcut)
        shortcut = self.shortcut_pool2(shortcut)
        shortcut = self.shortcut_conv_final(shortcut)

        x = x1 + shortcut  # Skip connection
        x = self.relu(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # 去除多余的维度
        feat = self.fc(x)
        return feat



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

model = GeneEmbeddingModel(input_shape=(num_cells, 1))

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

model.eval()
with torch.no_grad():
    gene_embeddings = model(input_data)
    all_embeddings = []
    for i in range(num_genes):
        for j in range(num_genes):
            diff = gene_embeddings[i] - gene_embeddings[j]
            diff_inverse = torch.reciprocal(diff + 1e-8)
            result = torch.tanh(diff_inverse)
            all_embeddings.append(result)
    all_embeddings = torch.stack(all_embeddings, dim=0).numpy()
print("基因对相似性矩阵维度为:", all_embeddings.shape)



class MSGNN_link_prediction(nn.Module):
    def __init__(self, num_features: int, hidden: int = 2, q: float = 0.25, K: int = 1, label_dim: int = 2,
                 activation: bool = True, trainable_q: bool = False, layer: int = 2, dropout: float = 0.5,
                 normalization: str = 'sym', cached: bool = False, absolute_degree: bool = True):
        super(MSGNN_link_prediction, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K,
                            q=q, trainable_q=trainable_q, normalization=normalization))

        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,
                                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached,
                                absolute_degree=absolute_degree))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden*4+64, label_dim)  # 16✖4+Fcorr_out_dim=total_dim
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor,
                query_edges: torch.LongTensor, all_embeddings: torch.FloatTensor,
                edge_weight: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)
        node_i, node_j = query_edges[:, 0], query_edges[:, 1]
        similarity_features = []
        for i, j in zip(node_i, node_j):
            similarity_feature = all_embeddings[i * num_genes + j]
            similarity_features.append(torch.tensor(similarity_feature, dtype=torch.float32))
        similarity_features = torch.stack(similarity_features, dim=0).to(real.device)
        real_node_features = torch.cat((real[node_i], real[node_j]), dim=-1)
        imag_node_features = torch.cat((imag[node_i], imag[node_j]), dim=-1)
        x = torch.cat((real_node_features, imag_node_features, similarity_features), dim=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x



class MSGNN_node_classification(nn.Module):
    r"""The MSGNN model for node classification.
    
    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 1.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`False`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MSConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`False`)
        normalization (str, optional): The normalization scheme for the signed directed
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} \odot \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} \odot \exp(i \Theta^{(q)})`
            `\odot` denotes the element-wise multiplication.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    """
    def __init__(self, num_features:int, hidden:int=2, q:float=0.25, K:int=1, label_dim:int=2, \
        activation:bool=False, trainable_q:bool=False, layer:int=2, dropout:float=False, normalization:str='sym', cached: bool=False, absolute_degree: bool=True):
        super(MSGNN_node_classification, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K, \
            q=q, trainable_q=trainable_q, normalization=normalization))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()
        for _ in range(1, layer):
            chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,\
                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached, absolute_degree=absolute_degree))
        self.Chebs = chebs
        self.Conv = nn.Conv1d(2*hidden, label_dim, kernel_size=1)        
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.Conv.reset_parameters()
        
    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_weight: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        
        Return types:
            * **z** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*hidden) for undirected graphs and (num_nodes, 4*hidden) for directed graphs.
            * **output** (PyTorch FloatTensor) - Log of prob, with shape (num_nodes, num_clusters).
            * **predictions_cluster** (PyTorch LongTensor) - Predicted labels.
            * **prob** (PyTorch FloatTensor) - Probability assignment matrix of different clusters, with shape (num_nodes, num_clusters).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)
        x = torch.cat((real, imag), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        z = torch.transpose(x[0], 0, 1).clone()
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        output = torch.transpose(x[0], 0, 1) # log_prob
        predictions_cluster = torch.argmax(output, dim=1)
        prob = F.softmax(output, dim=1)
        return F.normalize(z), output, predictions_cluster, prob
