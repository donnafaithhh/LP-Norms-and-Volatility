# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import re

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader

# training
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import copy

# ignore warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

class AdjacencyMatrixDataset(Dataset):
    """
    Dataset for loading adjacency matrices from pickle files.
    """
    
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        
        self.files = sorted(
            f for f in os.listdir(directory) 
            if f.endswith(".pkl") and f.startswith("2000") and f != '.ipynb_checkpoints'
        )
        
        self.dates = []
        self.matrices = []
        
        for f in self.files:
            filepath = os.path.join(directory, f)
            with open(filepath, "rb") as file:
                data = pickle.load(file)
            
            self.dates.append(data["date"])
            self.matrices.append(data["adjacency matrix"])
        
        self.matrices = np.array(self.matrices)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        matrix_tensor = torch.tensor(matrix, dtype=torch.float32)
        
        if len(matrix_tensor.shape) == 2:
            matrix_tensor = matrix_tensor.unsqueeze(0)
        
        date = self.dates[idx]
        
        if self.transform:
            matrix_tensor = self.transform(matrix_tensor)
            
        return matrix_tensor, date
    
    def get_all_matrices(self):
        return torch.tensor(self.matrices, dtype=torch.float32)
    
    def get_all_dates(self):
        return self.dates
    
    def get_shape(self):
        return self.matrices.shape

# access dataset
dataset = AdjacencyMatrixDataset("adjacency matrices 2")

# train test split
test_percent = 0.2
test_pts = int(np.floor(len(dataset) * test_percent))
train_pts = len(dataset) - test_pts

# get train test dataset
train_dataset = torch.utils.data.Subset(dataset, range(train_pts))
test_dataset = torch.utils.data.Subset(dataset, range(train_pts, len(dataset)))

# put inside dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, adj):
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        support = self.linear(x)
        output = torch.bmm(adj, support)
        return output

class TGCNCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(TGCNCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.gcn_reset = GCNLayer(in_channels + hidden_channels, hidden_channels)
        self.gcn_update = GCNLayer(in_channels + hidden_channels, hidden_channels)
        self.gcn_candidate = GCNLayer(in_channels + hidden_channels, hidden_channels)
        
    def forward(self, x, adj, h):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if h.dim() == 2:
            h = h.unsqueeze(0)
            
        combined = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.gcn_reset(combined, adj))
        z = torch.sigmoid(self.gcn_update(combined, adj))
        combined_reset = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.gcn_candidate(combined_reset, adj))
        h_new = (1 - z) * h + z * h_tilde
        return h_new.squeeze(0)

class TGCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.3):
        super(TGCNLayer, self).__init__()
        self.tgcn_cell = TGCNCell(in_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.norm = nn.LayerNorm(hidden_channels)
        
    def forward(self, x_seq, adj):
        batch_size, time_steps, num_nodes, _ = x_seq.shape
        h = torch.zeros(batch_size, num_nodes, self.tgcn_cell.hidden_channels, device=x_seq.device)
        outputs = []
        for t in range(time_steps):
            h = self.tgcn_cell(x_seq[:, t, :, :], adj, h)
            h = self.activation(h)
            h = self.norm(h)
            h = self.dropout(h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)

class TGCNModel(nn.Module):
    def __init__(self, num_nodes=498, in_channels=1, hidden_channels=128, 
                 num_layers=8, dropout=0.3, time_steps=10):
        super(TGCNModel, self).__init__()
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.tgcn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_channels = hidden_channels if i > 0 else hidden_channels
            self.tgcn_layers.append(
                TGCNLayer(layer_input_channels, hidden_channels, dropout)
            )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x_seq, adj):
        batch_size = x_seq.size(0)
        x_seq = self.input_proj(x_seq)
        h = x_seq
        for i, layer in enumerate(self.tgcn_layers):
            h_new = layer(h, adj)
            if i > 0 and i % 2 == 1:
                if h.shape == h_new.shape:
                    h = h + h_new
                else:
                    h = h_new
            else:
                h = h_new
        final_hidden = h[:, -1, :, :]
        return final_hidden
    
    def predict_links(self, node_embeddings, edge_pairs):
        batch_size = node_embeddings.size(0)
        src_nodes = edge_pairs[:, 0]
        dst_nodes = edge_pairs[:, 1]
        src_emb = node_embeddings[:, src_nodes, :]
        dst_emb = node_embeddings[:, dst_nodes, :]
        edge_features = torch.cat([src_emb, dst_emb], dim=-1)
        logits = self.decoder(edge_features).squeeze(-1)
        return logits

# setting parameters
verbose = True
epochs = 3000
patience = 50
dropout = 0.3
best_loss = np.inf
best_accuracy = -np.inf
best_model_wts = copy.deepcopy(model.state_dict())
counter = 0


# instantiate model
model = CNN(dropout=0.3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# temp var per model with x dropout
best_model_temp = copy.deepcopy(model.state_dict())
best_loss_temp = np.inf
best_accuracy_temp = -np.inf

# train every epoch
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

    # get performance
    epoch_loss = running_loss / len(train_loader.dataset)    
    epoch_accuracy = evaluate_model(model, test_loader)
    
    # early stopping
    if epoch_loss < best_loss_temp:
        best_loss_temp = epoch_loss
        best_accuracy_temp = epoch_accuracy
        best_model_temp = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        if verbose:
            print()
            print(f"Early stopping at epoch {epoch + 1}, dropout {dropout}")
            print(f"Loss of best model: {best_loss_temp}")
            print(f"Accuracy of best model: {best_accuracy_temp}")
            print()
        break

    if verbose:
        if epoch % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Dropout: {dropout}")

    if best_accuracy_temp > best_accuracy:
    # if best_loss_temp < best_loss:
        best_loss = best_loss_temp
        best_accuracy = best_accuracy_temp
        best_model_wts = best_model_temp
        

# save best model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "best tgcn weights.pth")