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
from torch.utils.data import Dataset, DataLoader, Subset

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
    
    def __init__(self, directory, time_steps=10, transform=None):
        self.directory = directory
        self.transform = transform
        self.time_steps = time_steps
        
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
        # Return number of sequences we can create
        return max(0, len(self.files) - self.time_steps)
    
    def __getitem__(self, idx):
        # Get sequence of time_steps matrices
        sequence = self.matrices[idx:idx + self.time_steps]
        # Target is the next matrix after the sequence
        target = self.matrices[idx + self.time_steps]
        
        # Convert to tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)  # (time_steps, N, N)
        target_tensor = torch.tensor(target, dtype=torch.float32)  # (N, N)
        
        # Add channel dimension if needed
        if len(sequence_tensor.shape) == 3:
            sequence_tensor = sequence_tensor.unsqueeze(2)  # (time_steps, N, 1, N)
        
        dates_sequence = self.dates[idx:idx + self.time_steps]
        target_date = self.dates[idx + self.time_steps]
        
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
            
        return sequence_tensor, target_tensor, dates_sequence, target_date
    
    def get_all_matrices(self):
        return torch.tensor(self.matrices, dtype=torch.float32)
    
    def get_all_dates(self):
        return self.dates
    
    def get_shape(self):
        return self.matrices.shape

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, adj):
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        
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
        return h_new

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
    def __init__(self, num_nodes=496, hidden_channels=128, 
                 num_layers=8, dropout=0.3, time_steps=10):
        super(TGCNModel, self).__init__()
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.hidden_channels = hidden_channels
        self.input_proj = nn.Linear(num_nodes, hidden_channels)
        self.tgcn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_channels = hidden_channels if i > 0 else hidden_channels
            self.tgcn_layers.append(
                TGCNLayer(layer_input_channels, hidden_channels, dropout)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_nodes)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x_seq, adj):
        batch_size, time_steps, num_nodes, _, _ = x_seq.shape
        x_seq = x_seq.squeeze(3)
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
        
        output = self.decoder(final_hidden)
        output = torch.bmm(output, output.transpose(1, 2))
        
        return output

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, targets, _, _ in dataloader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            adj = sequences[0, 0, :, 0, :]
            
            outputs = model(sequences, adj)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * sequences.size(0)
            
            probs = torch.sigmoid(outputs)
            
            all_preds.append(outputs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    all_probs = np.concatenate(all_probs, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    pred_binary = (all_probs > 0.5).astype(int)
    
    try:
        auc_roc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_roc = 0.5
    
    try:
        auc_pr = average_precision_score(all_targets, all_probs)
    except ValueError:
        auc_pr = 0.0
    
    try:
        f1 = f1_score(all_targets, pred_binary)
    except ValueError:
        f1 = 0.0
    
    return {
        'loss': epoch_loss,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': f1
    }

# access dataset
time_steps = 10
dataset = AdjacencyMatrixDataset("adjacency matrices 2", time_steps=time_steps)
all_matrices = dataset.get_all_matrices().numpy()
nan_count = np.isnan(all_matrices).sum()
if nan_count > 0:
    dataset.matrices = np.nan_to_num(dataset.matrices, nan=0.0)

# setting parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# results
cv_results = []
best_model_state = None
best_cv_score = -np.inf

# train model
for fold, (train_index, val_index) in enumerate(tscv.split(range(len(dataset)))):
    print(f"\nFold {fold + 1}/{n_splits}\n")  
    
    # train and validation subsets
    train_subset = Subset(dataset, train_index)
    val_subset = Subset(dataset, val_index)    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    
    # initialize model
    model = TGCNModel(num_nodes=496, hidden_channels=128, num_layers=8, 
                      dropout=0.3, time_steps=time_steps).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # parameters
    epochs = 1000
    patience = np.floor(epochs * 0.05)
    best_loss = np.inf
    best_accuracy = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0
    
    # loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for sequences, targets, _, _ in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            adj = sequences[0, 0, :, 0, :]
            
            optimizer.zero_grad()
            outputs = model(sequences, adj)
            loss = criterion(outputs, targets)

            # gradient clipping even if they didnt mention this
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * sequences.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # validation
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['f1_score']
        
        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if epoch % 50 == 0:
            print(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # store fold results
    cv_results.append({
        'fold': fold + 1,
        'best_loss': best_loss,
        'best_accuracy': best_accuracy
    })
    
    if best_accuracy > best_cv_score:
        best_cv_score = best_accuracy
        best_model_state = best_model_wts
    
    print(f"Fold {fold + 1} - Best Val Loss: {best_loss:.4f}, Best Val Acc: {best_accuracy:.4f}")

# CV results
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)
for result in cv_results:
    print(f"Fold {result['fold']}: Loss = {result['best_loss']:.4f}, Accuracy = {result['best_accuracy']:.4f}")

mean_accuracy = np.mean([r['best_accuracy'] for r in cv_results])
std_accuracy = np.std([r['best_accuracy'] for r in cv_results])
print(f"\nMean CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

# save best model
if best_model_state is not None:
    torch.save(best_model_state, "best_tgcn_weights.pth")
    print("\nBest model saved as 'best_tgcn_weights.pth'")