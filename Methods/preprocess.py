import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import Dataset, DataLoader

def get_common_genes(adata_rna, adata_spa):
    common_genes = np.intersect1d(adata_rna.var_names, adata_spa.var_names)
    return adata_rna[:, common_genes].copy(), adata_spa[:, common_genes].copy()

def process_labels(adata_rna, adata_spa, label_col):
    unique_labels = adata_rna.obs[label_col].unique().tolist()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    adata_rna.obs["label_idx"] = adata_rna.obs[label_col].map(label_map)
    
    if label_col in adata_spa.obs.columns:
        adata_spa.obs["label_idx"] = adata_spa.obs[label_col].map(label_map).fillna(-1).astype(int)
    else:
        adata_spa.obs["label_idx"] = -1
        
    return adata_rna, adata_spa, label_map

def normalize_and_hvg(adata, n_top_genes):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    return adata

def batch_scale_data(adata, chunk_size=10000):
    scaler = MaxAbsScaler(copy=False)
    scaler.fit(adata.X)
    
    n_samples = adata.shape[0]
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        adata.X[i:end] = scaler.transform(adata.X[i:end])
    return adata

class STDataset(Dataset):
    def __init__(self, adata):
        self.data = adata.X
        self.labels = adata.obs["label_idx"].values

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = torch.tensor(x.squeeze(), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, idx

def prepare_dataloaders(adata_rna, adata_spa, batch_size):
    ds_rna = STDataset(adata_rna)
    ds_spa = STDataset(adata_spa)
    
    dl_rna = DataLoader(ds_rna, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_spa = DataLoader(ds_spa, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dl_rna, dl_spa

def preprocess_pipeline(adata_rna, adata_spa, label_col="cell_type", n_top_genes=2000, batch_size=256):
    adata_rna, adata_spa = get_common_genes(adata_rna, adata_spa)
    
    adata_rna, adata_spa, label_mapping = process_labels(adata_rna, adata_spa, label_col)
    
    adata_rna = normalize_and_hvg(adata_rna, n_top_genes)
    adata_spa = normalize_and_hvg(adata_spa, n_top_genes)
    
    adata_rna, adata_spa = get_common_genes(adata_rna, adata_spa)
    
    adata_rna = batch_scale_data(adata_rna)
    adata_spa = batch_scale_data(adata_spa)
    
    dl_rna, dl_spa = prepare_dataloaders(adata_rna, adata_spa, batch_size)
    
    return dl_rna, dl_spa, label_mapping