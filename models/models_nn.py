import torch
import torch.nn as nn
from torch.utils.data import Dataset


class HousePriceDATASET(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y
    
    def __len__(self):
        return len(self.X_num)
    
    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]
    
class HousePriceMLP(nn.Module):
    def __init__(self, num_features, embedding_size):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_size, emb_dim)
            for cat_size, emb_dim in embedding_size
        ])

        total_emb_dim = sum([emb_dim for _, emb_dim in embedding_size])

        self.model = nn.Sequential(
            nn.Linear(num_features + total_emb_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.05),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.05),

            nn.Linear(32, 1)
        )

    def forward(self, x_num, x_cat):
        emb_features = []
        for i, emb_mass in enumerate(self.embeddings):
            emb = emb_mass(x_cat[:, i])
            emb_features.append(emb)

        x_cat_emb = torch.cat(emb_features, dim=1)

        x = torch.cat([x_num, x_cat_emb], dim=1)

        return self.model(x)