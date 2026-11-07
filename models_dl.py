import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(self, n_features, hidden=(256, 128), dropout=0.25):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchRegressor:
    def __init__(self, epochs=250, lr=1e-3, weight_decay=1e-4,
                 batch_size=32, seed=2026):
        self.epochs, self.lr = epochs, lr
        self.weight_decay, self.batch_size, self.seed = weight_decay, batch_size, seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(self.imputer.fit_transform(X))
        self.model = MLP(X.shape[1])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.model.train()
        for epoch in range(self.epochs):
            order = torch.randperm(len(Xt))
            for start in range(0, len(order), self.batch_size):
                ix = order[start:start+self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(Xt[ix]), yt[ix])
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        X = self.scaler.transform(self.imputer.transform(X))
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy()
