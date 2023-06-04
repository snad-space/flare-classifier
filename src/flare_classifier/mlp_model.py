import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets


class FlaresDataset(Dataset):
    def __init__(self, df, feature_names):
        self.dataframe = df[feature_names]
        self.labels = df["is_flare"]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx].to_numpy()
        x = torch.tensor(x, dtype=torch.float32)

        y = self.labels.iloc[idx]
        y = torch.tensor(y, dtype=torch.float)

        return x, y


class BinaryClassification(nn.Module):
    def __init__(self, n_input):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(n_input, 300)
        self.layer_2 = nn.Linear(300, 300)
        self.layer_3 = nn.Linear(300, 400)
        self.layer_out = nn.Linear(400, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.batchnorm1 = nn.BatchNorm1d(300, track_running_stats=False)
        self.batchnorm2 = nn.BatchNorm1d(300, track_running_stats=False)
        self.batchnorm3 = nn.BatchNorm1d(400, track_running_stats=False)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
