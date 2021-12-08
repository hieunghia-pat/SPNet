import torch
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from data_utils.utils import collate_fn

class SPDataset(Dataset):
    def __init__(self, csv_path, nscores=1):
        super(SPDataset, self).__init__()
        
        self.load_csv(csv_path)

        self.xset = self.data[self.family_features + self.school_features + self.personal_features].values
        if nscores == 1:
            self.scoreset = self.data[self.school_reports[-1]].values
        else:
            self.scoreset = self.data[self.school_reports]
        
    def load_csv(self, path):
        data = pd.read_csv(path, sep=";")
        for feature in data.columns:
            if data[feature].dtype == "object":
                data[feature] = LabelEncoder().fit_transform(data[feature])

        self.family_features = ["address", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "guardian", "famsize", "famrel"]
        self.school_features = ["school", "reason", "traveltime", "studytime", "failures", "schoolsup", "famsup", "activities", "paid", "internet", "nursery", "higher", "absences"]
        self.personal_features = ["sex", "age", "romantic", "freetime", "goout", "Walc", "Dalc", "health"]
        self.school_reports = ["G1", "G2", "G3"]

        self.data = data

    def get_folds(self, k=10):
        fold_size = len(self) // k
        fold_sizes = [fold_size] * (k-1) + [len(self) - (k-1)*fold_size]
        subdatasets = random_split(self, fold_sizes, generator=torch.Generator().manual_seed(13))

        folds = []
        for subdataset in subdatasets:
            folds.append(DataLoader(
                subdataset,
                batch_size=len(subdataset),
                shuffle=True,
                collate_fn=collate_fn
            ))

        return folds

    def __len__(self):
        return self.xset.shape[0]

    def __getitem__(self, idx):
        if idx > len(self):
            raise Exception("Index is not valid")

        return self.xset[idx], self.scoreset[idx]