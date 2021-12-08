from torch import nn
from torch.nn import functional as F

class SPNet_1(nn.Module):
    def __init__(self, 
                d_model, 
                score_dim=1, 
                dropout=0.1):

        super(SPNet_1, self).__init__()

        self.summarize = nn.Linear(d_model, score_dim)
        self.dropout = nn.Dropout(dropout)

        self.initialize()

    def initialize(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        scores = self.dropout(self.summarize(x))

        return scores.squeeze()

class SPNet_2(nn.Module):
    def __init__(self, 
                d_model, 
                score_dim=1, 
                dropout=0.1):

        super(SPNet_2, self).__init__()
        
        self.fc_1 = nn.Linear(d_model, d_model // 2)
        self.dropout_1 = nn.Dropout(dropout)

        self.summarize = nn.Linear(d_model // 2, score_dim)
        self.dropout = nn.Dropout(dropout)

        self.initialize()

    def initialize(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        x = self.dropout_1(F.relu(self.fc_1(x)))

        scores = self.dropout(self.summarize(x))

        return scores.squeeze()

class SPNet_3(nn.Module):
    def __init__(self, 
                d_model, 
                score_dim=1, 
                dropout=0.1):

        super(SPNet_3, self).__init__()
        
        self.fc_1 = nn.Linear(d_model, d_model // 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(d_model // 2, d_model // 4)
        self.dropout_2 = nn.Dropout(dropout)

        self.summarize = nn.Linear(d_model // 4, score_dim)
        self.dropout = nn.Dropout(dropout)

        self.initialize()

    def initialize(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        x = self.dropout_1(F.relu(self.fc_1(x)))
        x = self.dropout_2(F.relu(self.fc_2(x)))

        scores = self.dropout(self.summarize(x))

        return scores.squeeze()

class SPNet_4(nn.Module):
    def __init__(self, 
                d_model, 
                score_dim=1, 
                dropout=0.1):

        super(SPNet_4, self).__init__()
        
        self.fc_1 = nn.Linear(d_model, d_model // 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(d_model // 2, d_model // 4)
        self.dropout_2 = nn.Dropout(dropout)
        self.fc_3 = nn.Linear(d_model // 4, d_model // 8)
        self.dropout_3 = nn.Dropout(dropout)

        self.summarize = nn.Linear(d_model // 8, score_dim)
        self.dropout = nn.Dropout(dropout)

        self.initialize()

    def initialize(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        x = self.dropout_1(F.relu(self.fc_1(x)))
        x = self.dropout_2(F.relu(self.fc_2(x)))
        x = self.dropout_3(F.relu(self.fc_3(x)))

        scores = self.dropout(self.summarize(x))

        return scores.squeeze()

class SPNet_5(nn.Module):
    def __init__(self, 
                d_model, 
                score_dim=1, 
                dropout=0.1):

        super(SPNet_5, self).__init__()
        
        self.fc_1 = nn.Linear(d_model, d_model // 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(d_model // 2, d_model // 4)
        self.dropout_2 = nn.Dropout(dropout)
        self.fc_3 = nn.Linear(d_model // 4, d_model // 8)
        self.dropout_3 = nn.Dropout(dropout)
        self.fc_4 = nn.Linear(d_model // 8, d_model // 16)
        self.dropout_4 = nn.Dropout(dropout)

        self.summarize = nn.Linear(d_model // 16, score_dim)
        self.dropout = nn.Dropout(dropout)

        self.initialize()

    def initialize(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x):
        x = self.dropout_1(F.relu(self.fc_1(x)))
        x = self.dropout_2(F.relu(self.fc_2(x)))

        scores = self.dropout(self.summarize(x))

        return scores.squeeze()

