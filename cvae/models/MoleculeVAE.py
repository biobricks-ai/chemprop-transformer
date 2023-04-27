import torch, torch.nn as nn

class MoleculeVAE(nn.Module):

    def __init__(self):
        super(MoleculeVAE, self).__init__()
        
        def conv1(in_channels, out_channels, kernel_size, stride):
            c1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            return nn.Sequential(c1,nn.ReLU())
        
        self.conv_1 = conv1(in_channels=58, out_channels=9, kernel_size=9, stride=1)
        self.conv_2 = conv1(in_channels=9, out_channels=9, kernel_size=9, stride=1)
        self.conv_3 = conv1(in_channels=9, out_channels=10, kernel_size=11, stride=1)
        self.dense_1 = nn.Sequential(nn.Linear(940, 435),nn.ReLU())
        self.z_mean = nn.Linear(435, 292)
        self.z_log_var = nn.Linear(435, 292)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.dense_1(x.permute(0,2,1).reshape(-1,940))
        return self.z_mean(x), self.z_log_var(x)
    
    @staticmethod
    def load(ptfile):
        model = MoleculeVAE()
        model.load_state_dict(torch.load(ptfile))
        return model