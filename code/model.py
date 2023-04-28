import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularVAE(nn.Module):
    def __init__(self, len_charset, labels_size, verbose=False):
        super(MolecularVAE, self).__init__()

        self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
        self.bn_conv_1 = nn.BatchNorm1d(9)
        
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.bn_conv_2 = nn.BatchNorm1d(9)
        
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.bn_conv_3 = nn.BatchNorm1d(10)
        
        self.linear_0 = nn.Linear((len_charset-26)*10 + labels_size, 435)
        self.bn_linear_0 = nn.BatchNorm1d(435)
        
        self.linear_mean = nn.Linear(435, 292)
        self.bn_linear_mean = nn.BatchNorm1d(292)
        
        self.linear_logvar = nn.Linear(435, 292)
        self.bn_linear_logvar = nn.BatchNorm1d(292)

        self.linear_decoderin = nn.Linear(292 + labels_size, 292)
        self.bn_linear_decoderin = nn.BatchNorm1d(292)
        
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.bn_gru = nn.BatchNorm1d(120)
        
        self.linear_decoderout = nn.Linear(501, len_charset)
        self.bn_linear_decoderout = nn.BatchNorm1d(len_charset)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.verbose = verbose

    def encode(self, x, labels):
        x = self.bn_conv_1(self.relu(self.conv_1(x)))
        x = self.bn_conv_2(self.relu(self.conv_2(x)))
        x = self.bn_conv_3(self.relu(self.conv_3(x)))
        x = x.view(x.size(0), -1)
        
        # concatenate x with label tensor
        x = torch.cat((x, labels), dim=1)
        
        x = F.selu(self.linear_0(x))
        self.mean = self.linear_mean(x)
        self.logvar = self.linear_logvar(x)
        
        return self.mean, self.logvar

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z, labels):
        # concatenate x with label tensor
        z = torch.cat((z, labels), dim=1)
        
        z = self.bn_linear_decoderin(F.selu(self.linear_decoderin(z)))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_decoderout(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x, labels):
        z_mean, z_logvar = self.encode(x, labels)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z, labels), z_mean, z_logvar