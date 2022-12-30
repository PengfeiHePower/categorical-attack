from torch import nn
import torch
import torch.nn.functional as F


class geneRNN(nn.Module):
    def __init__(self):
        super(geneRNN, self).__init__()
        dropout_rate = 0.5
        input_size = 30
        self.hidden_size = 30
        n_labels = 3
        self.num_layers = 4
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(self.hidden_size, n_labels)
        self.attention = SelfAttention(self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.n_diagnosis_codes = 5
        self.embed = torch.nn.Embedding(self.n_diagnosis_codes, 30)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    # overload forward() method
    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.embed(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).relu().mean(dim=2)

        h0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        c0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        logit = self.softmax(logit)

        return logit


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class IPSRNN(nn.Module):
    def __init__(self):
        super(IPSRNN, self).__init__()
        dropout_rate = 0.5
        input_size = 70
        hidden_size = 70
        n_labels = 3
        self.n_diagnosis_codes = 1104
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, n_labels)
        self.attention = SelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.embed = torch.nn.Embedding(self.n_diagnosis_codes, input_size)
        # self.embed.weight = nn.Parameter(torch.FloatTensor(emb_weights))
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.embed(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).relu().mean(dim=2)

        h0 = torch.randn((self.num_layers, x.size()[1], x.size()[2])).cuda()
        c0 = torch.randn((self.num_layers, x.size()[1], x.size()[2])).cuda()
        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        logit = self.softmax(logit)
        return logit


def model_file(Dataset, Model_Type):
    return Model[Dataset][Model_Type]


Splice_Model = {
    'Normal': './classifier/Adam_RNN.4832',
    'adversarial': './classifier/Adam_RNN.17490'
}

IPS_Model = {
    'Normal': './classifier/Mal_RNN.942',
    'adversarial': './classifier/Mal_adv.705',
}

Model = {
    'Splice': Splice_Model,
    'IPS': IPS_Model,
}

