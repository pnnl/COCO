import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def ReLUnet(input_size, output_size, hidden_sizes=None, n_hidden=1, batch_norm=True, initialize=True, dropout=False):
    if hidden_sizes == None:
        hidden_sizes = [input_size]

    n_hidden = len(hidden_sizes)
    layers = []
    layers.append(  torch.nn.Linear(input_size, hidden_sizes[0])  )
    for i in range(1,n_hidden):

        S = output_size if (i==(n_hidden)) else hidden_sizes[i]
        if dropout:
            layers.append( torch.nn.Dropout(p=0.1, inplace=False) )
        layers.append( torch.nn.ReLU() )
        if batch_norm:
            layers.append( torch.nn.BatchNorm1d(hidden_sizes[i-1]) )
        layers.append( torch.nn.Linear(hidden_sizes[i-1],S) )

    predictor = torch.nn.Sequential( *layers )

    if initialize:
        predictor.apply(init_weights)

    return predictor
