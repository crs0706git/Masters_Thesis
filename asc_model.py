import torch
from torch import nn
import math

import pdb


def conv_layer(in_ch, out_ch, id):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding='same', kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Dropout(id),
        nn.BatchNorm2d(out_ch),
    )


class asc(nn.Module):
    def __init__(self, in_dim, in_y, in_x, num_out, in_dp, in_lstm_layers=3, in_bidirectional=True, init_out=8):
        super(asc, self).__init__()
                
        self.conv_in = conv_layer(in_dim, init_out, in_dp)
        self.conv_1 = conv_layer(init_out, init_out, in_dp)
        self.conv_2 = conv_layer(init_out, in_dim, in_dp)
        
        self.cls_hidden = in_y
        if in_bidirectional:
            lstm_hidden_size = self.cls_hidden // 2
        else:
            lstm_hidden_size = self.cls_hidden
        self.lstm_layer = nn.LSTM(
            input_size=self.cls_hidden,
            hidden_size=lstm_hidden_size,
            num_layers=in_lstm_layers,
            bidirectional=in_bidirectional,
            batch_first=False
        )
        
        self.fc_num = in_y
        if in_y % 2 != 0:
            self.fc_num -= 1
        fc_in = self.fc_num * in_x
        fc1_2 = 5
        fc2_3 = 10
        self.fcs = nn.Sequential(
            nn.Linear(fc_in, fc_in//fc1_2),
            nn.ReLU(inplace=True),
            
            nn.Linear(fc_in//fc1_2, fc_in//fc2_3),
            nn.ReLU(inplace=True),
                        
            nn.Linear(fc_in//fc2_3, num_out),
        )
    
    def forward(self, x):
        xb, xch, xf, xt = x.shape
        
        x = self.conv_in(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        # x = self.conv_3(x)
        
        x = x.squeeze(1)
        x = x.reshape(xb, xt, self.cls_hidden)
        
        x = self.lstm_layer(x)
        
        x = x[0].reshape(xb, x[0].shape[1]*x[0].shape[2])
        
        x = self.fcs(x)
        
        return x
