import torch
from torch import nn
import math

import pdb


def conv_layer(in_ch, out_ch, in_kernel=5):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=in_kernel),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
        nn.MaxPool2d(2),
        nn.Dropout(0.2),
    )


def shape_calc(in_w, in_h, in_pad=0, in_kernel=5, in_dil=1, in_stride=1):
    return math.floor((math.floor(in_w + 2*in_pad - in_dil*(in_kernel-1))/in_stride)/2), math.floor(math.floor((in_h + 2*in_pad - in_dil*(in_kernel-1))/in_stride)/2)


class isocc(nn.Module):
    def __init__(self, in_dim, in_y, in_x, num_out, init_out=16):
        super(isocc, self).__init__()
        
        self.cur_out = init_out
        self.conv_in = conv_layer(in_dim, self.cur_out)
        self.up_w, self.up_h = shape_calc(in_x, in_y)
        
        self.conv_1 = conv_layer(self.cur_out, self.cur_out * 2)
        self.cur_out *= 2
        self.up_w, self.up_h = shape_calc(self.up_w, self.up_h)
                
        self.conv_2 = conv_layer(self.cur_out, self.cur_out * 2)
        self.cur_out *= 2
        self.up_w, self.up_h = shape_calc(self.up_w, self.up_h)
                
        self.conv_3 = conv_layer(self.cur_out, self.cur_out)
        self.up_w, self.up_h = shape_calc(self.up_w, self.up_h)
        
        conv_fc = 1024
        fc_fc = 512
        self.fcs = nn.Sequential(
            nn.Linear(
                self.cur_out * self.up_w * self.up_h, conv_fc
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(conv_fc, fc_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(fc_fc, num_out),
        )
        	
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        
        x = x.view(x.size(0), -1)
                
        x = self.fcs(x)
        #pdb.set_trace()
        #x = nn.Sigmoid(x)
        x = self.sig(x)
        
        return x
