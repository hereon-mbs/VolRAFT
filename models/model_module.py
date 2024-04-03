import torch
import torch.nn as nn

# Module for normalization
class ModuleNorm(nn.ModuleList):
    def __init__(self, 
                 num_channels, 
                 num_groups = 0,
                 norm_type = 'none',
                 dtype=torch.float32):
        super(ModuleNorm, self).__init__()

        if norm_type.lower() == 'group':
            self.module = nn.GroupNorm(num_groups = num_groups, 
                                       num_channels = num_channels,
                                       dtype=dtype)
        elif norm_type.lower() == 'batch' or norm_type.lower() == 'batch3d':
            self.module = nn.BatchNorm3d(num_features = num_channels,
                                         dtype=dtype)
        elif norm_type.lower() == 'batch1d':
            self.module = nn.BatchNorm1d(num_features = num_channels,
                                         dtype=dtype)
        elif norm_type.lower() == 'instance' or norm_type.lower() == 'instance3d':
            self.module = nn.InstanceNorm3d(num_features = num_channels,
                                            dtype=dtype)
        elif norm_type.lower() == 'instance1d':
            self.module = nn.InstanceNorm1d(num_features = num_channels,
                                            dtype=dtype)            
        else:
            self.module = nn.Sequential()

    def forward(self, x):
        return self.module(x)
    
# Module for normalization
class ModuleDropout(nn.ModuleList):
    def __init__(self, 
                 p,
                 dropout_type='none',
                 inplace=False):
        super(ModuleDropout, self).__init__()

        if dropout_type.lower() == 'dropout' or dropout_type.lower() == 'dropout3d':
            self.module = nn.Dropout3d(p = p, inplace = inplace)
        elif dropout_type.lower() == 'dropout1d':
            self.module = nn.Dropout1d(p = p, inplace = inplace)
        else:
            self.module = nn.Sequential()

    def forward(self, x):
        return self.module(x)
    
# Module for pooling
class ModulePool(nn.ModuleList):
    def __init__(self,
                 kernel_size,
                 pool_type = 'none',
                 dtype=torch.float32):
        super(ModulePool, self).__init__()

        if pool_type.lower() == 'max':
            self.module = nn.MaxPool3d(kernel_size = kernel_size)
        elif pool_type.lower() == 'avg':
            self.module = nn.AvgPool3d(kernel_size = kernel_size)
        else:
            self.module = nn.Sequential()

    def forward(self, x):
        return self.module(x)
    
# Module for activation
class ModuleActi(nn.ModuleList):
    def __init__(self,
                 acti_type='none',
                 inplace=False):
        super(ModuleActi, self).__init__()

        if acti_type.lower() == 'relu':
            self.module = nn.ReLU(inplace = inplace)
        elif acti_type.lower() == 'leaky':
            self.module = nn.LeakyReLU(inplace = inplace)
        elif acti_type.lower() == 'prelu':
            self.module = nn.PReLU()
        elif acti_type.lower() == 'sigmoid':
            self.module = nn.Sigmoid()
        elif acti_type.lower() == 'tanh':
            self.module = nn.Tanh()
        elif acti_type.lower() == 'swish':
            self.module = nn.SiLU(inplace = inplace)
        else:
            self.module = nn.Sequential()

    def forward(self, x):
        return self.module(x)