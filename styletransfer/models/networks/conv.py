import torch.nn as nn


def option_setter(module, args):
    
    # Default instantiation
    module.n_filter = 16
    module.conv_size = 3
    module.conv_pad = 1
    module.pool_size = (3, 1)
    module.pool_pad = (1, 0)
    module.pool_stride = (2, 1)
    module.norm_layer = nn.BatchNorm2d
    module.n_downsample = 4
    module.use_bias = False
    module.shrinking_filter = False
    module.reshaped = False
     
    # Argument setting
    for name, value in args.items(): 
        setattr(module, name, value)

    if module.shrinking_filter:
        module.padw = 2 ** (module.n_downsample - 1)
        moduel.kw = 2 * module.padw + 1 
 

class ConvEncoder(nn.Module):
    def __init__(self, **kwargs):
        option_setter(self, kwargs) 
        model = [nn.Conv2d(2, self.ngf, kernel_size=self.conv_size, padding=self.conv_pad, bias=self.use_bias)]

        for i in range(self.n_downsample):
            next_mult = mult * 2
            model += [
                nn.Conv2d(mult, mult, kernel_size=kw, padding=padw, bias=use_bias),
                nn.Maxpool2d((poolw, 1), stride=(2, 1), padding=(1, 0)), # 3, 3, 1 or 3, 2, 2 works
                norm_layer(next_mult),
                nn.ReLU(True)
            ]
            # kw = padw + 1
            # padw = padw // 2
            mult = next_mult

        self.model = nn.Sequential(*model)

        def forward(self, input):
            return self.model(input)


class ConvDecoder(nn.Module):
    def __init__(self, ngf=16, norm_layer=nn.BatchNorm2d, n_downsample=4, use_bias=False):
        self.n_downsample = n_downsample 

        model = list()

        mult = ngf * (2 ** self.n_downsample)
        for i in range(self.n_downsample):  # add upsampling layers 
            # padw = 2 ** i
            # kw = 2 * padw + 1
            next_mult = mult // 2
            model += [
                nn.ConvTranspose2d(mult, next_mult, kernel_size=(kw, 1), padding=(padw, 0), stride=(2,1), bias=use_bias), 
                norm_layer(next_mult),
                nn.ReLU(True)
            ]
            mult = next_mult

        model += [
            ('conv_final_%s' % i, nn.ConvTranspose2d(ngf, 2, kernel_size=kw, padding=padw)),
            ('tanh_%s' % i, nn.Tanh())
        ]

    def forward(self, input):
        return self.model(input)
