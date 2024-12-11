import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,stride=stride, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU()
            )
            self.downsample = None
            if stride != 1 or n_input != n_output:
                # print(stride,n_input, n_output)
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, 1,stride=stride),
                                                      torch.nn.BatchNorm2d(n_output))
            
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward2(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

        
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            # print(self.net(x).shape, identity.shape)
            return self.net(x) + identity

    #def __init__(self):
    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=6, kernel_size=3):
    
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        L = []
        c = 3
        for l in layers:
            L.append(self.Block(c, l, kernel_size, 2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        #z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
        #return self.classifier(z.mean(dim=[2, 3]))
        z = self.network(x)
        z = z.mean([2,3])
        return self.classifier(z)
        #raise NotImplementedError('CNNClassifier.forward')

class FCN2(torch.nn.Module):

    def __init__(self, layers=[64,128], n_input_channels=3, n_output_channels =5):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(layers[0]),
             torch.nn.ReLU(inplace=True)
             ]
        c = layers[0]
        for i, l in enumerate(layers):
            L.append(self.Block(c, l, i, stride=2))
            c = l

        # output channels layer
        L.append(torch.nn.Conv2d(layers[-1], n_output_channels, kernel_size=1, stride=1))
        L.append(torch.nn.BatchNorm2d(n_output_channels))

        # final layer
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=1))
        L.append(torch.nn.BatchNorm2d(n_output_channels))
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=1))
        L.append(torch.nn.BatchNorm2d(n_output_channels))
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=1))

        self.net = torch.nn.Sequential(*L)


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        z = self.net(x)
        H = x.shape[2]
        W = x.shape[3]
        z = z[:, :, :H, :W]
        return z


class FCN(torch.nn.Module):
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=5, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, CNNClassifier.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)


class FCN2(torch.nn.Module):
    class UpBlock2(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1),
              torch.nn.ReLU())
        def forward(self, x):
            return self.net(x)
            
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=5, kernel_size=3, use_skip=True):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])
        
        n_input_cahnnels = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        print(skip_layer_size);

        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, CNNClassifier.Block(n_input_cahnnels, l, kernel_size, 2))
            n_input_cahnnels = l

        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(n_input_cahnnels, l, kernel_size, 2))
            n_input_cahnnels = l
            if self.use_skip:
                n_input_cahnnels += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(n_input_cahnnels, n_output_channels, 1)

        #raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)
        #raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
