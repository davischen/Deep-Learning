import torch
import torch.nn.functional as F
import torchvision

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

        L = []
        n_input_cahnnels = 3
        for l in layers:
            L.append(self.Block(n_input_cahnnels, l, kernel_size, 2))
            n_input_cahnnels = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(n_input_cahnnels, n_output_channels)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        #Using the mean and std of Imagenet is a common practice
        mean = torch.Tensor([0.485, 0.456, 0.406])#([0.3521554, 0.30068502, 0.28527516])
        std = torch.Tensor([0.229, 0.224, 0.225])#([0.18182722, 0.18656468, 0.15938024])
        #image = (image - mean) / std
        z = (x - mean[None, :, None, None].to(x.device)) / std[None, :, None, None].to(x.device)
        z = self.network(z)
        return self.classifier(z.mean(dim=[2, 3]))

        #raise NotImplementedError('CNNClassifier.forward')


class FCN(torch.nn.Module):
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
            
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
            # print(self.net(x).shape, identity.shape)
            return self.net(x) + identity
            
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2, use_skip=True):
            super().__init__()
            self.use_skip=use_skip
            self.net = torch.nn.Sequential(torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1),
              torch.nn.ReLU())
        def forward(self, x, up_input):
            z=self.net(x)
            # Fix the padding
            z = z[:, :, :up_input.size(2), :up_input.size(3)]
            # Add the skip connection
            if self.use_skip:
                #x = torch.cat([x2, x1], dim=1)
                z = torch.cat([z, up_input], dim=1)
            return z

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
        
        #print('----------start')
        n_input_cahnnels = 3
        self.use_skip = use_skip
        self.n_conv_layer = len(layers)
        skip_layer_size = [3] + layers[:-1]#layers[:-1] is [16, 32, 64], so skip_layer_size is [3, 16, 32, 64]
        self.conv_block=torch.nn.Sequential()
        for i, l in enumerate(layers):
            self.conv_block.add_module('blcok_conv%d'%i, self.Block(n_input_cahnnels, l, kernel_size, 2))
            n_input_cahnnels = l
        for i in self.conv_block._modules:
            print(i)
        #print('----------')
        self.dense_block = torch.nn.Sequential()
        for i, l in list(enumerate(layers))[::-1]:
            self.dense_block.add_module('dense_conv%d'%i, self.UpBlock(n_input_cahnnels, l, kernel_size, 2,self.use_skip))
            n_input_cahnnels = l
            if self.use_skip:
                n_input_cahnnels += skip_layer_size[i]

        for i in self.dense_block._modules:
            print(i)

        print('----------end')
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
        #Using the mean and std of Imagenet is a common practice
        #mean = torch.Tensor([0.485, 0.456, 0.406])#([0.3521554, 0.30068502, 0.28527516])
        #std = torch.Tensor([0.229, 0.224, 0.225]) #([0.18182722, 0.18656468, 0.15938024])
        #image = (image - mean) / std
        #z = (x - mean[None, :, None, None].to(x.device)) / std[None, :, None, None].to(x.device)
        normalize=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])
        z = normalize(x)
        conv_input = []
        for i in range(self.n_conv_layer):
            # Add all the information required for skip connections
            conv_input.append(z)
            z = self.conv_block._modules['blcok_conv%d'%i](z)

        z=self.conv_block(x)
        for i in reversed(range(self.n_conv_layer)):
            z = self.dense_block._modules['dense_conv%d'%i](z,conv_input[i])
            
            #print(str(z.shape))
            #print(up_layer[i].size(2))
            #print(up_layer[i].size(3))
        
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
