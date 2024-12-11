import torch
import torch.nn.functional as F
import torchvision

def spatial_argmax_b(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    batch_size, height, width = logit.size()

    # Create coordinate grids
    #torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]
    #torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]
    
    x_coords = torch.linspace(-1, 1, width, device=logit.device)
    y_coords = torch.linspace(-1, 1, height, device=logit.device)
    #print(x_coords.size()) #torch.Size([8])
    #print(y_coords.size()) #torch.Size([6])
    
    # Reshape the input tensor to size (BS, H * W)
    logit_flat = logit.view(batch_size, -1)

    # Apply the softmax function along the last dimension
    weights = F.softmax(logit_flat, dim=-1).view_as(logit)
    #print(weights.size()) #torch.Size([1, 6, 8])
    #print(weights.sum(2).size()) #torch.Size([1, 6])
    #print(weights.sum(1).size()) #torch.Size([1, 8])

    
    y_coords = y_coords.view(1, -1).expand(batch_size, -1)
    x_coords = x_coords.view(1, -1).expand(batch_size, -1)
    #print(x_coords.size()) #torch.Size([1, 8])
    #print(y_coords.size()) #torch.Size([1, 6])

    # Compute the weighted sum of coordinates using softmax probabilities
    x_pred = torch.sum(weights.sum(1) * x_coords, dim=1)
    y_pred = torch.sum(weights.sum(2) * y_coords, dim=1)
    
    #print(y_pred.size()) #torch.Size([1])
    #print(x_pred.size()) #torch.Size([1])

    # Stack and return the soft-argmax coordinates
    argmax_coords = torch.stack((x_pred, y_pred), dim=1)
    return argmax_coords
    #weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    #return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
    #                    (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    batch_size, height, width = logit.size()

    # Create coordinate grids
    #torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]
    #torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]
    
    x_coords = torch.linspace(-1, 1, width, device=logit.device)#torch.Size([8])
    y_coords = torch.linspace(-1, 1, height, device=logit.device)#torch.Size([6])
    
    # Reshape the input tensor to size (BS, H * W)
    logit_flat = logit.view(batch_size, -1)

    # Apply the softmax function along the last dimension
    #torch.Size([1, 6, 8])
    weights = F.softmax(logit_flat, dim=-1).view_as(logit)

    # Compute the weighted sum of coordinates using softmax probabilities
    x_pred = torch.sum(weights.sum(1) * x_coords, dim=1)
    y_pred = torch.sum(weights.sum(2) * y_coords, dim=1)

    # Stack and return the soft-argmax coordinates
    argmax_coords = torch.stack((x_pred, y_pred), dim=1)
    return argmax_coords
    #weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    #return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
    #                    (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner_b(torch.nn.Module):
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
            self.conv_block.add_module('down%d'%i, self.Block(n_input_cahnnels, l, kernel_size, 2))
            n_input_cahnnels = l

        self.up_block = torch.nn.Sequential()
        for i, l in list(enumerate(layers))[::-1]:
            self.up_block.add_module('up%d'%i, self.UpBlock(n_input_cahnnels, l, kernel_size, 2,self.use_skip))
            n_input_cahnnels = l
            if self.use_skip:
                n_input_cahnnels += skip_layer_size[i]

        self.classifier = torch.nn.Conv2d(n_input_cahnnels, n_output_channels, 1)


        #raise NotImplementedError('FCN.__init__')

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        normalize=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])
        z = normalize(img)
        conv_input = []
        for i in range(self.n_conv_layer):
            # Add all the information required for skip connections
            conv_input.append(z)
            z = self.conv_block._modules['down%d'%i](z)

        z=self.conv_block(img)
        for i in reversed(range(self.n_conv_layer)):
            z = self.up_block._modules['up%d'%i](z,conv_input[i])
        
        encoder = self.classifier(z)
        encoder = torch.squeeze(encoder,dim=1)
        #print('encoder size =',encoder.size())
        decoder = spatial_argmax(encoder[:, 0])
        return decoder#, self.size(z)

        
        #raise NotImplementedError("Planner.forward")

class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 64, 128]):
        super().__init__()

        """
        Your code here
        """
        
        h, _conv = 3, []
        for c in channels:
            _conv.extend(self.conv_block(c, h))
            h = c
        _conv.append(torch.nn.Conv2d(h, 1, kernel_size=1))
        self._conv = torch.nn.Sequential(*_conv)

        #raise NotImplementedError('Planner.__init__')
    def conv_block(self,c, h):
        #conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 5, 2, 2), torch.nn.ReLU(True)]
        return [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, kernel_size=5, stride=2, padding=2), torch.nn.ReLU(True)]

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        return spatial_argmax(x[:, 0])

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
