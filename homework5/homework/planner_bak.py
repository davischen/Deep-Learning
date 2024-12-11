import torch
import torch.nn.functional as F


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
        #raise NotImplementedError("Planner.forward")


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
