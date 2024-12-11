import torch
import torch.nn.functional as F

class ClassificationLoss(torch.nn.Module):
  def forward(self, input, target):
    return torch.nn.functional.cross_entropy(input, target)

class CNNClassifier(torch.nn.Module):
    def __init__(self,num_input_channel=3):
        """
        Your code here
        """
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(num_input_channel, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))   
            
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 6)
        #raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
        #raise NotImplementedError('CNNClassifier.forward')


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
