import torch
import torch.nn.functional as F
import torchvision


def extract_peak_a(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    
    """
    # Apply max pooling to find local maxima
    # Get local maxima peaks
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    #heatmap.size()=torch.Size([96, 128])
    #points size=torch.Size([96, 128])
    #filter data by greater than maxima peaks and min score
    points = (heatmap > min_score) & (heatmap >= max_cls)
    #filter peaks = torch.Size(253), serialize array
    peaks = heatmap[points]
    #============================
    #Limit the number of peaks using torch.topk
    #return value and index, and max_det number peaks
    #[(-2.059535503387451, 63, 53)....]
    # Find peaks that satisfy the conditions min(len(peaks), max_det)
    #============================
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    score, loc = torch.topk(possible_det.view(-1), min(len(peaks), max_det), sorted=True)
    # Get the values, indices, and coordinates of the peaks
    list_of_peaks = []
    #way 2
    for s,l in zip(score.cpu(), loc.cpu()):
      if s > min_score:
        list_of_peaks.append(
              (float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1)))
    return list_of_peaks

    #way 2
    #return indices if points = True
    #loc = torch.nonzero(points == True)
    # Get the values, indices, and coordinates of the peaks
    #values = top_values.values
    #for i in range(len(top_values.values)):
    #  list_of_peaks.append(
    #        (top_values.values[i].item(), loc[top_values.indices[i]][1].item(), (loc[top_values.indices[i]][0].item())))
    #return list_of_peaks

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    # Convert heatmap to PyTorch tensor
    heatmap = torch.tensor(heatmap)

    # Apply max pooling to find local maxima
    pooled = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)
    points=(heatmap == pooled[0, 0]) & (heatmap > min_score)
    # Find peaks that satisfy the conditions
    peaks = torch.nonzero(points)
    #print(peaks)
    # Limit the number of peaks
    top_values = torch.topk(heatmap[peaks[:, 0], peaks[:, 1]], min(len(peaks), max_det), sorted=True)
    
    # Get the values and indices of the peaks
    values = top_values.values
    indices = peaks[top_values.indices]
    #print(top_values.indices)
    # Calculate cx and cy coordinates
    cx = indices[:, 1]
    cy = indices[:, 0]
    # Convert peaks to a list of tuples
    results = [(score.item(), cx_item.item(), cy_item.item()) for score, cx_item, cy_item in zip(values, cx, cy)]
    #print(results[0])
    return results

def extract_peak_b(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    #heatmap as the input to extract_peak is a 2d heatmap
    #Implement local maxima detection using max pooling.
    #The parameters kernel_size, stride, padding, dilation
    #print(heatmap.size())
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    #@min_score: Only return peaks greater than min_score
    #@max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
    #heatmap.size()=torch.Size([96, 128])
    #points size=torch.Size([96, 128])
    #filter data by greater than maxima peaks and min score
    points = (heatmap > min_score) & (heatmap >= max_cls)
    #filter peaks = torch.Size(253), serialize array
    peaks = heatmap[points]
    #print(peaks.size())
    #It ensures that points is a 2D tensor of size (H, W).
    points.squeeze_(0).squeeze_(0).size()
    #ranking
    top_values = torch.topk(peaks, min(len(peaks), max_det), sorted=True)
    #print(top_values)
    list_of_peaks = []
    indices = (points == True).nonzero()
    for i in range(len(top_values.values)):
        list_of_peaks.append(
            (top_values.values[i].item(), indices[top_values.indices[i]][1].item(), (indices[top_values.indices[i]][0].item())))
    #print(list_of_peaks[0])
    return list_of_peaks

def extract_peak4(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)
    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]


def extract_peak_test(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    #example:
    tensor = torch.Tensor([[1, 2, 2, 7], [3, 1, 2, 4], [3, 1, 9, 4]])
    point22 = tensor >=2
    peaks22 = tensor[point22]
    top_values = torch.topk(peaks22, 3, sorted=True)
    loc = torch.nonzero(point22 == True)
    
    print(point22)
    print(loc)
    print(peaks22)
    print(top_values.values)
    print(top_values.indices[0])
    print('-----')

    tensor_det = tensor - (2 > tensor).float() * 1e5
    print(tensor_det.view(-1))
    score, loc = torch.topk(tensor_det.view(-1), 3)
    print(score)
    print(loc)
    list_of_peaks22 = []
    for s,l in zip(score.cpu(), loc.cpu()):
        print(int(l) % tensor.size(1))
        print(int(l) // tensor.size(1))
          #You can use % and // by width along with one of the outputs from topk to get cx, cy
        list_of_peaks22.append(
              (float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1)))
    print(list_of_peaks22)
    print('-----')

    #return ensor([[0, 1],
    #    [0, 2],
    #    [1, 2]])

    #heatmap as the input to extract_peak is a 2d heatmap
    #Implement local maxima detection using max pooling.
    #The parameters kernel_size, stride, padding, dilation
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    print('class size=')
    print(max_cls.size())

    print('filter size=')
    points = (heatmap > min_score) & (heatmap >= max_cls)
    #filter peaks
    peaks = heatmap[points]
    print(heatmap)
    #peaks_2 size=torch.Size([181, 2])
    #return indices if points = True
    loc = torch.nonzero(points == True)
    print(loc)
    #return value and index, and max_det number peaks
    top_values = torch.topk(peaks, min(len(peaks), max_det), sorted=True)
    # Get the values, indices, and coordinates of the peaks
    values = top_values.values
    indices = top_values.indices
    print(top_values.indices)
    print(indices.size())
    print(loc[top_values.indices[0]])
    cx = indices[:, 1]
    cy = indices[:, 0]
    print('result=')
    print(cx.szie())
    print(cy.size())

    #The score is the value of the heatmap at the peak, and is used to rank detections later. A peak with a high score is a more important detection.
    #min(len(peaks), max_det) = if max_det > possible_det.numel(): max_det = possible_det.numel()
    #
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    score, loc = torch.topk(possible_det.view(-1), min(possible_det.numel(), max_det))
    list_of_peaks = []
    for s,l in zip(score.cpu(), loc.cpu()):
        if s > min_score:
          #You can use % and // by width along with one of the outputs from topk to get cx, cy
          list_of_peaks.append(
              (float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1)))
    return list_of_peaks
    
    #possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    #if max_det > possible_det.numel():
    #    max_det = possible_det.numel()
    #score, loc = torch.topk(possible_det.view(-1), max_det)
    #return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
    #        for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]
    #raise NotImplementedError('extract_peak')


    


class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
      def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()

            self.net = torch.nn.Sequential(
              torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,stride=stride),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2),
              torch.nn.BatchNorm2d(n_output),
              torch.nn.ReLU(),
              torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2),
              torch.nn.BatchNorm2d(n_output)
            )
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)
               
      def forward(self, x):
            return F.relu(self.net(x)+ self.skip(x))

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

    def __init__(self, layers=[16, 32, 64, 128], n_output_class=3, n_class=3, kernel_size=3, use_skip=True):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()

        n_input_cahnnels = 3
        self.use_skip = use_skip
        self.n_conv_layer = len(layers)
        skip_layer_size = [3] + layers[:-1]#layers[:-1] is [16, 32, 64], so skip_layer_size is [3, 16, 32, 64]
        self.conv_block=torch.nn.Sequential()
        for i, l in enumerate(layers):
            self.conv_block.add_module('down_conv%d'%i, self.Block(n_input_cahnnels, l, kernel_size, 2))
            n_input_cahnnels = l
        #for i in self.conv_block._modules:
        #    print(i)
        #print('----------')
        self.dense_block = torch.nn.Sequential()
        for i, l in list(enumerate(layers))[::-1]:
            self.dense_block.add_module('up_conv%d'%i, self.UpBlock(n_input_cahnnels, l, kernel_size, 2, self.use_skip))
            n_input_cahnnels = l
            if self.use_skip:
                n_input_cahnnels += skip_layer_size[i]

        #for i in self.dense_block._modules:
        #    print(i)

        #print('----------end')
        self.classifier = torch.nn.Conv2d(n_input_cahnnels, n_output_class, 1)
        self.size = torch.nn.Conv2d(n_input_cahnnels, 2, 1)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        #Using the mean and std of Imagenet is a common practice
        #mean = torch.Tensor([0.485, 0.456, 0.406])#([0.3521554, 0.30068502, 0.28527516])
        #std = torch.Tensor([0.229, 0.224, 0.225]) #([0.18182722, 0.18656468, 0.15938024])
        #image = (image - mean) / std
        #z = (x - mean[None, :, None, None].to(x.device)) / std[None, :, None, None].to(x.device)
        #normalize=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
				#std=[0.229, 0.224, 0.225])
        normalize=torchvision.transforms.Normalize(mean=[0.3521554, 0.30068502, 0.28527516],
				std=[0.18182722, 0.18656468, 0.15938024])
        z = normalize(x)
        
        up_input = []
        for i in range(self.n_conv_layer):
            # Add all the information required for skip connections
            up_input.append(z)
            z = self.conv_block._modules['down_conv%d' % i](z)

        for i in reversed(range(self.n_conv_layer)):
            z = self.dense_block._modules['up_conv%d' % i](z,up_input[i])

        return self.classifier(z), self.size(z)


    def detect(self, image, **kwargs):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        #predict a dense heatmap of object centers,
        cls, size = self.forward(image[None])
        size = size.cpu()
        heapmap=cls[0]
        res = []
        #Each "peak" (local maxima) in this heatmap corresponds to a detected object
        for c in cls[0]:
            peaks = extract_peak(c,max_det=30, **kwargs)
            #the output is a list of peaks with score, x and y location of the center of the peak
            res.append([(score, x, y, float(size[0, 0, y, x]), float(size[0, 1, y, x]))
                 for score, x, y in peaks])
        return res
        #return [[(s, x, y, float(size[0, 0, y, x]), float(size[0, 1, y, x]))
        #         for s, x, y in extract_peak(c, max_det=30, **kwargs)] for c in cls[0]]
        #raise NotImplementedError('Detector.detect')


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
