import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data,accuracy,DetectionSuperTuxDataset
from . import dense_transforms
import torch.utils.tensorboard as tb

## Copied this code from the test code
def point_in_box(p, x0, y0, x1, y1):
    return x0 <= p[0] < x1 and y0 <= p[1] < y1


def point_close(p, x0, y0, x1, y1, d=5):
    return ((x0 + x1 - 1) / 2 - p[0]) ** 2 + ((y0 + y1 - 1) / 2 - p[1]) ** 2 < d ** 2


def box_iou(p, x0, y0, x1, y1, t=0.5):
    iou = abs(min(p[0] + p[2], x1) - max(p[0] - p[2], x0)) * abs(min(p[1] + p[3], y1) - max(p[1] - p[3], y0)) / \
          abs(max(p[0] + p[2], x1) - min(p[0] - p[2], x0)) * abs(max(p[1] + p[3], y1) - min(p[1] - p[3], y0))
    return iou > t


class PR:
    def __init__(self, min_size=20, is_close=point_in_box):
        self.min_size = min_size
        self.total_det = 0
        self.det = []
        self.is_close = is_close

    def add(self, d, lbl):
        small_lbl = [b for b in lbl if abs(b[2] - b[0]) * abs(b[3] - b[1]) < self.min_size]
        large_lbl = [b for b in lbl if abs(b[2] - b[0]) * abs(b[3] - b[1]) >= self.min_size]
        used = [False] * len(large_lbl)
        for s, *p in d:
            match = False
            for i, box in enumerate(large_lbl):
                if not used[i] and self.is_close(p, *box):
                    match = True
                    used[i] = True
                    break
            if match:
                self.det.append((s, 1))
            else:
                match_small = False
                for i, box in enumerate(small_lbl):
                    if self.is_close(p, *box):
                        match_small = True
                        break
                if not match_small:
                    self.det.append((s, 0))
        self.total_det += len(large_lbl)

    @property
    def curve(self):
        true_pos, false_pos = 0, 0
        r = []
        for t, m in sorted(self.det, reverse=True):
            if m:
                true_pos += 1
            else:
                false_pos += 1
            prec = true_pos / (true_pos + false_pos)
            recall = true_pos / self.total_det
            r.append((prec, recall))
        return r

    @property
    def average_prec(self, n_samples=11):
        max_prec = 0
        cur_rec = 1
        precs = []
        for prec, recall in self.curve[::-1]:
            max_prec = max(max_prec, prec)
            while cur_rec > recall:
                precs.append(max_prec)
                cur_rec -= 1.0 / n_samples
        return sum(precs) / max(1,len(precs))

class calc_metric(object):
    def __init__(self,model):
        # Compute detections
        self.pr_box = [PR() for _ in range(3)]
        self.pr_dist = [PR(is_close=point_close) for _ in range(3)]
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def add(self, dataset):
        for img, *gts in dataset:
            d = self.model.detect(img.to(self.device))
            for i, gt in enumerate(gts):
                self.pr_box[i].add([j[1:] for j in d if j[0] == i], gt)
                self.pr_dist[i].add([j[1:] for j in d if j[0] == i], gt)
    
    def calc(self):
        ap0 = self.pr_box[0].average_prec
        ap1 = self.pr_box[1].average_prec
        ap2 = self.pr_box[2].average_prec
        apb0 = self.pr_dist[0].average_prec
        apb1 = self.pr_dist[1].average_prec
        apb2 = self.pr_dist[2].average_prec
        return (ap0, ap1, ap2, apb0, apb1, apb2)

def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    #raise NotImplementedError('train')
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det_display.th')))

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum= args.momentum, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    
    train_data = load_detection_data(args.path_train, transform=transform, num_workers=args.num_workers)
    #valid_data = load_detection_data(args.path_valid, num_workers=args.num_workers)
    #Use the sigmoid and BCEWithLogitsLoss
    peak_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    size_loss = torch.nn.MSELoss(reduction='none')



    image_id = np.random.randint(0,100)
    valid_metric_dataset  = DetectionSuperTuxDataset(args.path_valid, min_size=0)
    transformer = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),  dense_transforms.ToTensor(), dense_transforms.ToHeatmap(radius=10)])  #, dense_transforms.ColorJitter()
    valid_transformer = dense_transforms.Compose([dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]) 
    train_dataset = DetectionSuperTuxDataset(args.path_train, transform=transformer)
    valid_dataset = DetectionSuperTuxDataset(args.path_valid, transform=valid_transformer)
    sample_image = train_dataset[image_id]
    sample_valid_image = valid_dataset[image_id]

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals, vacc_vals, peak_vals,size_vals = [], [], [],[],[]
        for data_batch in train_data:
            img, target_heatmap, target_heatmapsize = data_batch[0].to(device), data_batch[1].to(device), data_batch[2].to(device)

            size_w, _ = target_heatmap.max(dim=1, keepdim=True)

            input_heatmap, input_heatmap_size = model(img)
            # Continuous version of focal loss
            prob_heat = torch.sigmoid(input_heatmap * (1-2*target_heatmap))
            peak_loss_val = (peak_loss(input_heatmap, target_heatmap)*prob_heat).mean() / prob_heat.mean()
            size_loss_val = (size_w * size_loss(input_heatmap_size, target_heatmapsize)).mean() / size_w.mean()
            loss_val = peak_loss_val + size_loss_val * args.size_weight

            
            #acc_val = accuracy(input_heatmap, target_heatmap) #Accuracy on (128,6) logits of (128,3*64*64) batch
            

            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, target_heatmap, input_heatmap, global_step)

            if train_logger is not None:
                train_logger.add_scalar('label_loss', peak_loss_val, global_step)
                train_logger.add_scalar('size_loss', size_loss_val, global_step)
                train_logger.add_scalar('loss', loss_val, global_step)
           
            loss_vals.append(loss_val.detach().cpu().numpy()) #add batch loss_val to loss_vals list (with an s)
            #acc_vals.append(acc_val.detach().cpu().numpy())  #add accuracy acc_val to acc_vals list (with an s)
            peak_vals.append(peak_loss_val.detach().cpu().numpy())
            size_vals.append(peak_loss_val.detach().cpu().numpy())
            optimizer.zero_grad()
            # Calculate gradients (backpropogation)
            loss_val.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            global_step += 1

        avg_loss = sum(loss_vals) / len(loss_vals)
        #avg_acc = sum(acc_vals) / len(acc_vals)
        avg_peak = sum(peak_vals) / len(peak_vals)
        avg_size = sum(size_vals) / len(size_vals)
        model.eval()   #do this just for validation... Tensorboard....
        
        if valid_logger is None or train_logger is None:
            #train_logger.add_scalar("accuracy", avg_acc, global_step=global_step)
            #valid_logger.add_scalar("valid accuracy", avg_vacc, global_step=global_step)
            print('epoch %-3d \t loss = %0.3f \t peak loss = %0.3f \t size loss = %0.3f \t' % (epoch, avg_loss,avg_peak,avg_size))

        #print sample
        
        sample = sample_image[0].to(device)
        im = sample_image[0].unsqueeze(0)
        #heatmap = model(im.to(device))
        #heatmap = heatmap.squeeze(0)
        detection = model.detect(sample)
        print(detection)
        train_logger.add_image('Original',sample_image[0].cpu(), global_step=epoch)
        #train_logger.add_image('Heatmap',heatmap.cpu(), global_step=epoch)
        #train_logger.add_image('Heatmap_Sigmoid',torch.sigmoid(heatmap.cpu()), global_step=epoch)
        train_logger.add_image('Actual',sample_image[1].cpu(), global_step=epoch)
        
        #for data_batch in valid_data:  #iterate through validation data
        #    predit_label, actual_label = model(data_batch[0].to(device)), data_batch[1].to(device)
        #    vacc_vals.append(accuracy(predit_label, actual_label).detach().cpu().numpy())
        #avg_vacc = sum(vacc_vals) / len(vacc_vals)

        #Validate
        """
        if epoch % 10 == 0:
            print('validate')
            model.eval()
            valid_metric = calc_metric(model)
            valid_metric.add(valid_metric_dataset)
            ap0, ap1, ap2, apb0, apb1, apb2 = valid_metric.calc()
            
            #Record Valid results
            valid_logger.add_scalar('AP0', ap0, global_step=epoch)
            valid_logger.add_scalar('AP1', ap1, global_step=epoch)
            valid_logger.add_scalar('AP2', ap2, global_step=epoch)
            valid_logger.add_scalar('AP_box0', apb0, global_step=epoch)
            valid_logger.add_scalar('AP_box1', apb1, global_step=epoch)
            valid_logger.add_scalar('AP_box2', apb2, global_step=epoch)
            
            
            sample = sample_valid_image[0].to(device)
            im = sample_valid_image[0].unsqueeze(0)
            heatmap = model(im.to(device))
            heatmap = heatmap.squeeze(0)
            detection = model.detect(sample)
            print(detection)
            
            valid_logger.add_image('Original',sample_valid_image[0].cpu(), global_step=epoch)
            valid_logger.add_image('Heatmap',heatmap.cpu(), global_step=epoch)
            valid_logger.add_image('Heatmap_Sigmoid',torch.sigmoid(heatmap.cpu()), global_step=epoch)
            valid_logger.add_image('Actual',sample_valid_image[1].cpu(), global_step=epoch)
            
            train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step=epoch)
        """
        
        save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-pt', '--path_train', type=str, default='dense_data/train')
    parser.add_argument('-pv', '--path_valid', type=str, default='dense_data/valid')
    parser.add_argument('-nw', '--num_workers', type=int, default=4)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)

    parser.add_argument('-mom', '--momentum', type=float, default=0.9)
    parser.add_argument('-n', '--num_epoch', type=int, default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.7, 0.8, 0.8, 0.2), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)
    args = parser.parse_args()
    train(args)
