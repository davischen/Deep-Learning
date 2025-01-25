import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data,accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb

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
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum= args.momentum, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    
    train_data = load_detection_data(args.path_train, transform=transform, num_workers=args.num_workers)
    #valid_data = load_detection_data(args.path_valid, num_workers=args.num_workers)
    #Use the sigmoid and BCEWithLogitsLoss
    peak_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    size_loss = torch.nn.MSELoss(reduction='none')


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

        avg_loss = np.mean(loss_vals)
        #avg_acc = sum(acc_vals) / len(acc_vals)
        avg_peak = np.mean(peak_vals)
        avg_size = np.mean(size_vals)
        model.eval()   #do this just for validation... Tensorboard....
        
        if valid_logger is None or train_logger is None:
            #train_logger.add_scalar("accuracy", avg_acc, global_step=global_step)
            #valid_logger.add_scalar("valid accuracy", avg_vacc, global_step=global_step)
            print('epoch %-3d \t loss = %0.3f \t peak loss = %0.3f \t size loss = %0.3f \t' % (epoch, avg_loss,avg_peak,avg_size))

        #print sample
        
        #for data_batch in valid_data:  #iterate through validation data
        #    predit_label, actual_label = model(data_batch[0].to(device)), data_batch[1].to(device)
        #    vacc_vals.append(accuracy(predit_label, actual_label).detach().cpu().numpy())
        #avg_vacc = sum(vacc_vals) / len(vacc_vals)

        #Validate
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
    parser.add_argument('-n', '--num_epoch', type=int, default=140)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.7, 0.8, 0.8, 0.2), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)
    args = parser.parse_args()
    train(args)
