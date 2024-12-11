import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
from torchsummary import summary
from torchvision import transforms

def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    import torch
    # Find the device available to use using torch library
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Move model to the device specified above
    model = FCN().to(device)
    #summary(model, (3, 96, 128))
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    # Set the optimizer function using torch.optim as optim library
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
    # Set the error function using torch.nn as nn library
    loss = torch.nn.CrossEntropyLoss(weight=w / w.mean()).to(device)

    import inspect
    #transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    transform = dense_transforms.Compose([
      dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
      dense_transforms.RandomHorizontalFlip(),
      dense_transforms.ToTensor()
    ])
    train_data = load_dense_data(args.path_train, transform=transform, num_workers=args.num_workers)#, batch_size=args.batch_size
    valid_data = load_dense_data(args.path_valid, num_workers=args.num_workers)#, batch_size=args.batch_size

    best_accuracy = 0
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals = []
        train_metrics = ConfusionMatrix()
        for data_batch in train_data:
            img, actual_label = data_batch[0].to(device), data_batch[1].to(device).long()  
            predit_label = model(img)
            train_metrics.add(predit_label.argmax(1), actual_label)#save predict label and actual label
        
            loss_val = loss(predit_label, actual_label)
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, actual_label, predit_label, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            loss_vals.append(loss_val.detach().cpu().numpy()) #add batch loss_val to loss_vals list (with an s)
            
            optimizer.zero_grad()
            # Calculate gradients (backpropogation)
            loss_val.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            global_step += 1
        
        
        avg_loss = sum(loss_vals) / len(loss_vals)

        if train_logger:
            train_logger.add_scalar('global_accuracy', train_metrics.global_accuracy, global_step)
            train_logger.add_scalar('average_accuracy', train_metrics.average_accuracy, global_step)
            train_logger.add_scalar('iou', train_metrics.iou, global_step)

        model.eval()
        validate_metrics = ConfusionMatrix()
        for data_batch in valid_data:
            img, actual_label = data_batch[0].to(device), data_batch[1].to(device).long()  
            predit_label = model(img)
            validate_metrics.add(predit_label.argmax(1), actual_label)

        if valid_logger is not None:
            log(valid_logger, img, actual_label, predit_label, global_step)

        if valid_logger:
            valid_logger.add_scalar('global_accuracy', validate_metrics.global_accuracy, global_step)
            valid_logger.add_scalar('average_accuracy', validate_metrics.average_accuracy, global_step)
            valid_logger.add_scalar('iou', validate_metrics.iou, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f \t iou = %0.3f \t val iou = %0.3f' %
                  (epoch,avg_loss, train_metrics.global_accuracy, validate_metrics.global_accuracy, train_metrics.iou, validate_metrics.iou))
        if best_accuracy <= validate_metrics.global_accuracy or epoch<=10:
          best_accuracy = validate_metrics.global_accuracy
          save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-pt', '--path_train', type=str, default='dense_data/train')
    parser.add_argument('-pv', '--path_valid', type=str, default='dense_data/valid')
    parser.add_argument('-nw', '--num_workers', type=int, default=4)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    #parser.add_argument('-t', '--transform',
    #                    default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')


    args = parser.parse_args()
    train(args)
