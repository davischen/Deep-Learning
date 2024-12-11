from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt

import inspect

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, args.path_train), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, args.path_valid), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    import torch
    # Find the device available to use using torch library
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Move model to the device specified above
    model = CNNClassifier().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    # Set the optimizer function using torch.optim as optim library
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    # Set the error function using torch.nn as nn library
    loss = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30, cooldown=5)

    transform = eval(args.transform,
                     {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})
    #train_data = load_data('data/train', transform=transform, num_workers=4)
    #valid_data = load_data('data/valid', num_workers=4)
    train_data = load_data(args.path_train, transform=transform, num_workers=args.num_workers, batch_size=args.batch_size)
    valid_data = load_data(args.path_valid, num_workers=args.num_workers, batch_size=args.batch_size)

    best_accuracy = 0
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals = []
        train_metrics = ConfusionMatrix(len(LABEL_NAMES))#
        for data_batch in train_data:
            if train_logger is not None:
                train_logger.add_images('augmented_image', img[:4])
            img, actual_label = data_batch[0].to(device), data_batch[1].to(device)  
            predit_label = model(img)
            train_metrics.add(predit_label.argmax(1), actual_label)#save predict label and actual label

            loss_val = loss(predit_label, actual_label)
            
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            if train_logger:
              train_logger.add_scalar('global_accuracy', train_metrics.global_accuracy, global_step)
              train_logger.add_scalar('average_accuracy', train_metrics.average_accuracy, global_step)
              train_logger.add_scalar('iou', train_metrics.iou, global_step)

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
            train_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step=global_step)
        

        #===============evaluate model====================
        model.eval()
        validate_metrics = ConfusionMatrix(len(LABEL_NAMES))
        for data_batch in valid_data:
            img, actual_label = data_batch[0].to(device), data_batch[1].to(device)  
            predit_label = model(img)
            validate_metrics.add(predit_label.argmax(1), actual_label)

        #if valid_logger:
        #  acc_logging(valid_logger,validate_metrics,global_step)
        if valid_logger:
            valid_logger.add_scalar('global_accuracy', validate_metrics.global_accuracy, global_step)
            valid_logger.add_scalar('average_accuracy', validate_metrics.average_accuracy, global_step)
            valid_logger.add_scalar('iou', validate_metrics.iou, global_step)
        #================================================

        #if valid_logger is None or train_logger is None:
        #    print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, confusion_train.global_accuracy,
        #    validate_metrics.global_accuracy))
        
        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t lr %0.5f \t loss = %0.3f \t accuracy = %0.3f \t val accuracy = %0.3f \t iou = %0.3f \t val iou = %0.3f' %
                  (epoch,optimizer.param_groups[0]['lr'],avg_loss, train_metrics.global_accuracy, validate_metrics.global_accuracy, train_metrics.iou, validate_metrics.iou))
        
        scheduler.step(validate_metrics.global_accuracy)
        if best_accuracy <= validate_metrics.global_accuracy or epoch<=20:
          best_accuracy = validate_metrics.global_accuracy
          save_model(model)
          


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-pt', '--path_train', type=str, default='data/train')
    parser.add_argument('-pv', '--path_valid', type=str, default='data/valid')
    parser.add_argument('-nw', '--num_workers', type=int, default=4)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
