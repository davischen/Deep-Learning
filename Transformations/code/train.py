from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum= args.momentum, weight_decay=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    
    train_data = load_data(args.path_train, transform=transform, num_workers=args.num_workers)
    #valid_data = load_detection_data(args.path_valid, num_workers=args.num_workers)
    
    loss = torch.nn.L1Loss()

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals = []
        for data_batch in train_data:
            img, label = data_batch[0].to(device), data_batch[1].to(device)

            pred = model(img)
            # Continuous version of focal loss
            loss_val = loss(pred, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 100 == 0:
                    log(train_logger, img, label, pred, global_step)
           
            loss_vals.append(loss_val.detach().cpu().numpy()) #add batch loss_val to loss_vals list (with an s)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        avg_loss = np.mean(loss_vals)
        if train_logger is None:
            print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))

        save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-pt', '--path_train', type=str, default='drive_data')
    parser.add_argument('-nw', '--num_workers', type=int, default=4)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)

    parser.add_argument('-mom', '--momentum', type=float, default=0.9)
    parser.add_argument('-n', '--num_epoch', type=int, default=60)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)
    args = parser.parse_args()
    train(args)
