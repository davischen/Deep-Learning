from IPython.utils.text import string
from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch

def train(args):
    model = model_factory[args.model]()

    """
    Your code here

    """
    #raise NotImplementedError('train')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))
    
    loss = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    train_data = load_data(args.path_train,num_workers=args.num_workers,batch_size=args.batch_size)
    valid_data = load_data(args.path_valid,num_workers=args.num_workers,batch_size=args.batch_size)

    for each_epoch in range(args.epochs):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for data_batch in train_data:
            predit_label, actual_label = model(data_batch[0].to(device)), data_batch[1].to(device)

            loss_val = loss(predit_label, actual_label)
            acc_val = accuracy(predit_label, actual_label)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)

        model.eval()
        for data_batch in valid_data:
            predit_label, actual_label = model(data_batch[0].to(device)), data_batch[1].to(device)
            vacc_vals.append(accuracy(predit_label, actual_label).detach().cpu().numpy())
        avg_vacc = sum(vacc_vals) / len(vacc_vals)

        print('epoch %-3d \t averge loss = %0.3f \t accuracy = %0.3f \t val accuracy = %0.3f' % (each_epoch, avg_loss, avg_acc, avg_vacc))
    save_model(model)
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here
    parser.add_argument('-pt', '--path_train', type=str, default='data/train')
    parser.add_argument('-pv', '--path_valid', type=str, default='data/valid')
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-n', '--epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    
    args = parser.parse_args()
    train(args)
