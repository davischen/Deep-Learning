from .models import CNNClassifier, save_model,ClassificationLoss
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    #train_logger, valid_logger = None, None
    #if args.log_dir is not None:
    #    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    #    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = ClassificationLoss()

    train_data = load_data(args.path_train,num_workers=args.num_workers,batch_size=args.batch_size)
    valid_data = load_data(args.path_valid,num_workers=args.num_workers,batch_size=args.batch_size)

    for epoch in range(args.num_epoch):                        
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for data_batch in train_data:
                              
          #Begin training loop
            img, actual_label = data_batch[0].to(device), data_batch[1].to(device)  
            predit_label = model(img) #PARALLEL PROCESSING....img is entire BATCH, this returns (128, 6)

            loss_val = loss(predit_label, actual_label)  #Loss for one training batch, is this (128, 6)?  mean???
            acc_val = accuracy(predit_label, actual_label) #Accuracy on (128,6) logits of (128,3*64*64) batch

            loss_vals.append(loss_val.detach().cpu().numpy()) #add batch loss_val to loss_vals list (with an s)
            acc_vals.append(acc_val.detach().cpu().numpy())  #add accuracy acc_val to acc_vals list (with an s)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
                                           
          #end training loop
        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)

        model.eval()   #do this just for validation... Tensorboard....
        
        for data_batch in valid_data:  #iterate through validation data
            predit_label, actual_label = model(data_batch[0].to(device)), data_batch[1].to(device)
            vacc_vals.append(accuracy(predit_label, actual_label).detach().cpu().numpy())
        avg_vacc = sum(vacc_vals) / len(vacc_vals)

        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))
    

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-pt', '--path_train', type=str, default='data/train')
    parser.add_argument('-pv', '--path_valid', type=str, default='data/valid')
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()
    train(args)
