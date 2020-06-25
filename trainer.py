import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# from dataset import VideoDataset, VideoDataset1M
from newDataset import VideoDataset
from network import R2Plus1DClassifier

import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--dataset", type=str, default="RAP")
    parser.add_argument("--data_dir", type=str, default="./data/Fight/Fight-dataset-2020")
    parser.add_argument("--debug", action='store_false')
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr_ft", type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument("--lr_new", type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument('--device', default=0, type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')

    return parser


args = argument_parser().parse_args()


def train_model(root='videos/', num_classes=2, layer_sizes=[2, 2, 2, 2], num_epochs=100, model_path="model/"):
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Args:
            num_classes (int): Number of classes in the data
            directory (str): Directory where the data is to be loaded from
            layer_sizes (list, optional): Number of blocks in each layer. Defaults to [2, 2, 2, 2], equivalent to ResNet18.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """

    # initalize the ResNet 18 version of this model
    torch.backends.cudnn.benchmark = True

    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes)

    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()  # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # hyperparameters as given in paper sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15,
                                          gamma=0.4)  # the scheduler divides the lr by 10 every 10 epochs

    # prepare the dataloaders into a dict
    train_path = os.path.join(args.data_dir, 'train_split.txt')
    val_path = os.path.join(args.data_dir, 'val_split.txt')

    train_dataloader = DataLoader(
        VideoDataset(train_path, root=root, mode='train'),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=16
    )
    val_dataloader = DataLoader(
        VideoDataset(val_path, root=root, mode='val'),
        batch_size=args.batchsize,
        num_workers=16
    )
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    best_acc = 0
    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
        # each epoch has a training and validation step, in that order
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                # move inputs and labels to the device the training is taking place on
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                inputs = Variable(inputs).cuda()
                labels = Variable(labels.long()).cuda()

                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false, 
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # we're interested in the indices on the max values, not the values themselves
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            if not os.path.exists('logdir/'):
                os.makedirs('logdir/')
            with SummaryWriter(logdir='logdir/', comment='train_loss') as writer:
                if phase == 'train':
                    writer.add_scalar('train_loss', epoch_loss, epoch)
                    writer.add_scalar('train_acc', epoch_acc, epoch)
                else:
                    writer.add_scalar('val_loss', epoch_loss, epoch)
                    writer.add_scalar('val_acc', epoch_acc, epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'val_acc': epoch_acc}
                torch.save(state, os.path.join(model_path, 'epoch_{:d}_acc{:4f}.pth'.format(epoch, epoch_acc)))

            elif epoch % 10 == 0:
                state = {"epcoh": epoch,
                         "optimizer_state_dict": optimizer.state_dict(),
                         "state_dict": model.state_dict(),
                         "val_acc": epoch_acc,
                         "best_acc": best_acc}
                torch.save(state, os.path.join(model_path, 'epoch_{:d}_acc{:4f}.pth'.format(epoch, epoch_acc)))

    time_elapsed = time.time() - start
    print(f"Training complete in {time_elapsed // 3600}h {(time_elapsed % 3600) // 60}m {time_elapsed % 60}s")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    train_model(
        root=os.path.join(args.data_dir),
        num_classes=2,
        layer_sizes=[2, 2, 2, 2],
        num_epochs=args.epoch,
        model_path="model/"
    )
