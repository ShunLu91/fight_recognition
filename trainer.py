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

# Use GPU if available else revert to CPU

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    assert (torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True

    model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes)

    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda() # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # hyperparameters as given in paper sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.4)  # the scheduler divides the lr by 10 every 10 epochs

    # prepare the dataloaders into a dict
    train_path = 'train_split.txt'
    val_path = 'val_split.txt'

    train_dataloader = DataLoader(VideoDataset(train_path, root=root, mode='train'), batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(val_path, root=root, mode='val'), batch_size=4, num_workers=4)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    # check if there was a previously saved checkpoint
    # if os.path.exists(path):
    #     # loads the checkpoint
    #     checkpoint = torch.load(path)
    #     print("Reloading from previously saved checkpoint")
    #
    #     # restores the model and optimizer state_dicts
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['opt_dict'])
    #
    #     # obtains the epoch the training is to resume from
    #     epoch_resume = checkpoint["epoch"]

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
                with torch.set_grad_enabled(phase=='train'):
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
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")


if __name__ == '__main__':
    train_model(root='data/', num_classes=2, layer_sizes=[2, 2, 2, 2], num_epochs=100, model_path="model/")