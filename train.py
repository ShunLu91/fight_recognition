import os
import torch
import argparse

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from newDataset import VideoDataset
from network import R2Plus1DClassifier


def argument_parser():
    parser = argparse.ArgumentParser(description="fight recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="R2Plus1D")
    parser.add_argument("--data_dir", type=str, default="./data/Fight/Fight-dataset-2020")
    parser.add_argument("--snapshots", type=str, default="./snapshots")
    parser.add_argument("--debug", action='store_false')
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate of feature extractor')
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
print(args)


def train_model(root, num_classes, layer_sizes, num_epochs, model_path):
    if args.model == 'R2Plus1D':
        model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes)

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # the scheduler divides the lr by 10 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.4)

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

    best_acc = 0
    start_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        print('epoch:', epoch)

        for phase in ['train', 'val']:

            loss = 0.0
            running_corrects = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs).cuda()
                labels = Variable(labels.long()).cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    _loss = criterion(outputs, labels)
                    if phase == 'train':
                        _loss.backward()
                        optimizer.step()
                loss += _loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step(epoch)

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

            with SummaryWriter(logdir='logdir/', comment='train_loss') as writer:
                if phase == 'train':
                    writer.add_scalar('train_loss', epoch_loss, epoch)
                    writer.add_scalar('train_acc', epoch_acc, epoch)
                else:
                    writer.add_scalar('val_loss', epoch_loss, epoch)
                    writer.add_scalar('val_acc', epoch_acc, epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_acc': epoch_acc
                }
                torch.save(state, os.path.join(model_path, 'epoch_{:d}_acc{:4f}.pth'.format(epoch, epoch_acc)))

        print('-' * 60)
        print('')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.snapshots):
        os.makedirs(args.snapshots)
    if not os.path.exists('logdir/'):
        os.makedirs('logdir/')

    train_model(
        root=os.path.join(args.data_dir, 'videos'),
        num_classes=2,
        layer_sizes=[2, 2, 2, 2],
        num_epochs=args.epoch,
        model_path=args.snapshots
    )
