import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import models
import dataset
import utils
from fast_adv.attacks import DDN

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Adversarial TinyImageNet Training with the DDN Attack')

parser.add_argument('data', help='path to dataset')
parser.add_argument('--arch', '-a', default='resnet18',
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--save-folder', '--sf', required=True, help='folder where the models will be saved')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')

parser.add_argument('--evaluate', '--eval', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', type=str, help='path to latest checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrained', '--pt', dest='pretrained', action='store_true', help='use pre-trained model')

parser.add_argument('--batch-size', '-b', default=64, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--lr-step', '--learning-rate-step', default=5, type=int,
                    help='step size for learning rate decrease')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')

parser.add_argument('--adv', action='store_true', help='Use adversarial training')
parser.add_argument('--start-adv-epoch', '--sae', type=int, default=0,
                    help='epoch to start training with adversarial images')
parser.add_argument('--max-norm', type=float, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=100, type=int, help='number of steps for the attack')

parser.add_argument('--visdom-port', '--vp', type=int, help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')


def main():
    global args

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_mean = torch.tensor([0.4802, 0.4481, 0.3975]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.2770, 0.2691, 0.2821]).view(1, 3, 1, 1)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        m = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        m = models.__dict__[args.arch]()

    model = utils.NormalizedModel(m, image_mean, image_std)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            prec1 = checkpoint['prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.last_epoch = checkpoint['epoch'] - 1
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = dataset.TinyImageNet(args.data, mode='train', transform=train_transform)
    val_dataset = dataset.TinyImageNet(args.data, mode='val', transform=test_transform)

    if args.visdom_port:
        from visdom_logger.logger import VisdomLogger
        callback = VisdomLogger(port=args.visdom_port)
    else:
        callback = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True)

    attack = DDN(steps=args.steps, device=device)

    if args.evaluate:
        validate(val_loader, model, criterion, device, 0, callback=callback)
        return

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        print('Learning rate for epoch {}: {:.2e}'.format(epoch, optimizer.param_groups[0]['lr']))

        # train for one epoch
        train(train_loader, model, m, criterion, optimizer, attack, device, epoch, callback)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, device, epoch + 1, callback)

        utils.save_checkpoint(
            state={'epoch': epoch + 1,
                   'arch': args.arch,
                   'state_dict': model.state_dict(),
                   'prec1': prec1,
                   'optimizer': optimizer.state_dict()},
            filename=os.path.join(args.save_folder, 'checkpoint_{}.pth'.format(args.arch)))

        utils.save_checkpoint(
            state=model.state_dict(),
            filename=os.path.join(args.save_folder, '{}_epoch-{}.pt'.format(args.arch, epoch + 1)),
            cpu=True
        )


def train(train_loader, model, m, criterion, optimizer, attack, device, epoch, callback=None):
    model.train()
    cudnn.benchmark = True
    length = len(train_loader)

    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    losses_adv = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    adv_acc = utils.AverageMeter()
    l2_adv = utils.AverageMeter()

    end = time.time()
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device, non_blocking=True)

        if args.adv and epoch >= args.start_adv_epoch:
            model.eval()
            utils.requires_grad_(m, False)

            clean_logits = model(data)
            loss = criterion(clean_logits, labels)

            adv = attack.attack(model, data, labels)
            l2_norms = (adv - data).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - data, p=2, dim=0, maxnorm=args.max_norm) + data
            l2_adv.append(mean_norm.item())

            utils.requires_grad_(m, True)
            model.train()

            adv_logits = model(adv.detach())
            loss_adv = criterion(adv_logits, labels)

            loss_to_optimize = loss_adv

            losses_adv.append(loss_adv.item())
            l2_adv.append((adv - data).view(args.batch_size, -1).norm(p=2, dim=1).mean().item())
            adv_acc.append((adv_logits.argmax(1) == labels).sum().item() / args.batch_size)
        else:
            clean_logits = model(data)
            loss = criterion(clean_logits, labels)
            loss_to_optimize = loss

        optimizer.zero_grad()
        loss_to_optimize.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(clean_logits, labels, topk=(1, 5))
        losses.append(loss.item())
        top1.append(prec1)
        top5.append(prec5)

        # measure elapsed time
        batch_time.append(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == length:

            if args.adv and epoch >= args.start_adv_epoch:
                print('Epoch: [{0:>2d}][{1:>3d}/{2:>3d}] Time {batch_time.last_avg:.3f}'
                      '\tLoss {loss.last_avg:.4f}\tAdv {loss_adv.last_avg:.4f}'
                      '\tPrec@1 {top1.last_avg:.3%}\tPrec@5 {top5.last_avg:.3%}'.format(epoch, i + 1, len(train_loader),
                                                                                        batch_time=batch_time,
                                                                                        loss=losses,
                                                                                        loss_adv=losses_adv,
                                                                                        top1=top1, top5=top5))
            else:
                print('Epoch: [{0:>2d}][{1:>3d}/{2:>3d}] Time {batch_time.last_avg:.3f}\tLoss {loss.last_avg:.4f}'
                      '\tPrec@1 {top1.last_avg:.3%}\tPrec@5 {top5.last_avg:.3%}'.format(epoch, i + 1, len(train_loader),
                                                                                        batch_time=batch_time,
                                                                                        loss=losses,
                                                                                        top1=top1, top5=top5))

            if callback:
                if args.adv and epoch >= args.start_adv_epoch:
                    callback.scalars(['train_loss', 'adv_loss'], i / length + epoch,
                                     [losses.last_avg, losses_adv.last_avg])
                    callback.scalars(['train_prec@1', 'train_prec@5', 'adv_acc'], i / length + epoch,
                                     [top1.last_avg * 100, top5.last_avg * 100, adv_acc.last_avg * 100])
                    callback.scalar('adv_l2', i / length + epoch, l2_adv.last_avg)

                else:
                    callback.scalar('train_loss', i / length + epoch, losses.last_avg)
                    callback.scalars(['train_prec@1', 'train_prec@5'], i / length + epoch,
                                     [top1.last_avg * 100, top5.last_avg * 100])


def validate(val_loader, model, criterion, device, epoch, callback=None):
    model.eval()
    cudnn.benchmark = False

    batch_time = utils.AverageMeter()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        end = time.time()
        for i, (data, labels) in enumerate(val_loader):
            labels = labels.to(device, non_blocking=True)
            data = data.to(device)

            # compute output
            output = model(data)
            all_logits.append(output)
            all_labels.append(labels)

            batch_time.append(time.time() - end)
            end = time.time()

        all_logits = torch.cat(all_logits, 0)
        all_labels = torch.cat(all_labels, 0)
        # measure accuracy and record loss for clean samples
        loss = criterion(output, labels).item()
        prec1, prec5 = utils.accuracy(all_logits, all_labels, topk=(1, 5))

    print('Val | Time {:.3f}\tLoss {:.4f} | Clean: Prec@1 {:.3%}\tPrec@5 {:.3%}'.format(batch_time.avg, loss,
                                                                                        prec1, prec5))
    if callback:
        callback.scalar('val_loss', epoch, loss)
        callback.scalars(['val_prec@1', 'val_prec@5'], epoch, [prec1, prec5])

    return prec1


if __name__ == '__main__':
    main()
