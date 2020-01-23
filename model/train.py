import time
import torch
import torch.nn.parallel
import glob
import os
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

def save_checkpoint(model, filename):
    print("Saving Checkpoint: " + filename)
    torch.save(model.state_dict(), filename + '.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    correct_triplets = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img_name, img, att_p, att_n) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        img = torch.autograd.Variable(img)
        att_p = torch.autograd.Variable(att_p)
        att_n = torch.autograd.Variable(att_n)

        # compute output
        img, att_p, att_n, correct = model(img, att_p, att_n)
        loss = criterion(img, att_p, att_n)

        # measure and record loss
        loss_meter.update(loss.data.item(), att_p.size()[0])
        correct_triplets.update(torch.sum(correct))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Correct Triplets {correct_triplets.val:.3f} ({correct_triplets.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, correct_triplets=correct_triplets))

    plot_data['train_loss'][plot_data['epoch']] = loss_meter.avg
    plot_data['train_correct_triplets'][plot_data['epoch']] = correct_triplets.avg


    return plot_data


def validate(val_loader, model, criterion, epoch, print_freq, plot_data):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    correct_triplets = AverageMeter()

    # switch to train mode
    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (img_name, img, att_p, att_n) in enumerate(val_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            img = torch.autograd.Variable(img)
            att_p = torch.autograd.Variable(att_p)
            att_n = torch.autograd.Variable(att_n)

            # compute output
            img, att_p, att_n, correct = model(img, att_p, att_n)
            loss = criterion(img, att_p, att_n)

            # measure and record loss
            loss_meter.update(loss.data.item(), att_p.size()[0])
            correct_triplets.update(torch.sum(correct))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % print_freq == 0:
                print('Validation: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Correct Triplets {correct_triplets.val:.3f} ({correct_triplets.avg:.3f})\t'
                      .format(
                       epoch, i, len(val_loader), batch_time=batch_time,
                       data_time=data_time, loss=loss_meter, correct_triplets=correct_triplets))

    plot_data['val_loss'][plot_data['epoch']] = loss_meter.avg
    plot_data['val_correct_triplets'][plot_data['epoch']] = correct_triplets.avg


    return plot_data



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count