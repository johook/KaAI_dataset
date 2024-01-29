import os
import sys
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as FF
from torch import optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms.functional as F
from math import log10

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformRandomSample, UniformEndSample, UniformIntervalCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set_inside,get_training_set_outside, get_validation_set_inside, get_validation_set_outside
from utils import Logger
from models.convolution_lstm import encoder, classifier
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy
import pytorch_ssim

if __name__ == '__main__':


    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path_inside = os.path.join(opt.root_path, opt.video_path_inside)
        opt.video_path_outside = os.path.join(opt.root_path, opt.video_path_outside)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path_inside = os.path.join(opt.root_path, opt.result_path_inside)
        opt.result_path_outside = os.path.join(opt.root_path, opt.result_path_outside)
        if opt.resume_path_inside:
            opt.resume_path_inside = os.path.join(opt.root_path, opt.resume_path_inside)
        if opt.resume_path_outside:
            opt.resume_path_outside = os.path.join(opt.root_path, opt.resume_path_outside)
            
        if opt.pretrain_path_inside:
            opt.pretrain_path_inside = os.path.join(opt.root_path, opt.pretrain_path_inside)
        if opt.pretrain_path_outside:
            opt.pretrain_path_outside = os.path.join(opt.root_path, opt.pretrain_path_outside)
    opt.scales_inside = [opt.initial_scale]
    for i in range(1, opt.n_scales_inside):
        opt.scales_inside.append(opt.scales_inside[-1] * opt.scale_step)
        
    opt.scales_outside = [opt.initial_scale]
    for i in range(1, opt.n_scales_outside):
        opt.scales_outside.append(opt.scales_outside[-1] * opt.scale_step)
        
    opt.arch_inside = '{}-{}'.format(opt.model, opt.model_depth)
    opt.arch_outside = 'ConvLSTM'
    
    opt.mean_inside = get_mean(opt.norm_value_inside, dataset=opt.mean_dataset)
    opt.mean_outside = get_mean(opt.norm_value_outside, dataset=opt.mean_dataset)
    
    opt.std_inside = get_std(opt.norm_value_inside)
    opt.std_outside = get_std(opt.norm_value_outside)
    print(opt)
    
    with open(os.path.join(opt.result_path_inside, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    weights = [1, 2, 4, 2, 4]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if not opt.no_cuda:
        criterion = criterion.cuda()
        
    model_outside = encoder(hidden_channels=[128, 64, 64, 32], sample_size=opt.sample_size, sample_duration=opt.sample_duration_outside).cuda()
    model_outside = nn.DataParallel(model_outside, device_ids=None)
    parameters_outside = model_outside.parameters()
    
    model_classifier = classifier().cuda()
    # print(model)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        if opt.model == 'resnet':
            norm_method = Normalize(opt.mean_inside, [1, 1, 1])
        else:
            norm_method = Normalize(opt.mean_outside, [1, 1, 1])
    else:
        if opt.model == 'resnet':
            norm_method = Normalize(opt.mean_inside, opt.std_inside)
        else:
            norm_method = Normalize(opt.mean_outside, opt.std_outside)
        
        
        
        
        
    #data 전처리(증강) -> 여기서는 driver focus를 사용하여 운전자 자리를 좀 더 집중하여 crop 해줌
    #inside부분
    if not opt.no_train: 
        assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales_inside, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales_inside, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales_inside, opt.sample_size, crop_positions=['c'])
        elif opt.train_crop == 'driver focus':
            crop_method = DriverFocusCrop(opt.scales_inside, opt.sample_size)
        train_spatial_transform = Compose([
            crop_method,
            MultiScaleRandomCrop(opt.scales_inside, opt.sample_size),
            ToTensor(opt.norm_value_inside), norm_method
        ])
        # 랜덤하게 프레임을 잘라서 시간적으로도 crop하여 시간적인 데이터 증강을한다.
        train_temporal_transform = UniformRandomSample(opt.sample_duration_inside, opt.end_second)
        train_target_transform = ClassLabel()
        train_horizontal_flip = RandomHorizontalFlip()
        training_data_inside = get_training_set_inside(opt, train_spatial_transform, train_horizontal_flip,
                                         train_temporal_transform, train_target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data_inside,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path_inside, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path_inside, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        
    
    #data 전처리(증강) -> 여기서는 driver focus를 사용하여 운전자 자리를 좀 더 집중하여 crop 해줌
    #outside부분
    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales_outside, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales_outside, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales_outside, opt.sample_size, crop_positions=['c'])
    elif opt.train_crop == 'driver focus':
        crop_method = DriverFocusCrop(opt.scales_outside, opt.sample_size)
    train_spatial_transform = Compose([
        Scale(opt.sample_size),		
        ToTensor(opt.norm_value_outside) #, norm_method
    ])
    train_temporal_transform = UniformIntervalCrop(opt.sample_duration_outside, opt.interval)
    train_target_transform = Compose([
        Scale(opt.sample_size),
        ToTensor(opt.norm_value_outside)#, norm_method
    ])
    train_horizontal_flip = RandomHorizontalFlip()
    training_data_outside = get_training_set_outside(opt, train_spatial_transform, train_horizontal_flip,
                                        train_temporal_transform, train_target_transform)
    train_loader_outside = torch.utils.data.DataLoader(
            training_data_outside,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
    train_logger = Logger(
        os.path.join(opt.result_path_outside, 'convlstm-train.log'),
        ['epoch', 'loss', 'lr'])
    train_batch_logger = Logger(
        os.path.join(opt.result_path_outside, 'convlstm-train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'lr'])
        
        

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=opt.lr_step, gamma=0.1)
    
    
    
    
    #inside부분
    if not opt.no_val:
        val_spatial_transform = Compose([
            DriverCenterCrop(opt.scales_inside, opt.sample_size),
            ToTensor(opt.norm_value_inside), norm_method
        ])
        val_temporal_transform = UniformEndSample(opt.sample_duration_inside, opt.end_second)
        val_target_transform = ClassLabel()
        validation_data = get_validation_set_inside(
            opt, val_spatial_transform, val_temporal_transform, val_target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=24,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path_inside, 'val.log'), ['epoch', 'loss', 'acc'])
    
    #outside부분
    if not opt.no_val:
        val_spatial_transform = Compose([
			Scale(opt.sample_size),
			ToTensor(opt.norm_value_outside)#, norm_method
		])
        val_temporal_transform = UniformIntervalCrop(opt.sample_duration_outside, opt.interval)
        val_target_transform = val_spatial_transform
        validation_data = get_validation_set_outside(
		    opt, val_spatial_transform, val_temporal_transform, val_target_transform)
        val_loader_outside = torch.utils.data.DataLoader(
			validation_data,
			batch_size=1,
			shuffle=True,
			num_workers=opt.n_threads,
			pin_memory=True)
        val_logger = Logger(
			os.path.join(opt.result_path_outside, 'convlstm-val.log'), ['epoch', 'loss', 'ssim', 'psnr'])




    if opt.resume_path_inside:
        print('loading checkpoint {}'.format(opt.resume_path_inside))
        checkpoint = torch.load(opt.resume_path_inside)
        assert opt.arch_inside == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
    if opt.resume_path_outside:
        print('loading checkpoint {}'.format(opt.resume_path_outside))
        checkpoint = torch.load(opt.resume_path_outside)
        assert opt.arch_outside == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
#===============================================================================================
           

    print('run')
    global best_prec
    global best_loss
    best_prec = 0
    best_loss = torch.tensor(float('inf'))
    
    
    
    
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):

        if not opt.no_train:
            print('train at epoch {}'.format(epoch))

            model.train()
            model_outside.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            
            batch_time_outside = AverageMeter()
            data_time_outside = AverageMeter()
            losses_outside = AverageMeter()

            end_time = time.time()
            
            for i, ((inputs_in, targets_in),(inputs_out, targets_out)) in enumerate(zip(train_loader,train_loader_outside)):
                data_time.update(time.time() - end_time)

                if not opt.no_cuda:
                    targets_in = targets_in.cuda(non_blocking=True)
                inputs_in = Variable(inputs_in)
                targets_in = Variable(targets_in)
                outputs_in = model(inputs_in)
                
                if not opt.no_cuda:
                    targets_out = targets_out.cuda(non_blocking=True)
                inputs_out = Variable(inputs_out)
                targets_out = Variable(targets_out)
                outputs_out = model_outside(inputs_out)
                
                outputs = torch.cat((outputs_in, outputs_out), 1)
                
                outputs = model_classifier(outputs)
                
                loss = criterion(outputs, targets_in)
                acc = calculate_accuracy(outputs, targets_in)

#####여기 호로록 막함
                losses.update(loss.item(), inputs_in.size(0))
                accuracies.update(acc, inputs_in.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                train_batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(train_loader) + (i + 1),
                    'loss': losses.val,
                    'acc': accuracies.val,
                    'lr': optimizer.param_groups[0]['lr']
                })
                if i % 5 == 0:
                  print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                          epoch,
                          i + 1,
                          len(train_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))

            train_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

            if epoch % opt.checkpoint == 0:
                save_file_path = os.path.join(opt.result_path_inside,
                                              'save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch_inside,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

        if not opt.no_val:
            print('Validation at epoch {}'.format(epoch))

            model.eval()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()

            end_time = time.time()

            conf_mat = torch.zeros(opt.n_finetune_classes, opt.n_finetune_classes)
            output_file = []

            for i, ((inputs_in, targets_in),(inputs_out, targets_out)) in enumerate(zip(val_loader,val_loader_outside)):
                data_time.update(time.time() - end_time)
                
                
                if not opt.no_cuda:
                    targets_in = targets_in.cuda(non_blocking=True)
                inputs_in = Variable(inputs_in)
                targets_in = Variable(targets_in)
                outputs_in = model(inputs_in)
                
                if not opt.no_cuda:
                    targets_out = targets_out.cuda(non_blocking=True)
                inputs_out = Variable(inputs_out)
                targets_out = Variable(targets_out)
                outputs_out = model_outside(inputs_out)
                
                outputs = torch.cat((outputs_in, outputs_out), 1)
                
                outputs = model_classifier(outputs)
                
                loss = criterion(outputs, targets_in)
                acc = calculate_accuracy(outputs, targets_in)

                ### print out the confusion matrix
                _,pred = torch.max(outputs,1)
                for t,p in zip(targets_in.view(-1), pred.view(-1)):
                    conf_mat[t,p] += 1

                losses.update(loss.item(), inputs_in.size(0))
                accuracies.update(acc, inputs_in.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                          epoch,
                          i + 1,
                          len(val_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))
            print(conf_mat)

            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

            is_best = accuracies.avg > best_prec
            best_prec = max(accuracies.avg, best_prec)
            print('\n The best prec is %.4f' % best_prec)
            if is_best:
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch_inside,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                 }
                save_file_path = os.path.join(opt.result_path_inside,
                                    'save_best.pth')
                torch.save(states, save_file_path)

        if not opt.no_train and not opt.no_val:
            scheduler.step()



