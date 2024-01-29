import os
import sys
import json
import torch
import torch.nn as nn
# from main_inside import train_inside
from opts import parse_opts
from model import generate_model
from dataset import get_training_set_inside,get_training_set_outside, get_validation_set_inside, get_validation_set_outside
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformRandomSample, UniformEndSample, UniformIntervalCrop

from target_transforms import ClassLabel
from torch.autograd import Variable
from mean import get_mean, get_std
from models.convolution_lstm import encoder, classifier
from utils import AverageMeter, calculate_accuracy
import warnings
from tqdm import tqdm
#import matplotlib.pyplot as plt
from torch import optim
from torch.optim import lr_scheduler
import time
from utils import Logger
from torch.utils.tensorboard import SummaryWriter



# 경고 무시 설정
warnings.filterwarnings("ignore", category=UserWarning)

# 내부, 외부에서 추출한 벡터를 합쳐서 분류기에 넣어줌
class conv_classifier(nn.Module):
    def __init__(self):
        super(conv_classifier, self).__init__()
        
        # max pool함
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),           
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),           
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),   
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(512)     
                  
        )

        self.classifier_fc = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Softmax(dim=1) 
        )  


    def forward(self, inside, outside):
        out = self.conv_block(outside)
        out = out.view(out.size(0),-1)
        combined = torch.cat((inside, out), dim=1)

        x = self.classifier_fc(combined)
        return x
    



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)                   #nan발생시 검출
    writer = SummaryWriter()
    
    
    # inside 훈련 데이터 증강 및 데이터 로더 부분
    opt = parse_opts()

    
    opt.scales_inside = [opt.initial_scale]
    for i in range(1, opt.n_scales_inside):
        opt.scales_inside.append(opt.scales_inside[-1] * opt.scale_step)
    opt.arch_inside = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean_inside = get_mean(opt.norm_value_inside, dataset=opt.mean_dataset)
    opt.std_inside = get_std(opt.norm_value_inside)
    torch.manual_seed(opt.manual_seed)

    if opt.no_mean_norm and not opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean_inside, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean_inside, opt.std_inside)
          
    if not opt.no_train_inside:
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
        train_logger_inside = Logger(
            os.path.join(opt.result_path_inside, 'train_inside.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger_inside = Logger(
            os.path.join(opt.result_path_inside, 'train_inside_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        train_logger_classifier = Logger(
            os.path.join(opt.result_path_outside, 'train_classifier.log'),
            ['epoch', 'loss', 'acc'])
        
        
    # inside 검증 데이터 증강 및 데이터 로더 부분
    if not opt.no_val_inside:
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
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger_inside = Logger(
            os.path.join(opt.result_path_inside, 'val_inside.log'), ['epoch', 'loss', 'acc'])
        val_logger_classifier = Logger(
            os.path.join(opt.result_path_outside, 'val_classifier.log'), ['epoch', 'loss', 'acc'])
    
    

    # inside 모델 불러오기
    model_inside, parameters_inside = generate_model(opt)
    weights = [1, 2, 4, 2, 4]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion_inside = nn.CrossEntropyLoss(weight=class_weights)
    if not opt.no_cuda:
        criterion_inside = criterion_inside.cuda()
        
    if opt.nesterov_inside:
        dampening = 0
    else:
        dampening = opt.dampening
    
    optimizer_inside = optim.SGD(
        parameters_inside,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov_inside)
    
    scheduler_inside = lr_scheduler.MultiStepLR(
        optimizer_inside, milestones=opt.lr_step, gamma=0.1)
    
    if opt.resume_path_inside:
        print('loading checkpoint {}'.format(opt.resume_path_inside))
        checkpoint = torch.load(opt.resume_path_inside)
        assert opt.arch_inside == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model_inside.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train_inside:
            optimizer_inside.load_state_dict(checkpoint['optimizer'])
            
    
    
    
    # outside 훈련 데이터 증강 및 데이터 로더 부분
    opt.scales_outside = [opt.initial_scale]
    for i in range(1, opt.n_scales_outside):
        opt.scales_outside.append(opt.scales_outside[-1] * opt.scale_step)
    opt.arch_outside = 'ConvLSTM'
    opt.mean_outside = get_mean(opt.norm_value_outside, dataset=opt.mean_dataset)
    opt.std_outside = get_std(opt.norm_value_outside)
    
    if opt.no_mean_norm and not opt.std_norm:
            norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean_outside, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean_outside, opt.std_outside)
        
    if not opt.no_train_outside:    
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
    train_logger_outside = Logger(
		os.path.join(opt.result_path_outside, 'train_outside.log'),
		['epoch', 'loss', 'lr'])
    train_batch_logger_outside = Logger(
        os.path.join(opt.result_path_outside, 'train_outside_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'lr'])
    
    
    # outside 검증 데이터 증강 및 데이터 로더 부분
    if not opt.no_val_outside:
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
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger_outside = Logger(
            os.path.join(opt.result_path_outside, 'val_outside.log'), ['epoch', 'loss'])
    
    # outside 모델 불러오기
    model_outside = encoder(hidden_channels=[128, 64, 64, 32], sample_size=opt.sample_size, sample_duration=opt.sample_duration_outside).cuda()
	
    model_outside = nn.DataParallel(model_outside, device_ids=None)
    parameters_outside = model_outside.parameters()
    
    criterion_outside = nn.MSELoss()
    if not opt.no_cuda:
        criterion_outside = criterion_outside.cuda()
        
    if opt.nesterov_outside:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer_outside = optim.SGD(
		parameters_outside,
		lr=opt.learning_rate,
		momentum=opt.momentum,
		dampening=dampening,
		weight_decay=opt.weight_decay,
		nesterov=opt.nesterov_outside)
    scheduler_outside = lr_scheduler.MultiStepLR(
        optimizer_outside, milestones=opt.lr_step, gamma=0.1)
    
    
    if opt.resume_path_outside:
        print('loading checkpoint {}'.format(opt.resume_path_outside))
        checkpoint = torch.load(opt.resume_path_outside)
        assert opt.arch_outside == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model_outside.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train_outside:
            optimizer_outside.load_state_dict(checkpoint['optimizer'])

    My_Conv_classifier = conv_classifier().to(device)
    
    weights = [1, 2, 4, 2, 4]
    print(torch.cuda.is_available())
    class_weights = torch.FloatTensor(weights).cuda()
    criterion_classifier = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_classifier = torch.optim.Adam(My_Conv_classifier.parameters(), lr=0.001)
    
    if not opt.no_cuda:
        criterion = criterion_classifier.cuda()
        
    global best_prec_inside
    global best_loss_outside
    global best_prec_classifier
    best_prec_inside = 0
    best_loss_outside = 100
    best_prec_classifier = 0
    
    for epoch in range(opt.n_epochs + 1):  
        avg_acc = []
        avg_loss = []          
        
        if not opt.no_train_inside and not opt.no_train_outside:
            # print('train_inside at epoch {}'.format(epoch))
            
            losses_inside_train = AverageMeter()
            accuracies_inside_train = AverageMeter()
            
            # print('train_outside at epoch {}'.format(epoch))
            
            losses_outside_train = AverageMeter()
            
            # print('train_classifier at epoch {}'.format(epoch))
            
            
            data_loader_train = tqdm(zip(train_loader, train_loader_outside), total=len(train_loader), desc = "Training")
            # inside, outside 훈련 과정 실행
            for i, ((inputs_in, targets_in), (inputs_out, targets_out)) in enumerate(data_loader_train):
                model_inside.train()
                # inside 훈련 과정 실행
                if not opt.no_cuda:
                    targets = targets_in.cuda(non_blocking=True)
                inputs_in = Variable(inputs_in)
                targets_in = Variable(targets_in)
                targets_in = targets_in.to(device) # inputs_in은 GPU인데 targets_in은 CPU라 오류떠서 해줘야돼
                
                outputs_in, outputs_in_not_fc = model_inside(inputs_in)
                loss_in = criterion_inside(outputs_in, targets_in)
                acc_in = calculate_accuracy(outputs_in, targets_in)
                
                model_outside.train()
                # outside 훈련 과정 실행
                if not opt.no_cuda:
                    targets_out = targets_out.cuda(non_blocking=True)
                inputs_out = Variable(inputs_out)
                targets_out = Variable(targets_out)
                targets_out = targets_out.to(device)
                
                outputs_out = model_outside(inputs_out)
                loss_out = criterion_outside(outputs_out, targets_out)
                
                My_Conv_classifier.train()
                # classifier 훈련 과정 실행
                output = My_Conv_classifier(outputs_in_not_fc, outputs_out)
                loss_classifier = criterion_classifier(output, targets_in)
                acc = calculate_accuracy(output, targets_in)
                avg_acc.append(acc)
                avg_loss.append(loss_classifier)
                
                # loss update
                losses_inside_train.update(loss_in.item(), inputs_in.size(0))
                losses_outside_train.update(loss_out.item(), inputs_out.size(0))
                
                accuracies_inside_train.update(acc_in, inputs_in.size(0))
                
                # optimizer update
                optimizer_inside.zero_grad()
                optimizer_outside.zero_grad()
                optimizer_classifier.zero_grad()
                
                loss_in.backward(retain_graph=True)
                loss_out.backward(retain_graph=True)
                loss_classifier.backward()
                
                optimizer_inside.step()
                optimizer_outside.step()
                optimizer_classifier.step()
               
                writer.add_scalar('Training Loss Inside', losses_inside_train.avg, epoch)
                writer.add_scalar('Training Accuracy Inside', accuracies_inside_train.avg, epoch)
                
                writer.add_scalar('Training Loss Outside', losses_outside_train.avg, epoch)

                

                
                data_loader_train.set_postfix(loss=loss_classifier.item(), acc=acc)
                
                # logger update
                train_batch_logger_inside.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(train_loader) + (i + 1),
                    'loss': losses_inside_train.val,
                    'acc': accuracies_inside_train.val,
                    'lr': optimizer_inside.param_groups[0]['lr']
                })
                train_batch_logger_outside.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(train_loader_outside) + (i + 1),
                    'loss': losses_outside_train.val,
                    'lr': optimizer_outside.param_groups[0]['lr']
                })
                
                
            print('Epoch_inside: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            epoch,
                            i + 1,
                            len(train_loader),
                            loss=losses_inside_train,
                            acc=accuracies_inside_train))
            print('Epoch_outside: [{0}][{1}/{2}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                                epoch,
                                i + 1,
                                len(train_loader_outside),
                                loss=losses_outside_train))
            avg_acc_value_train = sum(avg_acc) / len(avg_acc)
            avg_loss_value_train = sum(avg_loss) / len(avg_loss)
            print(f"Epoch_classifier [{epoch}/{opt.n_epochs}], Loss: {avg_loss_value_train}, avg_acc: {avg_acc_value_train}")
            
            train_logger_inside.log({
                'epoch': epoch,
                'loss': losses_inside_train.avg,
                'acc': accuracies_inside_train.avg,
                'lr': optimizer_inside.param_groups[0]['lr']
            })
            train_logger_classifier.log({
                    'epoch': epoch,
                    'loss': avg_loss_value_train,
                    'acc': avg_acc_value_train
                })

            if epoch % opt.checkpoint == 0:
                save_file_path_inside = os.path.join(opt.result_path_inside,
                                            'save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch_inside,
                    'state_dict': model_inside.state_dict(),
                    'optimizer': optimizer_inside.state_dict(),
                }
                
            train_logger_outside.log({
                'epoch': epoch,
                'loss': losses_outside_train.avg,
                'lr': optimizer_outside.param_groups[0]['lr']
            })

            if epoch % opt.checkpoint == 0:
                save_file_path_outside = os.path.join(opt.result_path_outside,
                                                'convlstm-save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch_outside,
                    'state_dict': model_outside.state_dict(),
                    'optimizer': optimizer_outside.state_dict(),
                }  
            writer.add_scalar('Training Loss Classifier', avg_acc_value_train, epoch)
            writer.add_scalar('Training Accuracy Classifier', avg_loss_value_train, epoch)
        ################################################################################################################
        avg_acc_val = []
        avg_loss_val = []
        if not opt.no_val_inside and not opt.no_val_outside:
            # print('validation_inside at epoch {}'.format(epoch))
            model_inside.eval()
            losses_inside = AverageMeter()
            accuracies_inside = AverageMeter()
            
            # print('validation_outside at epoch {}'.format(epoch))
            model_outside.eval()
            losses_outside = AverageMeter()
            
            # print('validation_classifier at epoch {}'.format(epoch))
            My_Conv_classifier.eval()
            
            data_loader_val = tqdm(zip(val_loader, val_loader_outside), total=len(val_loader), desc = "Validation")
            # inside, outside 검증 과정 실행
            with torch.no_grad():
                for i, ((inputs_in, targets_in), (inputs_out, targets_out)) in enumerate(data_loader_val):
                    # inside 검증 과정 실행
                    if not opt.no_cuda:
                        targets = targets_in.cuda(non_blocking=True)
                    inputs_in = inputs_in.to(device)
                    targets_in = targets_in.to(device) # inputs_in은 GPU인데 targets_in은 CPU라 오류떠서 해줘야돼
                    
                    outputs_in, outputs_in_not_fc = model_inside(inputs_in)
                    loss_in = criterion_inside(outputs_in, targets_in)
                    acc_in = calculate_accuracy(outputs_in, targets_in)
                    
                    # outside 검증 과정 실행
                    if not opt.no_cuda:
                        targets_out = targets_out.cuda(non_blocking=True)
                    inputs_out = inputs_out.to(device)
                    targets_out = targets_out.to(device)
                    
                    outputs_out = model_outside(inputs_out)
                    loss_out = criterion_outside(outputs_out, targets_out)
                    
                    # classifier 검증 과정 실행
                    output = My_Conv_classifier(outputs_in_not_fc, outputs_out)
                    loss_classifier = criterion_classifier(output, targets_in)
                    acc = calculate_accuracy(output, targets_in)
                    avg_acc_val.append(acc)
                    avg_loss_val.append(loss_classifier)
                    
                    # loss update
                    losses_inside.update(loss_in.item(), inputs_in.size(0))
                    losses_outside.update(loss_out.item(), inputs_out.size(0))
                    accuracies_inside.update(acc_in, inputs_in.size(0))

                    writer.add_scalar('Validation Loss Inside', losses_inside.avg, epoch)
                    writer.add_scalar('Validation Accuracy Inside', accuracies_inside.avg, epoch)
                    writer.add_scalar('Validation Loss Outside', losses_outside.avg, epoch)
                    
                    data_loader_val.set_postfix(loss=loss_classifier.item(), acc=acc)
                
                
            print('Epoch_inside: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            epoch,
                            i + 1,
                            len(val_loader),
                            loss=losses_inside,
                            acc=accuracies_inside))
            print('Epoch_outside: [{0}][{1}/{2}]\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                                epoch,
                                i + 1,
                                len(val_loader_outside),
                                loss=losses_outside))
            
            avg_acc_value = sum(avg_acc_val) / len(avg_acc_val)
            avg_loss_value = sum(avg_loss_val) / len(avg_loss_val)
            print(f"Epoch_classifier [{epoch}/{opt.n_epochs}], Loss: {avg_loss_value}, avg_acc: {avg_acc_value}")
            
            val_logger_inside.log({
                'epoch': epoch,
                'loss': losses_inside.avg,
                'acc': accuracies_inside.avg,
            })

            val_logger_classifier.log({
                        'epoch': epoch,
                        'loss': avg_loss_value,
                        'acc': avg_acc_value,
                    })

            val_logger_outside.log({
                'epoch': epoch,
                'loss': losses_outside.avg
                })
            writer.add_scalar('Training Loss Inside', avg_acc_value, epoch)
            writer.add_scalar('Training Accuracy Inside', avg_loss_value, epoch)

            is_best_inside = accuracies_inside.avg > best_prec_inside
            best_prec_inside = max(accuracies_inside.avg, best_prec_inside)
            if is_best_inside:
                print('\n The best inside prec is %.4f' % best_prec_inside)
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch_inside,
                    'state_dict': model_inside.state_dict(),
                    'optimizer': optimizer_inside.state_dict(),
                }
                save_file_path_inside = os.path.join(opt.result_path_inside,
                                    'save_best_inside.pth')
                torch.save(states, save_file_path_inside)
                
            
            

            is_best_outside = losses_outside.avg < best_loss_outside
            best_loss_outside = min(losses_outside.avg, best_loss_outside)
            
            if is_best_outside:
                print('\n The best outside loss is %.4f' % best_loss_outside)
                states = {
                    'epoch': epoch + 1,
                    'arch': opt.arch_outside,
                    'state_dict': model_outside.state_dict(),
                    'optimizer': optimizer_outside.state_dict(),
                }
                save_file_path_outside = os.path.join(opt.result_path_outside,
                                    'save_best_outside.pth')
                torch.save(states, save_file_path_outside)
                
            
            is_best_classifier = avg_acc_value > best_prec_classifier
            best_prec_classifier = max(avg_acc_value, best_prec_classifier)
            
            if is_best_classifier:
                print('\n The best classifier prec is %.4f' % best_prec_classifier)
                states = {
                    'epoch': epoch + 1,
                    'state_dict': My_Conv_classifier.state_dict(),
                    'optimizer': optimizer_classifier.state_dict(),
                }
                save_file_path_classifier = os.path.join(opt.result_path_outside,
                                    'save_best_classifier.pth')
                torch.save(states, save_file_path_classifier)    
        if not opt.no_train_inside and not opt.no_train_outside and not opt.no_val_inside and not opt.no_val_outside:
            scheduler_inside.step()
            scheduler_outside.step() 
    
    print(f'Test Accuracy: {best_prec_classifier:.4f}')
            

######################################################################################################################
    
    def load_checkpoint(model, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(filename))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
        return model


    writer.close()