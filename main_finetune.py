from random import shuffle
from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

import argparse
import time
import datetime
import os
from pathlib import Path
import model_ft

cudnn.benchmark = True

def get_args_parser():
    parser = argparse.ArgumentParser('Fine tune', add_help=False)

    parser.add_argument('--batch_size', default=640, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--eval', default=0, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--model', default='Resnet50', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--data_path', default='/DATA5_DB8/data/ILSVRC2012', type=str,help='dataset path')
    parser.add_argument('--log_dir', default='./log_dir',help='path where to tensorboard log')
    
    return parser
    

def train(model, teacher_model, criterion, data_loader, dataset_train_sizes,  optimizer, scheduler, device, log_writer, epoch, batch_size):
    model.train()
    teacher_model.eval()

    running_loss = 0.0
    running_corrects = 0
    kilo_loss = 0.0

    for i, (train_batch, labels_batch) in enumerate(data_loader):
        train_batch = train_batch.to(device)
        labels_batch = labels_batch.to(device)

        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        output_batch = model(train_batch)
        _, preds = torch.max(output_batch, 1)
        
        with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch)
        output_teacher_batch = output_teacher_batch.to(device)

        loss = criterion(output_batch, labels_batch, output_teacher_batch)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * train_batch.size(0)
        running_corrects += torch.sum(preds == labels_batch.data)
        kilo_loss += loss.item() * train_batch.size(0)

        if i % 10 == 9:
            print(f'{(i + 1) * batch_size}/{dataset_train_sizes}')

        if i % 50 == 49: 
            log_writer.add_scalar('epoch loss', kilo_loss / (50 * batch_size), epoch * (dataset_train_sizes // batch_size) + i + 1 )
            print(f'Loss: {kilo_loss / (50 * batch_size)}')
            kilo_loss = 0.0


    scheduler.step()

    epoch_loss = running_loss / dataset_train_sizes
    epoch_acc = running_corrects.double() / dataset_train_sizes
    log_writer.add_scalar('loss', epoch_loss, epoch)
    log_writer.add_scalar('train accuracy', epoch_acc, epoch)


def evaluate(data_loader, dataset_val_sizes, model, device):
    model.eval()

    #running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for i, (data_batch, labels_batch) in enumerate(data_loader):

            data_batch = data_batch.to(device)
            labels_batch = labels_batch.to(device)

            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            
            output_batch = model(data_batch)
            _, preds = torch.max(output_batch, 1)

            running_corrects += torch.sum(preds == labels_batch.data)
            
            if i % 10 == 9:
                print(f'{(i + 1) * 640}/{dataset_val_sizes}')
    
    epoch_acc = running_corrects.double() / dataset_val_sizes
    
    return epoch_acc 


def main(args):

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
    device = torch.device(args.device)

    # Data
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    
    ])

    print('Start to load data')
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)
    dataset_train_sizes = len(dataset_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    print(dataset_val)
    dataset_val_sizes = len(dataset_val)

    #sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        shuffle = False,
        batch_size=args.batch_size,
        num_workers=4,
    )
    print('Data loader finished')

    # tensorboard
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # Model
    print(f'Start to prepare {args.model} model')
    if args.model == 'Resnet50':
        model = model_ft.Resnet50S()
        teacher_model = model_ft.Resnet50T()
    elif args.model == 'Resnet101':
        model = model_ft.Resnet101S()
        teacher_model = model_ft.Resnet101T()
    else:
        print('Please choose one from Resnet50, Resnet101, ViT or DeiT!')
        return
    #model = nn.DataParallel(model)
    #teacher_model = nn.DataParallel(model)
    model.to(device)
    teacher_model.to(device)
    print(f'Model {args.model} preparation finished')

    # Load saved model, evaluate it and exit
    if args.eval == 1:
        print('Loading saved model')
        model.load_state_dict(torch.load('./log_dir/model_weights.pth'))
        
        print('Testing saved model')
        acc = 0.0
        acc = evaluate(data_loader_val, dataset_val_sizes, model, device)
        
        print(f"Accuracy of the student network on the {dataset_val_sizes} test images: {(100 * acc):.1f}%")

        return

    # Train and evaluate per epoch
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = model_ft.loss_fn_kd
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    current_acc = 0.0
    teacher_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f'The {epoch} epoch begins')

        # Evaluate model without train first to prove correctness
        current_acc = evaluate(data_loader_val, dataset_val_sizes, model, device)
        # Only once
        if epoch == 0:
            teacher_acc = evaluate(data_loader_val, dataset_val_sizes, teacher_model, device)

        if current_acc > best_acc:
            best_acc = current_acc
            if epoch > 0:
                torch.save(model.state_dict(), './log_dir/model_weights.pth')
        
        print(f"Accuracy of the student network on the {dataset_val_sizes} test images: {(100 * current_acc):.1f}%")
        print(f'Max accuracy: {(100 * best_acc):.2f}%')
        print(f"Accuracy of the teacher network on the {dataset_val_sizes} test images: {(100 * teacher_acc):.1f}%")
        
        log_writer.add_scalar('evaluate accuracy', current_acc, epoch)
        log_writer.add_scalar('best accuracy', best_acc, epoch)
        log_writer.add_scalar('teacher model accuracy', teacher_acc, epoch)

        train(model, teacher_model, criterion, data_loader_train,dataset_train_sizes, optimizer, scheduler, device, log_writer, epoch, args.batch_size)

    log_writer.flush()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
