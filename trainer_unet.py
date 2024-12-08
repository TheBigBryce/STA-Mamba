import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import DiceLoss
from torchvision import transforms
# from utils import test_single_volume
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class KDloss(nn.Module):

    def __init__(self,lambda_x):
        super(KDloss,self).__init__()
        self.lambda_x = lambda_x

    def inter_fd(self,f_s, f_t):
        s_C, t_C, s_H, t_H = f_s.shape[1], f_t.shape[1], f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        
        idx_s = random.sample(range(s_C),min(s_C,t_C))
        idx_t = random.sample(range(t_C),min(s_C,t_C))

        #inter_fd_loss = F.mse_loss(f_s[:, 0:min(s_C,t_C), :, :], f_t[:, 0:min(s_C,t_C), :, :].detach())

        inter_fd_loss = F.mse_loss(f_s[:, idx_s, :, :], f_t[:, idx_t, :, :].detach())
        return inter_fd_loss 
    
    def intra_fd(self,f_s):
        sorted_s, indices_s = torch.sort(F.normalize(f_s, p=2, dim=(2,3)).mean([0, 2, 3]), dim=0, descending=True)
        f_s = torch.index_select(f_s, 1, indices_s)
        intra_fd_loss = F.mse_loss(f_s[:, 0:f_s.shape[1]//2, :, :], f_s[:, f_s.shape[1]//2: f_s.shape[1], :, :])
        return intra_fd_loss
    
    def forward(self,feature,feature_decoder,final_up):
        # f1 = feature[0][-1] # 
        # f2 = feature[1][-1]
        # f3 = feature[2][-1]
        # f4 = feature[3][-1] # lower feature 

        f1_0 = feature[0] # 
        f2_0 = feature[1]
        f3_0 = feature[2]
        f4_0 = feature[3] # lower feature 

        # f1_d = feature_decoder[0][-1] # 14 x 14
        # f2_d = feature_decoder[1][-1] # 28 x 28
        # f3_d = feature_decoder[2][-1] # 56 x 56

        f1_d_0 = feature_decoder[0] # 14 x 14
        f2_d_0 = feature_decoder[1] # 28 x 28
        f3_d_0 = feature_decoder[2] # 56 x 56

        #print(f3_d.shape)

        final_layer = final_up
        #print(final_layer.shape)


        # loss =  (self.intra_fd(f1)+self.intra_fd(f2)+self.intra_fd(f3)+self.intra_fd(f4))/4
        loss = (self.intra_fd(f1_0)+self.intra_fd(f2_0)+self.intra_fd(f3_0)+self.intra_fd(f4_0))/4
        loss += (self.intra_fd(f1_d_0)+self.intra_fd(f2_d_0)+self.intra_fd(f3_d_0))/3
        # loss += (self.intra_fd(f1_d)+self.intra_fd(f2_d)+self.intra_fd(f3_d))/3


        
        loss += (self.inter_fd(f1_d_0,final_layer)+self.inter_fd(f2_d_0,final_layer)+self.inter_fd(f3_d_0,final_layer)
                   +self.inter_fd(f1_0,final_layer)+self.inter_fd(f2_0,final_layer)+self.inter_fd(f3_0,final_layer)+self.inter_fd(f4_0,final_layer))/7

        
        
        loss = loss * self.lambda_x
        return loss 
    
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule



def trainer_synapse(args, model, snapshot_path):
    from block_dataset.dataset_synapse import Synapse_dataset, RandomGenerator,RandomGenerator_DINO,RandomGenerator_DINO_Deform
    from torchvision.transforms import functional as VF

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    
    # Define your transformations
    # train_transforms = transforms.Compose([
    #     RandomGenerator(output_size=[args.img_size, args.img_size]),
    #     transforms.RandomRotation(degrees=30),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.RandomResizedCrop(size=args.img_size, scale=(0.8, 1.0)),
    #     # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    #     transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
    #     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    #     # Add custom elastic transform if necessary
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    # ])

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size]), ]),
                                transform_dino=transforms.Compose(
                                   [RandomGenerator_DINO(output_size=[args.img_size, args.img_size])])) #,alpha = args.alpha,sigma=args.sigma
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    #teacher_model.eval()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # kd_loss = KDloss(lambda_x=args.lambda_x)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0005)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    # scheduler = cosine_scheduler(base_lr, 0, max_iterations, niter_per_ep = len(trainloader), warmup_epochs = 10, start_warmup_value = 0)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # momentum_schedule = cosine_scheduler()

    

    for epoch_num in iterator:
        # for i_batch, (sampled_batch,dino_batch) in enumerate(trainloader):
        loss_ce=0
        loss_dice=0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

          
      

            # outputs, kd_encorder,kd_decorder, final_up = model(image_batch)
            outputs = model(image_batch)
          

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_kd = kd_loss(kd_encorder,kd_decorder,final_up)

            # loss = 0.4 * loss_ce + 0.6 * loss_dice + loss_kd # + args.dino_weight*loss_dino
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the learning rate using the scheduler 
        
            # lr_ = optimizer.param_groups[0]['lr']
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/dice_loss', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_dino', loss_dino,iter_num)

            logging.info('iteration %d : loss : %f, loss_dice %f loss_ce: %f' % (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))
            iter_num += 1
            
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        print()
        print("epoch num: {} loss_ce: {}, loss_dice: {}".format(epoch_num, loss_ce, loss_dice))

        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def trainer_assd(args, model, snapshot_path):
    from block_dataset.dataset_synapse import ASSD_dataset, RandomGenerator,RandomGenerator_DINO,RandomGenerator_DINO_Deform
    from torchvision.transforms import functional as VF

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    
    # Define your transformations
    # train_transforms = transforms.Compose([
    #     RandomGenerator(output_size=[args.img_size, args.img_size]),
    #     transforms.RandomRotation(degrees=30),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.RandomResizedCrop(size=args.img_size, scale=(0.8, 1.0)),
    #     # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    #     transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
    #     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    #     # Add custom elastic transform if necessary
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    # ])

    db_train = ASSD_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size]), ]),
                                transform_dino=transforms.Compose(
                                   [RandomGenerator_DINO(output_size=[args.img_size, args.img_size])])) #,alpha = args.alpha,sigma=args.sigma
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    #teacher_model.eval()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # kd_loss = KDloss(lambda_x=args.lambda_x)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0005)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    # scheduler = cosine_scheduler(base_lr, 0, max_iterations, niter_per_ep = len(trainloader), warmup_epochs = 10, start_warmup_value = 0)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # momentum_schedule = cosine_scheduler()

    

    for epoch_num in iterator:
        # for i_batch, (sampled_batch,dino_batch) in enumerate(trainloader):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # outputs, kd_encorder,kd_decorder, final_up = model(image_batch)
            outputs = model(image_batch.unsqueeze(1).float())
          

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_kd = kd_loss(kd_encorder,kd_decorder,final_up)

            # loss = 0.4 * loss_ce + 0.6 * loss_dice + loss_kd # + args.dino_weight*loss_dino
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the learning rate using the scheduler 
        
            # lr_ = optimizer.param_groups[0]['lr']
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/dice_loss', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_dino', loss_dino,iter_num)

            logging.info('iteration %d : loss : %f, loss_dice %f loss_ce: %f' % (iter_num, loss.item(), loss_dice.item(), loss_ce.item()))
            iter_num += 1
            
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if epoch_num > 20:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"