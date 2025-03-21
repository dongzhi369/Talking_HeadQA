# -*- coding: utf-8 -*-
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import VideoDataset_images_with_audio_features
from utils import performance_fit
from utils import L1RankLoss
import random
import swin_transformer
#import PVTv2
#import CSwin
from torchvision import transforms
import time

def set_rand_seed(seed=2000):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(config):
    print('Begin!')
    set_rand_seed(seed=3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.model_name == 'st':
        print('The current model is ' + config.model_name)
        model = swin_transformer.SwinTransformer().to(device)
        #model = PVTv2.pvt_v2_b2(pretrained=True).to(device)
        #model = CSwin.CSWin_64_24322_small_224(pretrained=True).to(device)
        model = model.to(device)
    if config.multi_gpu:
        # model = torch.nn.DataParallel(model, device_ids=[4])
        # device = torch.device('cuda:4')
        model = model.to(device)
    else:
        model = model.to(device)

    model_dict = model.state_dict()
    pretrained_weights = torch.load('swin_tiny_patch4_window7_224.pth')
    if 'model' in pretrained_weights:
        pretrained_weights = pretrained_weights['model']
    if not config.multi_gpu:
        pretrained_weights = {k.replace('module.', ''): v for k, v in pretrained_weights.items()}
    else:
        pretrained_weights = pretrained_weights
    #pretrained_weights = {f'module.{k}': v for k, v in pretrained_weights.items()}
    updated_model_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(updated_model_dict)
    model.load_state_dict(model_dict)
    print(f"Number of weights loaded: {len(updated_model_dict)}")

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    if config.loss_type == 'L1RankLoss':
        criterion = L1RankLoss()

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    videos_dir_train = 'Talking_head/images_train'
    videos_dir_val = 'Talking_head/images_test'
    feature_dir_train = 'Talking_head/train_slowfast_videos'
    feature_dir_val = 'Talking_head/test_slowfast_videos'
    datainfo_train = 'Talking_head/thqa_ntire_train.csv'
    audio_train = 'Talking_head/train_features_audio'
    audio_val = 'Talking_head/test_features_audio'
    transformations_train = transforms.Compose(
        [transforms.Resize(config.resize), transforms.RandomCrop(config.crop_size), transforms.ToTensor(), \
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize), transforms.CenterCrop(config.crop_size), transforms.ToTensor(), \
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = VideoDataset_images_with_audio_features(videos_dir_train, feature_dir_train, datainfo_train, audio_train,transformations_train,
                                                        'train',config.crop_size, 'SlowFast')
    testset = VideoDataset_images_with_audio_features(videos_dir_val, feature_dir_val, None,audio_val, transformations_test,
                                                       'test', config.crop_size, 'SlowFast')

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    # best_test_criterion = -1  # SROCC min
    # best_test = []

    best_loss = float('inf')
    best_predictions = []
    best_video_names = []
    print('Starting training:')

    for epoch in range(config.epochs):
        print(
            '-------------------------------------------------------------------------------------------------------------------')
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, feature_3D, feature_audio, mos, _) in enumerate(train_loader):

            video = video.to(device)
            feature_3D = feature_3D.to(device)
            feature_audio = feature_audio.to(device)
            labels = mos.to(device).float()

            outputs = model(video, feature_3D,feature_audio)
            optimizer.zero_grad()

            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                      (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                       avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))
        print(
            '-------------------------------------------------------------------------------------------------------------------')
        if avg_loss < best_loss:
            best_loss = avg_loss

            # do validation after each epoch
            with torch.no_grad():
                model.eval()
                predictions = []
                video_names = []
                for i, (video, feature_3D, feature_audio, _) in enumerate(test_loader):
                    video = video.to(device)
                    feature_3D = feature_3D.to(device)
                    feature_audio = feature_audio.to(device)
                    # label[i] = mos.item()
                    outputs = model(video, feature_3D, feature_audio)

                    predictions.append(outputs.item())
                    video_names.append(_[0])
                    # y_output[i] = outputs.item()

                best_predictions = predictions
                best_video_names = video_names

                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                save_model_name = os.path.join(config.ckpt_path,
                                               f'best_model_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pth')
                torch.save(model.state_dict(), save_model_name)
                print(f'Saved best model with training loss: {avg_loss:.4f}')

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

    with open('output_T8_ST3407_audio.txt', 'w') as f:
        for name, pred in zip(best_video_names, best_predictions):
            f.write(f'{name}.mp4,{pred:.1f}\n')
    print(
        '-------------------------------------------------------------------------------------------------------------------')
    print('Training completed.')
    print(f'Best training loss: {best_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str, default='st')
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int)
    parser.add_argument('--print_samples', type=int, default=900)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=520)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=50)
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--loss_type', type=str, default='L1RankLoss')

    config = parser.parse_args()
    main(config)