import math
import cv2
import numpy as np
import numpy.random as nprnd
import os
import torch
import torch.utils.data
import tqdm
from model import FOTSModel
from torch.utils.data import DataLoader
import copy
from data_loaders.datasets import TotalText
from utils.data import collate_fn
from loss import DetectionLoss
from modules.parse_polys import parse_polys


def load_fots_model(fots):
    '''
    Load FOTS pretrained model and get all the checkpoint variables
    '''
    # initialize the model; the model was saved in gpu mode thus required gpu mode to load
    model = FOTSModel().to(torch.device('cuda'))

    # define optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=32, verbose=True, threshold=0.05, threshold_mode='rel')

    # load the pre-trained FOTS model
    checkpoint = torch.load(fots)

    # load the state_dict for model
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    best_score = checkpoint['best_score']

    return (epoch, model, optimizer, lr_scheduler, best_score)


def save_checkpoint(epoch, model, optimizer, lr_scheduler, best_score, folder, save_as_best):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # if epoch > 60 and epoch % 6 == 0:
    if True:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_score': best_score  # not current score
        }, os.path.join(folder, 'epoch_{}_checkpoint.pt'.format(epoch)))

    if save_as_best:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_score': best_score  # not current score
        }, os.path.join(folder, 'best_checkpoint.pt'))
        print('Updated best_model')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_score': best_score  # not current score
    }, os.path.join(folder, 'last_checkpoint.pt'))


def to_cuda_tensors(data):
    # move all parameters to gpu
    device = 'cuda'
    img_id, image, boxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx = data
    image = image.to(device)
    score_map = score_map.to(device)
    geo_map = geo_map.to(device)
    angle_map = angle_map.to(device)
    training_mask = training_mask.to(device)
    #bbox_to_img_idx = bbox_to_img_idx.to(device)
    data = (img_id, image, boxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx)
    return data


def fit(start_epoch, model, optimizer, lr_scheduler, best_score, checkpoint_dir, train_dl, valid_dl, num_epochs=590):
    for epoch in range(start_epoch, num_epochs):
        train_loss_stats = 0.0
        # fancy visualizer for training process
        pbar = tqdm.tqdm(train_dl, 'Epoch ' + str(epoch), ncols=80)
        # iterate over data
        for batch_id, data in enumerate(pbar):
            # get ground truth labels and move to gpu
            data = to_cuda_tensors(data)
            img_id, image, boxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx = data
            #image = image.to('cuda')    # move image to gpu

            optimizer.zero_grad()
            score_map_pred, geo_map_pred, angle_map_pred = model(image)
            dt_loss = DetectionLoss()
            loss = dt_loss.forward(score_map, geo_map, angle_map, score_map_pred, geo_map_pred, angle_map_pred,
                                   training_mask)
            train_loss_stats += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{train_loss_stats/len(train_loader):.5f}'}, refresh=False)
        lr_scheduler.step(train_loss_stats)

        # evaluate the model
        if valid_dl is not None:
            model.eval().cuda()
            with torch.no_grad():
                val_loss = 0.0
                val_loss_count = 0
                for batch_id, data in enumerate(valid_dl):
                    data = to_cuda_tensors(data)
                    img_id, image, boxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx = data
                    score_map_pred, geo_map_pred, angle_map_pred = model(image)
                    dt_loss = DetectionLoss()
                    loss = dt_loss.forward(score_map, geo_map, angle_map, score_map_pred, geo_map_pred, angle_map_pred,
                                           training_mask)
                    val_loss += loss.item()
                    val_loss_count += len(image)
            val_loss /= val_loss_count

        if best_score > val_loss:
            best_score = val_loss
            save_as_best = True
        else:
            save_as_best = False
        save_checkpoint(epoch, model, optimizer, lr_scheduler, best_score, checkpoint_dir, save_as_best)


if __name__ == '__main__':
    print("Loading datasets...")
    # define datasets root path and load data
    root_train = '../datasets/TotalText/Trainsets'
    root_test = '../datasets/TotalText/Testsets'
    data_train = TotalText(root_train)
    data_val = TotalText(root_test)
    print(len(data_train), len(data_val))
    train_loader = DataLoader(data_train, batch_size=3, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(data_val, batch_size=3, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(len(train_loader), len(test_loader))
    print("Loading pretrained weights")
    # define model parameters
    fots = 'epoch_582_checkpoint.pt'  # path to the pretrained model
    # load the state_dict for model
    (epoch, pretrained_model, optimizer, lr_scheduler, best_score) = load_fots_model(fots)
    #best_model_wts = copy.deepcopy(pretrained_model.state_dict())
    shared_features = pretrained_model.remove_artifacts
    print("FOTS keys")
    print(pretrained_model.state_dict().keys())
    model = torch.nn.DataParallel(pretrained_model)
    fit(epoch, pretrained_model, optimizer, lr_scheduler, best_score, "trained", train_loader, test_loader)
