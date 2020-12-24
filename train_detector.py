import math
import cv2
import numpy as np
import os
import torch
import torch.utils.data
import tqdm
from model import FOTSModel
from torch.utils.data import DataLoader
from data_loaders.datasets import TotalText
from utils.data import collate_fn
from loss import DetectionLoss
import torch.optim as optim
from modules.parse_polys import parse_polys
from utils.data import prepare_image
from utils.util import to_cuda_tensors


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


def fit(start_epoch, model, optimizer, lr_scheduler, best_score, checkpoint_dir, train_dl, valid_dl, num_epochs=584):
    for epoch in range(start_epoch, num_epochs):
        print("Start training\n")
        train_loss_stats = 0.0
        # fancy visualizer for training process
        pbar = tqdm.tqdm(train_dl, 'Epoch ' + str(epoch), ncols=80)
        # iterate over data
        for batch_id, data in enumerate(pbar):
            # get ground truth labels and move to gpu
            data = to_cuda_tensors(data)
            img_id, image, boxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx = data
            #image = image.to('cuda')    # move image to gpu
            print(img_id)

            optimizer.zero_grad()
            score_map_pred, geo_map_pred, angle_map_pred = model(image)
            criterion = DetectionLoss()
            loss = criterion.forward(score_map, geo_map, angle_map, score_map_pred, geo_map_pred, angle_map_pred,
                                     training_mask)
            train_loss_stats += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{train_loss_stats/len(train_dl):.5f}'}, refresh=False)
        lr_scheduler.step(train_loss_stats)

        print("Start evaluating\n")
        # evaluate the model
        if valid_dl is not None:
            model.eval().cuda()
            with torch.no_grad():
                val_loss = 0.0
                val_loss_count = 0
                for batch_id, data in enumerate(valid_dl):
                    data = to_cuda_tensors(data)
                    img_id, image, boxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx = data
                    print(img_id)
                    score_map_pred, geo_map_pred, angle_map_pred = model(image)
                    dt_loss = DetectionLoss()
                    loss = dt_loss.forward(score_map, geo_map, angle_map, score_map_pred, geo_map_pred, angle_map_pred,
                                           training_mask)
                    val_loss += loss.item()
                    val_loss_count += len(image)
            val_loss /= val_loss_count
            #pbar.set_postfix({'Loss': f'{val_loss:.5f}'}, refresh=False)
            print('Epoch: {} \tLoss: {:.6f}'.format(epoch, val_loss))
        '''
        if best_score > val_loss:
            best_score = val_loss
            save_as_best = True
        else:
            save_as_best = False
        #save_checkpoint(epoch, model, optimizer, lr_scheduler, best_score, checkpoint_dir, save_as_best)
        '''
        # save the model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'last_checkpoint.pt'))


def train():
    print("Loading datasets...")
    # torch.cuda.empty_cache()    # clear the cache
    # define datasets root path and load data
    root_train = '../datasets/TotalText/Trainsets'
    root_test = '../datasets/TotalText/Testsets'
    data_train = TotalText(root_train)
    data_val = TotalText(root_test)
    print(len(data_train), len(data_val))
    train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(data_val, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(len(train_loader), len(test_loader))
    print("Loading pretrained weights")
    # define model parameters
    fots = 'epoch_582_checkpoint.pt'  # path to the pretrained model
    # load the state_dict for model
    (epoch, model, optimizer, lr_scheduler, best_score) = load_fots_model(fots)
    #best_model_wts = copy.deepcopy(model.state_dict())
    model = torch.nn.DataParallel(model)
    fit(epoch, model, optimizer, lr_scheduler, best_score, "trained", train_loader, test_loader)


def construct_bbox(model, img_dir, output_dir):
    height = width = 640
    pbar = tqdm.tqdm(os.listdir(img_dir), desc='Test', ncols=80)
    for image_name in pbar:
        prefix = image_name[:image_name.rfind('.')]
        image = cv2.imread(os.path.join(img_dir, image_name), cv2.IMREAD_COLOR)
        image, rh, rw = prepare_image(image, height, width)
        scaled_image = image[:, :, ::-1].astype(np.float32)
        scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()
        confidence, distances, angle = model(image_tensor.cuda())
        polys = parse_polys(image_name, image, output_dir, confidence, distances, angle, 0.85, 0.3)


def test():
    img_dir = '../datasets/smalltotal/Test'
    output_dir = '../datasets/tested'
    dtb = 'trained/last_checkpoint_trial2_3.pt'
    model = FOTSModel().to(torch.device('cuda'))
    # must activate DataParallel if the model is trained with this module
    model = torch.nn.DataParallel(model)
    # load the trained FOTS model
    checkpoint = torch.load(dtb)
    # print(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval().cuda()
    # print(model.state_dict().keys())
    with torch.no_grad():
        construct_bbox(model, img_dir, output_dir)


if __name__ == '__main__':
    # train()
    test()

