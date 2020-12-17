import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SRModel
from data_loaders.datasets import TotalText
from data_loaders.datasets import SynthText
import modules.alphabet
from utils.data import collate_fn
from utils.tokenizer import Tokenizer
from loss import FOTSLoss
import copy
from fots import FOTSModel
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def fit(model, data_loader, optimizer, criterion, tokenizer, n_epochs=1):
    print("Start training...")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(n_epochs):
        for batch_id, data in enumerate(data_loader):
            # get ground truth labels
            img_id, image, boxes, texts, score_maps, geo_maps, angle_maps, training_masks, bbox_to_img_idx = data
            indexed_tokens_true, seq_lens_true = tokenizer.encode(texts)
            print(img_id, type(image), image.shape, type(boxes), boxes.shape, texts)
            #print(training_masks.shape, torch.sum(training_masks))
            # zero the gradients
            optimizer.zero_grad()
            #for phas in ['train', 'val']:
                #score_maps_pred, geo_maps_pred, angle_maps_pred, bboxes_pred, bbox_to_img_idx_pred, log_probs, seq_lens_pred = model(image, boxes, bbox_to_img_idx, pretrained)
            
            '''
            # compute the looss
            detect_loss, recog_loss, loss = criterion(
                score_maps, geo_maps, angle_maps,
                score_maps_pred, geo_maps_pred, angle_maps_pred, training_masks,
                log_probs, indexed_tokens_true, seq_lens_pred, seq_lens_true
            )
            # backprop and update weights
            loss.backward()
            optimizer.step()
            print(f"[{epoch+1}, {batch_id+1}] detection loss: {detect_loss:.4f}, recognition loss: {recog_loss:.4f}, total loss: {loss:.4f}")
            '''

if __name__ == "__main__":
    print("Loading datasets...")
    # define datasets root path and load data
    root = '../datasets'
    #synth_text = SynthText(root)
    total_text = TotalText(root)
    dataset = total_text
    print(len(dataset))
    tokenizer = Tokenizer(modules.alphabet.CHARS)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(len(train_loader))
    print("Loading pretrained weights")
    print(torch.cuda.device_count())
    model = SRModel(is_training=True).to(device)
    print("SR keys")
    print(model.state_dict().keys())
    '''
    for name, param in pretrained_model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)
    '''
    #criterion = FOTSLoss()
    #fit(model, train_loader, optimizer, criterion, tokenizer)