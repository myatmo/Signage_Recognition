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
from loss import RecognitionLoss
import tqdm
import copy
from fots import FOTSModel
from utils.util import to_cuda_tensors
from utils.bbox import restore_bbox
from modules.parse_polys import parse_polys
from utils.data import prepare_image
import os
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def construct_bbox(model, img_dir, output_dir):
    height = width = 640
    pbar = tqdm.tqdm(os.listdir(img_dir), desc='Test', ncols=80)
    for image_name in pbar:
        prefix = image_name[:image_name.rfind('.')]
        '''
        image = cv2.imread(os.path.join(img_dir, image_name), cv2.IMREAD_COLOR)
        image, rh, rw = prepare_image(image, height, width)
        scaled_image = image[:, :, ::-1].astype(np.float32)
        scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()
        confidence, distances, angle = model(image_tensor.cuda())
        polys = parse_polys(image_name, image, output_dir, confidence, distances, angle, 0.85, 0.3)
        '''


def fit(model, dt_model, shared_features, train_dl, valid_dl, optimizer, criterion, tokenizer, n_epochs=1):
    for epoch in range(n_epochs):
        print("Start training\n")
        train_loss_stats = 0.0
        # fancy visualizer for training process
        pbar = tqdm.tqdm(train_dl, 'Epoch ' + str(epoch), ncols=80)
        # iterate over data
        for batch_id, data in enumerate(pbar):
            # get ground truth labels and move to gpu
            data = to_cuda_tensors(data)
            # get ground truth labels
            img_id, image, bboxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx = data
            # image = image.to('cuda')    # move image to gpu


            #with torch.no_grad():
                #score_map_pred, geo_map_pred, angle_map_pred = dt_model(image)
                #boxes, bbox_to_img_idx = restore_bbox(score_map_pred, geo_map_pred, angle_map_pred)

                #polys = parse_polys(img_id, None, None, score_map_pred, geo_map_pred, angle_map_pred, 0.85, 0.3)
            #print(score_map_pred.shape, type(score_map_pred), type(polys), polys.shape)
            indexed_tokens_true, seq_lens_true = tokenizer.encode(texts)
            # zero the gradients
            optimizer.zero_grad()
            score_map_pred, geo_map_pred, angle_map_pred, bboxes_pred, bbox_to_img_idx_pred, log_probs, seq_lens_pred = model(image, bboxes, bbox_to_img_idx)
            _, _, loss = criterion.forward(score_map, geo_map, angle_map, score_map_pred, geo_map_pred, angle_map_pred,
                                     training_mask, log_probs, indexed_tokens_true, seq_lens_pred, seq_lens_true)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{train_loss_stats / len(train_dl):.5f}'}, refresh=False)
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
    root_train = '../datasets/TotalText/Trainsets'
    root_test = '../datasets/TotalText/Testsets'
    data_train = TotalText(root_train)
    data_val = TotalText(root_test)
    print(len(data_train), len(data_val))
    train_loader = DataLoader(data_train, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(data_val, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(len(train_loader), len(test_loader))
    '''
    print("Loading pretrained weights")
    dtb = 'trained/last_checkpoint.pt'
    detection_model = FOTSModel().to(torch.device('cuda'))
    # load the trained FOTS model
    checkpoint = torch.load(dtb)
    detection_model.load_state_dict(checkpoint)
    detection_model.eval().cuda()
    shared_features = detection_model.remove_artifacts
    confidence = detection_model.confidence
    #print(detection_model.state_dict().keys())
    with torch.no_grad():
        for para in shared_features.parameters():
            print(para.shape, type(para))
        for para in confidence.parameters():
            print("conf: ", para.shape, type(para))
    #print(shared_features.parameters())
    '''

    # define the recognition model
    model = SRModel(is_training=True)
    tokenizer = Tokenizer(modules.alphabet.CHARS)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = FOTSLoss()
    #criterion = RecognitionLoss()
    model = torch.nn.DataParallel(model)
    fit(model, None, None, train_loader, test_loader, optimizer, criterion, tokenizer)
