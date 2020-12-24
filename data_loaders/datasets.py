import os
import re

import cv2
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data import collate_fn
from utils.data import aug, sort_points_clockwise, generate_rbox, prepare_image

class TotalText(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'data')
        self.gt_dir = os.path.join(self.root_dir, 'gt')
        self.img_filenames = [file.name.split('.')[0] for file in os.scandir(self.img_dir)] # keep the prefix, drop the file extension
        self.height = 640
        self.width = 640
        self.transform = transforms.Compose([
            transforms.ToTensor(), # this modifies the shape of img from H x W x C to C x H x W
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ) # from ImageNet
        ])

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, f'{img_filename}.jpg')
        image = cv2.imread(img_path)
        img, rh, rw = prepare_image(image, self.height, self.width)
        bboxes, texts = self._load_annotations(img_filename)
        scale = np.array([[rw, rh]])
        bboxes = (bboxes * scale).astype(bboxes.dtype)
        score_map, geo_map, angle_map, training_mask = generate_rbox(img, bboxes, texts)
        img = self.transform(img)
        return img_filename, img, bboxes, texts, score_map, geo_map, angle_map, training_mask
    
    def _load_annotations(self, img_filename):
        gt_path = os.path.join(self.gt_dir, f'gt_{img_filename}.mat')
        gt = loadmat(gt_path)['polygt']
        bboxes = []
        texts = []
        # ground truth format is ['x:', x-coords, 'y:', y-coords, text, orientation]
        for _, x, _, y, text, orientation in gt:
            if orientation == "#": # the pound symbol means "don't care"
                continue
                #text = "#"
            # remove unnecessary dimensions
            x, y = np.squeeze(x), np.squeeze(y)
            # zip the x and y arrays to get a list of points
            # data type must be np.int32; otherwise, cv2.minAreaRect will raise an error
            original_bbox = np.array(list(zip(x, y)), dtype=np.int32)
            # get the minimum bounding rectangle for the bbox (may be rotated)
            # to make sure that the bounding box is a rectangle, instead of a 
            # parallelogram or any other shape
            min_bounding_rect = cv2.minAreaRect(original_bbox) 
            # get the four corner points of the rectangle
            new_bbox = cv2.boxPoints(min_bounding_rect)
            # sort the points in top left, top right, bottom right, bottom left order
            new_bbox = sort_points_clockwise(new_bbox)
            bboxes.append(new_bbox)
            texts.append(text[0]) # text is an array with one element
        # make sure the number of bboxes is the same as the number of texts
        assert len(bboxes) == len(texts)
        # convert to numpy arrays
        bboxes = np.int0(bboxes)
        texts = np.array(texts)
        return bboxes, texts

    def _get_random_item(self):
        rand_idx = np.random.randint(0, len(self)-1)
        return self[rand_idx]

class SynthText(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'SynthText')
        gt_path = os.path.join(self.root_dir, 'SynthText/gt.mat')
        gt = loadmat(gt_path, squeeze_me=True, variable_names=['imnames', 'wordBB', 'txt'])
        self.img_filenames = gt['imnames']
        self.all_bboxes = gt['wordBB']
        self.all_texts = gt['txt']
        self._clean_up_data(self.all_bboxes, self.all_texts)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ) # from ImageNet
        ])

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        img = cv2.imread(img_path)
        bboxes, texts = self._load_annotations(idx)
        score_map, geo_map, angle_map, training_mask = generate_rbox(img, bboxes, texts)
        img = self.transform(img)
        return img_filename, img, bboxes, texts, score_map, geo_map, angle_map, training_mask

    def _fix_bbox_dim(self, bboxes):
        # if the dim of the box is 2, change the dim to 3 so it matches with the rest
        if bboxes.ndim == 2:
            bboxes = bboxes[:, :, np.newaxis]
            print("fixed dim")
        return bboxes

    def _check_dim_mismatch(self, bboxes, texts):
        # drop the data if the dim of boxes doesn't match with the len of texts
        # shape[2] because before reshape the bbox shape[2] contains N numbers of boxes
        if bboxes.shape[2] != len(texts):
            print("mismatch found")
            return True
        return False

    def _clean_up_data(self, bboxes, texts):
        print("info: ", type(bboxes), type(texts), len(bboxes), len(texts))
        N = len(texts)
        print(N)
        print(np.where(bboxes.ndim == 2))
        # filter bboxes that are not compatible with corresponding texts

    def _load_annotations(self, idx):
        # the format of the ground truth annotation file can be found in
        # https://www.robots.ox.ac.uk/~vgg/data/scenetext/readme.txt
        #idx = self._clean_up_data(idx)
        texts = [text for texts in self.all_texts[idx] for text in re.split('\n| ', texts.strip()) if text]
        bboxes = self.all_bboxes[idx]
        bboxes = self._fix_bbox_dim(bboxes)
        print(self.img_filenames[idx], bboxes.shape, len(texts))
        # zip x and y to get to a list of points
        bboxes = [list(zip(bboxes[0, :, i], bboxes[1, :, i])) for i in range(bboxes.shape[2])]
        # convert to numpy arrays
        bboxes = np.int0(bboxes)
        texts = np.array(texts)
        # make sure the number of bboxes is the same as the number of texts
        assert len(bboxes) == len(texts)
        return bboxes, texts

    def _get_random_item(self):
        rand_idx = np.random.randint(0, len(self)-1)
        return self[rand_idx]


def test_datasets():
    print("Loading datasets...")
    root = '../../datasets/smalltotal/Trainsets'
    data_train = TotalText(root)
    train_loader = DataLoader(data_train, batch_size=3, shuffle=True, num_workers=2, collate_fn=collate_fn)
    print(len(train_loader))
    filename = data_train.img_filenames[2]
    img_path = root + '/data/' + filename + '.jpg'
    image = cv2.imread(img_path)
    img, rh, rw = prepare_image(image, 640, 640)
    scale = np.array([[rw, rh]])
    bboxes, texts = data_train._load_annotations(filename)
    bboxes = (bboxes * scale).astype(bboxes.dtype)
    '''
    for i in range(bboxes.shape[0]):
        pts = bboxes[i, :, :]
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255))
    cv2.imshow("img", img)
    cv2.waitKey()
    '''
    score_map, geo_map, angle_map, training_mask = generate_rbox(img, bboxes, texts)
    print(img.shape, score_map.shape, geo_map.shape, angle_map.shape, training_mask.shape)
