import torch
import torch.nn as nn

from modules.shared_conv import SharedConv
from modules.detector import Detector
from fots import FOTSModel
from modules.roi_rotate import ROIRotate
from modules.recognizer import Recognizer
import modules.alphabet

from utils.bbox import restore_bbox

class SRModel(nn.Module):
    def __init__(self, is_training):
        """
        Args:
            is_training (bool): whether the model is being trained or not
        """
        super().__init__()
        self.SharedConv = SharedConv()
        self.Detector = Detector()
        self.ROIRotate = ROIRotate()
        self.Recognizer = Recognizer(num_of_classes=modules.alphabet.NUM_OF_CLASSES)
        self.is_training = is_training

    #def forward(self, bboxes, bbox_to_img_idx, pretrained):
    def forward(self, imgs, bboxes=None, bbox_to_img_idx=None):
        """
        Args:
            imgs: Input images.
            bboxes: Coordinates of the ground-truth bounding boxes, ignored if 
                self.is_training is False.
            bbox_to_img_idx: Mapping between the bounding boxes and images, 
                ignored if self.is_training is False.
        """
        #score_maps, geo_maps, angle_maps = FOTSModel()
        shared_features = self.SharedConv(imgs)
        score_maps, geo_maps, angle_maps = self.Detector(shared_features)
        #score_maps, geo_maps, angle_maps = pretrained
        # TODO: transform into tensors
        #shared_features = pretrained.remove_artifacts()
        #if not self.is_training:
            # get the predicted bounding boxes
            #bboxes, bbox_to_img_idx = restore_bbox(score_maps, geo_maps, angle_maps)
        rois, seq_lens = self.ROIRotate(shared_features, bboxes, bbox_to_img_idx)
        log_probs = self.Recognizer(rois, seq_lens)
                
        return score_maps, geo_maps, angle_maps, bboxes, bbox_to_img_idx, log_probs, seq_lens
