# Binary.py

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import os
import cv2
import numpy as np
from tqdm import tqdm


class Detector:
    def __init__(self, model_type = "OD"):
        self.cfg = get_cfg()
        self.model_type = model_type
        
        # Load model config and pretrained model
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        elif model_type == "KP":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "PS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
            
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" # cpu or cuda
        
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        # Load the image
        image = cv2.imread(imagePath)
        
        # Resize the image to 960x512
        image = cv2.resize(image, (960, 512))
        cv2.imwrite(imagePath, image)
        
        # Perform the prediction
        predictions = self.predictor(image)
        instances = predictions["instances"].to("cpu")
        
        # Extract the mask for each instance and combine all the masks into a single binary image
        mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in tqdm(range(len(instances)), desc="Processing instance masks"):
            mask = instances.pred_masks[i]
            mask_image[mask] = 255
        
        # Get the original file name and construct the output path
        file_name = os.path.basename(imagePath)
        input_folder = os.path.dirname(imagePath)
        output_folder = input_folder + "_mask"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, file_name)
        
        # Save the binary mask image to the specified output path
        cv2.imwrite(output_path, mask_image)