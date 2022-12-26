import torch
import torch.nn as nn
import torchvision.transforms as transforms

import ast
import cv2
import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import onnxruntime as rt

from image_rotation.upsidedowndetector import UpsideDowndetector
from config import model_config


class ImageRotation:

    def __init__(self, cfg=model_config) -> None:
        self.cfg = cfg
        self.use_onnx = self.cfg['image_rotation']['use_onnx']
        self.device = self.cfg['device']
        self.upside_down_detector = UpsideDowndetector()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        

        if self.use_onnx:
            self.upside_down_detector = rt.InferenceSession(self.cfg['image_rotation']['onnx_path'][0])
        else:
            self.upside_down_detector.load_state_dict(torch.load(self.cfg['image_rotation']['model_path'], map_location=self.device))
            self.upside_down_detector.eval()
            self.upside_down_detector.to(self.device)


    def rotate_image(self, image, ground_truth_box=None):
        
        original_image = image.copy()
        image = Image.fromarray(image)
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        
        if self.use_onnx:
            outputs = self.upside_down_detector.run(None, {self.upside_down_detector.get_inputs()[0].name: image.numpy()})[0]
            outputs = torch.tensor(outputs)
        else:
            outputs = self.upside_down_detector(image)
        
        outputs = outputs.squeeze(0)
        outputs = torch.softmax(outputs, dim=0)
        
        if outputs[1] > 0.8:

            original_image = cv2.rotate(original_image, cv2.ROTATE_180)
            if ground_truth_box is not None:
                boxes = ast.literal_eval(ground_truth_box["anno_polygons"].iloc[0])
        
                for i, text_box in enumerate(boxes):
                    for j, seg in enumerate(text_box['segmentation']):
                        for k in range(0, len(seg), 2):
                            boxes[i]['segmentation'][j][k] = original_image.shape[1] - boxes[i]['segmentation'][j][k]
                            boxes[i]['segmentation'][j][k+1] = original_image.shape[0] - boxes[i]['segmentation'][j][k+1]
                        
                        boxes[i]['segmentation'][j] = boxes[i]['segmentation'][j][4:] + boxes[i]['segmentation'][j][:4]
                        boxes[i]['bbox'][0] = original_image.shape[1] - (boxes[i]['bbox'][0] + boxes[i]['bbox'][2])
                        boxes[i]['bbox'][1] = original_image.shape[0] - (boxes[i]['bbox'][1] + boxes[i]['bbox'][3])

                ground_truth_box["anno_polygons"].iloc[0] = str(boxes)


        img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

        angles = []

        if lines is None:
            original_image = Image.fromarray(original_image)
            return original_image, ground_truth_box

        for [[x1, y1, x2, y2]] in lines:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        median_angle = np.median(angles)

        if abs(median_angle) < 7:
            center = tuple(np.array(original_image.shape[1::-1]) / 2)
            M_angle = cv2.getRotationMatrix2D(center, angle=median_angle, scale=1.0)
            img_rotated = cv2.warpAffine(original_image, M_angle, original_image.shape[1::-1])

            if ground_truth_box is not None:
                boxes = ast.literal_eval(ground_truth_box["anno_polygons"].iloc[0])
                for i, text_box in enumerate(boxes):
                    for j, seg in enumerate(text_box['segmentation']):
                        for k in range(0, len(seg), 2):
                            boxes[i]['segmentation'][j][k], boxes[i]['segmentation'][j][k+1] = np.dot(M_angle, np.array([boxes[i]['segmentation'][j][k], boxes[i]['segmentation'][j][k+1], 1.0]))
                        
                        old_x, old_y = boxes[i]['bbox'][0], boxes[i]['bbox'][1]
                        boxes[i]['bbox'][0], boxes[i]['bbox'][1] = np.dot(M_angle, np.array([boxes[i]['bbox'][0], boxes[i]['bbox'][1], 1.0]))
                        boxes[i]['bbox'][2], boxes[i]['bbox'][3] = np.dot(M_angle, np.array([boxes[i]['bbox'][2] + old_x, boxes[i]['bbox'][3]+old_y, 1.0]))
                        boxes[i]['bbox'][2] = boxes[i]['bbox'][2] - boxes[i]['bbox'][0]
                        boxes[i]['bbox'][3] = boxes[i]['bbox'][3] - boxes[i]['bbox'][1]

                ground_truth_box["anno_polygons"].iloc[0] = str(boxes)

        else:
            original_image = Image.fromarray(original_image)
            return original_image, ground_truth_box

        
        img_rotated = Image.fromarray(img_rotated)
        return img_rotated, ground_truth_box


    def rotate_from_directory(self, src_path, save_path, csv_path=None):
        if csv_path is not None:
            df_grouth_truth = pd.read_csv(csv_path)

        for file_name in tqdm(os.listdir(src_path)):
            image = cv2.imread(src_path + '/' + file_name)
            ground_truth_box = df_grouth_truth.loc[df_grouth_truth['img_id'] == file_name] if csv_path is not None else None
            image, ground_truth_box = self.rotate_image(image, ground_truth_box)

            df_grouth_truth.loc[df_grouth_truth['img_id'] == file_name] = ground_truth_box
            cv2.imwrite(save_path + '/' + file_name, image)

        df_grouth_truth.to_csv("/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_train_data/mcocr_train_df_br_f90_align.csv", index=False)



if __name__ == '__main__':
    r = Rotation()
    r.rotate_from_directory('/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_train_data/train_images_br_f90',
    '/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_train_data/train_images_br_f90_align',
    '/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_train_data/mcocr_train_df_br_f90.csv')