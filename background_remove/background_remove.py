import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
from tqdm import tqdm
import ast
import pandas as pd


class BackgroundRemove:

    def __init__(self, src='/home/huycq/OCR/Project/KIE/invoice/invoice_kie/background_remove/yolov5', model_path='/home/huycq/OCR/Project/KIE/invoice/invoice_kie/background_remove/yolov5/weights/best_ok.onnx') -> None:
        self.model = torch.hub.load(src, 'custom', model_path, source='local')
        self.size = 640

    def remove_background_from_directory(self, directory, output_directory, csv_file='/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_val_data/mcocr_val_sample_df.csv'):
        if csv_file is not None:
            df_grouth_truth = pd.read_csv(csv_file)
            #print(df_grouth_truth)
        else:
            df_grouth_truth = None

        for file in tqdm(os.listdir(directory)):
            if file.endswith(".jpg") or file.endswith(".png"):
                ground_truth_box = df_grouth_truth.loc[df_grouth_truth['img_id'] == file] if df_grouth_truth is not None else None
                row = self.remove_background_from_image(directory + '/' + file, output_directory + '/' + file, ground_truth_box)
                if df_grouth_truth is not None:
                    df_grouth_truth.loc[df_grouth_truth['img_id'] == file] = row
            else:
                continue

        if df_grouth_truth is not None:
            df_grouth_truth.to_csv("/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_val_data/mcocr_val_df_br.csv", index=False)

    def remove_background_from_image(self, image_path, output_path, ground_truth_box=None):
        #new_ground_truth_box = ground_truth_box.copy()

        img = Image.open(image_path)
        results = self.model(img)
        
        #get max confidence
        if results.pred[0].shape[0] !=0 :
            max_conf_box = torch.argmax(results.pred[0][:, 4], axis=0)
            x1, y1, x2, y2 = results.pred[0][max_conf_box, :4].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img = img.crop((x1, y1, x2, y2))

            if ground_truth_box is not None:
                x_translation = x1
                y_translation = y1

                #for index, image in ground_truth_box.iterrows():
                #print(ground_truth_box)
                boxes = ast.literal_eval(ground_truth_box["anno_polygons"].iloc[0])
                
                #boxes = ground_truth_box["anno_polygons"]
                for i, text_box in enumerate(boxes):
                    for j, seg in enumerate(text_box['segmentation']):
                        for k in range(len(seg)):
                            if k % 2 == 0:
                                boxes[i]['segmentation'][j][k] = boxes[i]['segmentation'][j][k] - x_translation
                            else:
                                boxes[i]['segmentation'][j][k] = boxes[i]['segmentation'][j][k] - y_translation

                    boxes[i]['bbox'][0] = boxes[i]['bbox'][0] - x_translation
                    boxes[i]['bbox'][1] = boxes[i]['bbox'][1] - y_translation

                #print(ground_truth_box["anno_polygons"].iloc[0])
                ground_truth_box["anno_polygons"].iloc[0] = str(boxes)

        img.save(output_path)

        return ground_truth_box

    
    def remove_background(self, img):
        #img = Image.open('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/data/mc_ocr_train/images/mcocr_public_145013aagqw.jpg')
        img = img.resize((self.size, self.size))
        results = self.model(img)
        #get max confidence
        #print(results.shape)
        if results.pred[0].shape[0] !=0 :
            max_conf_box = torch.argmax(results.pred[0][:, 4], axis=0)
            x1, y1, x2, y2 = results.pred[0][max_conf_box, :4].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img = img.crop((x1, y1, x2, y2))
        return img

if __name__ == '__main__':
    img = Image.open('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/data/mc_ocr_train/images/mcocr_public_145013aagqw.jpg')
    br = BackgroundRemove()
    br.remove_background(img)
    img.save("test.jpg")
    #br.remove_background_from_image('data/images/zidane.jpg', 'data/images/zidane_out.jpg')
    #br.remove_background_from_directory('/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_val_data/val_images', '/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_val_data/val_images_br')