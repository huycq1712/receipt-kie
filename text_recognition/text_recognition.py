import torch
import torch.nn as nn

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

import codecs
import json

import cv2
import numpy as np
import PIL
from PIL import Image, ImageEnhance

import os
import glob
from tqdm import tqdm

class TextRecognition:

    def __init__(self) -> None:
        self.config = Cfg.load_config_from_name('vgg_seq2seq')
        self.config['weights'] = '/content/drive/MyDrive/deploy/invoice_kie/text_recognition/vietocr/weights/transformerocr.pth'
        self.config['cnn']['pretrained']=False
        self.config['device'] = 'cpu'
        self.config['predictor']['beamsearch']=False
        self.recognizer = Predictor(self.config)

    def recognize_from_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        save_text_box_path = 'text_box/' + data["image_path"].split('/')[-1].split('.')[0]
        image = Image.open(data["image_path"]).convert("RGB")
        #os.mkdir(save_text_box_path)

        image = np.asarray(image)
        image = cv2.fastNlMeansDenoising(image, None, 20,7,21)
        image = Image.fromarray(image)
        image = ImageEnhance.Contrast(image).enhance(2.5)
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #new_img = Image.fromarray(new_img)
        image = Image.fromarray(image)

        for i, box in enumerate(data["boxes"]):

            x = [box['box'][i][0] for i in range(4)]
            y = [box['box'][i][1] for i in range(4)]

            left = max(min(x) - 2, 0)
            top = max(min(y) - 2, 0)
            right = min(max(x) + 2, image.width)
            bottom = min(max(y) + 2, image.height)

            cropped_image = image.crop((left, top, right, bottom))
            cropped_image = cropped_image.convert("L")
            cropped_image.save(save_text_box_path + "/" + "text_{}.jpg".format(i))
            #cropped_image = cropped_image.convert
            text = self.recognizer.predict(cropped_image)
            data["boxes"][i]["text"] = text

        return data

    def recognize_from_directory(self, image_path, save_path):
        for file_name in tqdm(os.listdir(image_path)):
            if file_name.endswith('.json'):
                file_path = os.path.join(image_path, file_name)
                results = self.recognize_from_json(file_path)

                with codecs.open(save_path + '/' + file_name, 'w', "utf-8") as outfile:
                    outfile.write(json.dumps(results, ensure_ascii=False))
                #self.draw_results(results, save_path)

    def recognize_text(self, image, text_boxes):
        result = []
        for i, box in enumerate(text_boxes):
            box = box.tolist()

            text_box = {}
            text_box["box"] = box

            x = [box[i][0] for i in range(4)]
            y = [box[i][1] for i in range(4)]

            left = max(min(x) - 3, 0)
            top = max(min(y) - 3, 0)
            right = min(max(x) + 3, image.width)
            bottom = min(max(y) + 3, image.height)

            cropped_image = image.crop((left, top, right, bottom))
            cropped_image = cropped_image.convert("L")
            text = self.recognizer.predict(cropped_image)

            text_box["text"] = text
            result.append(text_box)
        print("$$$$$$$$$$$$$$$$$")
        print(result)
        return result


if __name__ == "__main__":
    recognizer = TextRecognition()
    recognizer.recognize_from_directory("/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_train_data/res_text_det",
     "/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_train_data/res_text_recog")