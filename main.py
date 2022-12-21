from background_remove import background_remove
from image_rotation import image_rotation
from text_detection import text_detection
from text_recognition import text_recognition
from pick import key_information_extractor

import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

import time


class Pipeline:
    def __init__(self, output_folder=''):
        self.num_test = 0
        self.output_folder = output_folder

        self.background_remove = background_remove.BackgroundRemove()
        self.image_rotation = image_rotation.ImageRotation()
        self.text_detection = text_detection.TextDetector()
        self.text_recognition = text_recognition.TextRecognition()
        self.key_information_extraction = key_information_extractor.KeyInforExtraction()

    def run(self, image):
        self.num_test = self.num_test + 1
        start_br = time.time()
        image_br = self.background_remove.remove_background(image) #PIL image
        start_br_flip = time.time()
        image_br_flip = self.text_detection.flip_90(image_br) #PIL image
        image_final, _ = self.image_rotation.rotate_image(image_br_flip) #PIL image
        #mage_final.save(self.output_folder + "/image_final_test_{}.jpg".format(self.num_test))
        start_text = time.time()
        text_boxes, image_text_box = self.text_detection.detect_text(image_final)
        start_recog = time.time()
        text_boxes = self.text_recognition.recognize_text(image_final, text_boxes)
        #print(text_boxes)
        image_text_box = draw_text_box(image_final, text_boxes)

        #create draw bouding boxes with label
        start_kie = time.time()

        result = self.key_information_extraction.extract_key_information(image_final, text_boxes, self.output_folder, self.output_folder)
        
        end = time.time()
        result = postprocess_result(result)
        result = pd.DataFrame(result)

        print("-----------------------------------")
        print("Back ground remove {} second".format(round(start_br_flip - start_br, 2)))
        print("Image align {} second".format(round(start_text - start_br_flip, 2)))
        print("Text detection {} second".format(round(start_recog - start_text, 2)))
        print("Text recognition {} second".format(round(start_kie - start_recog, 2)))
        print("KIE {} second".format(round(end - start_kie, 2)))
        print("-----------------------------------")

        return image_br, image_final, image_text_box, result


def postprocess_result(entities_result):
    result = {"SELLER":[' '],
    "ADDRESS":[' '],
    "TIMESTAMP":[' '],
    "TOTAL_COST":[' ']
    }

    for entity in entities_result:
        for key in result:
            if key == entity['entity_name']:
                
                if key=='TOTAL_COST' and entity['text'][0].isnumeric():
                    result[key][0] = entity['text'] + " " + result[key][0]
                
                elif key == 'TIMESTAMP':
                    result[key][0] = result[key][0] + fix_datetime(entity['text'])

                else:
                    result[key][0] = result[key][0] + " " + entity['text']
    
    return result

  
def fix_datetime(input_str):  # for string len >42

    TIMESTAMP_keys = ['Ngày', 'Thời gian']
    TIMESTAMP_noise_keys = ['Số HĐ', 'Số GD']
    input_str = input_str.lstrip(' ').rstrip(' ')
    lower_input_str = input_str.lower()
    final_time_pos = -1
    for k in TIMESTAMP_keys:
        lower_k = k.lower()
        time_pos = lower_input_str.find(lower_k)
        if time_pos != -1:
            final_time_pos = time_pos

    final_noise_pos = -1
    for k in TIMESTAMP_noise_keys:
        lower_k = k.lower()
        noise_pos = lower_input_str.find(lower_k)
        if noise_pos != -1:
            final_noise_pos = noise_pos

    final_str = input_str
    if final_noise_pos > 0:
        final_str = input_str[:final_noise_pos]
    if final_time_pos > 0:
        final_str = input_str[final_time_pos:]

    final_str = final_str.lstrip(' ').rstrip(' ')

    return final_str

def draw_text_box(image_in, text_boxes):
    #image = PIL.Image.fromarray(image)
    image = image_in.copy()
    font = ImageFont.truetype("/content/drive/MyDrive/deploy/invoice_kie/arial/arial.ttf", 15)

    #font = ImageFont.truetype(font='ARIALUNI.TTF',size=20)
    for i, box in enumerate(text_boxes):
        #x, y = box['box'][0]
        #w, h = box['box'][2][0] - box['box'][0][0], box['box'][2][1] - box['box'][0][1]
        text = box['text']
        rec = [box['box'][0][0], box['box'][0][1], box['box'][2][0], box['box'][2][1]]

        image_dr = ImageDraw.Draw(image)  
        image_dr.rectangle(xy = rec, fill=None, outline ="red")
        image_dr.text((box['box'][0][0], box['box'][0][1]-15), text, fill ="red", font=font)
        #image = ImageDraw.Draw(image)  
        #image = image.polygon(box, fill=None, outline ="blue")
        #
        #image = image.polyline(box, outline='red', width=2)
        #image = cv2.polylines(image, [box], True, (0, 255, 0), 2)
        #image = cv2.putText(image, text, (box[0][0], box[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return np.array(image)
if __name__ == '__main__':
    pass