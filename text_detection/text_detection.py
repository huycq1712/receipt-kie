# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import time
import sys

import PaddleOCR.tools.infer.utility as utility
from PaddleOCR.ppocr.utils.logging import get_logger
from PaddleOCR.ppocr.utils.utility import get_image_file_list, check_and_read
from PaddleOCR.ppocr.data import create_operators, transform
from PaddleOCR.ppocr.postprocess import build_post_process
import json
from tqdm import tqdm
logger = get_logger()


class TextDetector(object):
    def __init__(self, args=None):
        #self.args = args
        self.det_algorithm = 'DB'
        #elf.use_onnx = args.use_onnx
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': 'max',
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        #if self.det_algorithm == "DB":
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = 0.3
        postprocess_params["box_thresh"] = 0.6
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = 'fast'
        postprocess_params["box_type"] = 'quad'
        

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        #self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
        #    args, 'det', logger)

        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_infer("/content/drive/MyDrive/deploy/invoice_kie/text_detection/PaddleOCR/ch_ppocr_server_v2.0_det_infer", 'det')

        self.preprocess_op = create_operators(pre_process_list)


    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def detect_text(self, img):
        img = np.array(img)
        ori_im = img.copy()
        data = {'image': img}

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        
        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        
        preds['maps'] = outputs[0]

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        src_im = utility.draw_text_det_res(dt_boxes, ori_im)
        
        dt_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]), reverse=True)
        #print(dt_boxes)
        return dt_boxes, src_im

    def detect_image(self, file_name):
        results = {}
        results['image_path'] = file_name
        results['boxes'] = []
        img = cv2.imread(file_name)
        imgs = [img]
        
        for index, img in enumerate(imgs):
            
            dt_boxes, _ = text_detector(img)
            if dt_boxes is not None:
                for box_cor in dt_boxes:
                    box = {}
                    box['box'] = box_cor.tolist()
                    results['boxes'].append(box)

        return results

    def flip_90(self, img):
        img = np.array(img)
        ori_im = img.copy()
        data = {'image': img}

        data = transform(data, self.preprocess_op)
        img, shape_list = data

        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

       
        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}

        preds['maps'] = outputs[0]

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']

        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        
        widths = np.sqrt((dt_boxes[:, 0, 0] - dt_boxes[:, 1, 0])**2 + (dt_boxes[:, 0, 1] - dt_boxes[:, 1, 1])**2)
        heights = np.sqrt((dt_boxes[:, 0, 0] - dt_boxes[:, 3, 0])**2 + (dt_boxes[:, 0, 1] - dt_boxes[:, 3, 1])**2)
        a_0 = np.sum(widths > heights)
        if np.sum(widths>heights) < len(dt_boxes)/2:
            image_res = cv2.rotate(ori_im, cv2.ROTATE_90_CLOCKWISE)

        else:
            image_res = ori_im
    
        return  image_res

    def detect_directory(self, image_path="/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_train_data/train_images_br_f90_align", save_path="/content/drive/MyDrive/invoice_kie/data/mcocr_public_train_test_shared_data/mcocr_train_data/res_text_det"):
        for file_name in tqdm(os.listdir(image_path)):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                file_path = os.path.join(image_path, file_name)
                results = self.detect_image(file_path)

                with open(save_path + '/' + file_name.replace('jpg', 'json'), 'w') as f:
                    json.dump(results, f, indent=2)
                #self.draw_results(results, save_path)

if __name__ == "__main__":
    args = utility.parse_args()
    text_detector = TextDetector(args)
    text_detector.detect_directory()
    """args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)
    total_time = 0
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)

    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = text_detector(img)

    save_results = []
    for idx, image_file in enumerate(image_file_list):
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            dt_boxes, s , image_save = text_detector.rotate_90_detect(img)

            save_file = image_file
            img_path = os.path.join(
                draw_img_save_dir,
                "det_res_{}".format(os.path.basename(save_file)))
            cv2.imwrite(img_path, image_save)
            logger.info("The visualized image saved in {}".format(img_path))"""