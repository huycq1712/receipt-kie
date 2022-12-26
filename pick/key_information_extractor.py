# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse
import torch
from tqdm import tqdm
from pathlib import Path

import sys, os
sys.path.append('/content/drive/MyDrive/deploy/invoice_kie/pick')


from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from pick.parse_config import ConfigParser
import pick.model.pick as pick_arch_module
from pick.data_utils import documents
#from pick.data_utils.pick_dataset import PICKDataset
from pick.data_utils.pick_infer_dataset import PICKInferDataset
from pick.data_utils.pick_infer_dataset import BatchCollateFn
from pick.utils.util import iob_index_to_str, text_index_to_str

#import onnx

from export_pick import PICKOnnxModel
#PICKOnnxModel = []

class KeyInforExtraction:

    def __init__(self, use_onnx=False) -> None:
        super().__init__()
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = torch.load('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/model_best.pth', map_location=self.device)
        self.config = self.checkpoint['config']
        self.state_dict = self.checkpoint['state_dict']
        self.monitor_best = self.checkpoint['monitor_best']
        self.use_onnx = use_onnx

        self.pickmodel = self.config.init_obj('model_arch', pick_arch_module)
        self.pickmodel = self.pickmodel.to(self.device)
        self.pickmodel.load_state_dict(self.state_dict)
        self.pickmodel.eval()
        if self.use_onnx:
            self.pickmodel_onnx = PICKOnnxModel()

    def extract_key_information(self, image, boxes_and_transcript,  image_file, boxes_and_transcripts_folder):
        boxes_and_transcripts_folder: Path = Path(boxes_and_transcripts_folder)
        image_file = Path(image_file)
        boxes_and_transcripts = [documents.Document(boxes_and_transcripts_file=boxes_and_transcripts_folder,
                                           image_index=0,
                                           image_file=image_file,
                                           resized_image_size=(480, 960),
                                           training=False,
                                           boxes_and_transcripts_list=boxes_and_transcript,
                                           image=image)]
                                    
        test_dataset = PICKInferDataset(document_list=boxes_and_transcripts)
        test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      num_workers=2, collate_fn=BatchCollateFn(training=False))



        with torch.no_grad():
            for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
                print('step_idx: ', step_idx)
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(self.device)

                output = self.pickmodel(**input_data_item) if not self.use_onnx else self.pickmodel_onnx(**input_data_item)


                #output = self.pickmodel(**input_data_item)
                logits = output['logits']  # (B, N*T, out_dim)
                new_mask = output['new_mask']
                image_indexs = input_data_item['image_indexs']  # (B,)
                text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
                mask = input_data_item['mask']
                # List[(List[int], torch.Tensor)]

                if self.use_onnx:
                    logits = torch.from_numpy(logits)
                    new_mask = torch.from_numpy(new_mask)
                    
                best_paths = self.pickmodel.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
                predicted_tags = []
                for path, score in best_paths:
                    predicted_tags.append(path)

                # convert iob index to iob string
                decoded_tags_list = iob_index_to_str(predicted_tags)
                # union text as a sequence and convert index to string
                decoded_texts_list = text_index_to_str(text_segments, mask)

                for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                    # List[ Tuple[str, Tuple[int, int]] ]
                    spans = bio_tags_to_spans(decoded_tags, [])
                    spans = sorted(spans, key=lambda x: x[1][0])

                    entities = []  # exists one to many case
                    for entity_name, range_tuple in spans:
                        entity = dict(entity_name=entity_name,
                                        text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                        entities.append(entity)
            
            return entities



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default=None, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default=None, type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default='predict_results', type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('-g', '--gpu', default=-1, type=int,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args = args.parse_args()
    #main(args)
