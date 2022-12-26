# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse
import torch
from tqdm import tqdm
from pathlib import Path


import sys, os
sys.path.append('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick')

import onnxruntime as rt

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from pick.parse_config import ConfigParser
import pick.model.pick as pick_arch_module
from pick.data_utils import documents
#from pick.data_utils.pick_dataset import PICKDataset
from pick.data_utils.pick_infer_dataset import PICKInferDataset
from pick.data_utils.pick_infer_dataset import BatchCollateFn
from pick.utils.util import iob_index_to_str, text_index_to_str

#from key_information_extractor import KeyInforExtraction

class PICKOnnxExporter:
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = torch.load('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/model_best.pth', map_location=self.device)
        self.config = self.checkpoint['config']
        self.state_dict = self.checkpoint['state_dict']
        self.monitor_best = self.checkpoint['monitor_best']

        self.pickmodel = self.config.init_obj('model_arch', pick_arch_module)
        self.pickmodel = self.pickmodel.to(self.device)
        self.pickmodel.load_state_dict(self.state_dict)
        self.pickmodel.eval()
        
        
    def embedding_export(self):
        word_emb = self.pickmodel.word_emb
        word_emb = torch.jit.script(word_emb)
            
        input_dummy = torch.randint(size=(1, 10, 245), high=243)
        input_names = ['text_segments']
        output_names = ['text_emb']
        dynamic_axes = {'text_segments': {0: 'B', 1:'N' ,2: 'T'},
                        'text_emb': {0: 'B', 1:'N', 2: 'T', '3': 'D'}}
        
        torch.onnx.export(word_emb,
                          args=input_dummy,
                          f='/content/drive/MyDrive/deploy/invoice_kie/pick/weights/embedding.onnx',
                          verbose=True,
                          input_names=input_names,
                          output_names=output_names,
                          export_params=True,
                          dynamic_axes=dynamic_axes)
        
        
    def encoder_export(self):
        encoder = self.pickmodel.encoder
        
        #B=1, N=10, C=512, H=480, W=960 T=40
        whole_image_dummy = torch.rand((1, 3, 480, 960)) #B, C, H, W
        boxes_coordinate_dummpy = torch.randint(size=(1, 10, 8), high=255) #B, N, 8
        text_emb_dummy = torch.rand((1, 10, 40 ,512)) #B, N, T, D
        src_key_padding_mask_dummy = torch.rand((10, 40)) #B*N, T
        
        #x_dummpy = torch.rand(size=(10, 40, 512), high=243)
        
        #inputs = (whole_image_dummy, boxes_coordinate_dummpy, text_emb_dummy, src_key_padding_mask_dummy)
        inputs = {
            'images': whole_image_dummy,
            'boxes_coordinate': boxes_coordinate_dummpy,
            'transcripts': text_emb_dummy,
            'src_key_padding_mask': src_key_padding_mask_dummy,
        }
        input_names = ['images', 'boxes_coordinate', 'transcripts', 'src_key_padding_mask']
        output_names = ['x']
        
        dynamic_axes = {'images': {0: 'B', 2: 'H', 3: 'W'},
                        'boxes_coordinate': {0: 'B', 1: 'N'},
                        'transcripts': {0: 'B', 1: 'N', 2: 'T'},
                        'src_key_padding_mask': {0: 'B*N', 1: 'T'},
                        }
        
        torch.onnx.export(encoder,
                          args=inputs,
                          f='/content/drive/MyDrive/deploy/invoice_kie/pick/weights/encoder.onnx',
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          export_params=True,
                         dynamic_axes=dynamic_axes)
        
        print("ok")
    
    
    def graph_export(self):
        graph = self.pickmodel.graph
        
        x_gcn_dummy = torch.rand(size=(1, 10, 512)) #B, N, D
        relation_features_dummpy = torch.rand(size=(1, 10, 10, 6)) #B, N, N, 6
        init_adj_dummy = torch.rand(size=(1, 10, 10)) #B, N, N
        boxes_num_dummy = torch.tensor([[10]]) #B, 1
        
        inputs = (x_gcn_dummy, relation_features_dummpy, init_adj_dummy, boxes_num_dummy)
        input_names = ['x_gcn', 'relation_features', 'init_adj', 'boxes_num']
        output_names = ['x_gcn', 'soft_adj']
        
        dynamic_axes = {'x_gcn': {0: 'B', 1: 'N', 2: 'D'},
                        'relation_features': {0: 'B', 1: 'N', 2: 'N'},
                        'init_adj': {0: 'B', 1: 'N', 2: 'N'},
                        'boxes_num': {0: 'B', 1: '1'},}
        
        torch.onnx.export(graph,
                          args=inputs,
                          f='/content/drive/MyDrive/deploy/invoice_kie/pick/weights/graph.onnx',
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          export_params=True,
                          dynamic_axes=dynamic_axes)
        
        
    def decoder_export(self):
        decoder = self.pickmodel.decoder
        
        x_dummpy = torch.rand(size=(2, 10, 40, 512)) #B, N, T, D
        x_gcn_dummy = torch.rand(size=(2, 10, 512)) #B, N, D
        mask_dummy = torch.rand(size=(2, 10, 40)) #B, N, T
        text_length_dummpy = torch.randint(size=(2, 10), high=40) #B, N
        iob_tags_label_dummy = torch.randint(size=(2, 10, 40), high=243) #B, N, T
        
        inputs = (x_dummpy, x_gcn_dummy, mask_dummy, text_length_dummpy, iob_tags_label_dummy)
        input_names = ['x', 'x_gcn', 'mask', 'text_length', 'iob_tags_label']
        output_names = ['logits', 'new_mask']
        
        dynamic_axes = {'x': {0: 'B', 1: 'N', 2: 'T', 3: 'D'},
                        'x_gcn': {0: 'B', 1: 'N', 2: 'D'},
                        'mask': {0: 'B', 1: 'N', 2: 'T'},
                        'text_length': {0: 'B', 1: 'N'},
                        'iob_tags_label': {0: 'B', 1: 'N', 2: 'T'},
                        'logits': {0: 'B', 1: 'N', 2: 'T', 3: 'D'},
                        'new_mask': {0: 'B', 1: 'N', 2: 'T'},
                        #'log_linear_loss': {0: 'B', 1: 'N', 2: 'T'}
                        }
        
        torch.onnx.export(decoder,
                          args=inputs,
                          f='/content/drive/MyDrive/deploy/invoice_kie/pick/weights/decoder.onnx',
                          input_names = input_names,
                          output_names = output_names,
                          dynamic_axes = dynamic_axes
                          )
        
        print("ok")
    def crf_export(self):
        crf = self.pickmodel.decoder.crf
        
        
    def pick_export(self):
        self.embedding_export()
        self.encoder_export()
        self.graph_export()
        self.decoder_export()

class PICKOnnxModel:
    def __init__(self):
        self.training = False
        self.word_emb = rt.InferenceSession('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/embedding.onnx')
        self.encoder = rt.InferenceSession('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/encoder.onnx')
        self.graph = rt.InferenceSession('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/graph.onnx')
        self.decoder = rt.InferenceSession('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/decoder.onnx')
    
    
    def __call__(self, **kwargs):
        whole_image = kwargs['whole_image']  # (B, 3, H, W)
        relation_features = kwargs['relation_features']  # initial relation embedding (B, N, N, 6)
        text_segments = kwargs['text_segments']  # text segments (B, N, T)
        text_length = kwargs['text_length']  # (B, N)
        iob_tags_label = kwargs['iob_tags_label'] if self.training else None  # (B, N, T)
        mask = kwargs['mask'].to(torch.long)  # (B, N, T)
        boxes_coordinate = kwargs['boxes_coordinate']  # (B, num_boxes, 8)
        
        text_emb = self.word_emb.run(None, {self.word_emb.get_inputs()[0].name: text_segments.cpu().numpy()})[0]
        
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)
        #print(src_key_padding_mask)
        print(src_key_padding_mask.cpu().numpy())
        x = self.encoder.run(None, {
            self.encoder.get_inputs()[0].name: whole_image.cpu().numpy(),
            self.encoder.get_inputs()[1].name: boxes_coordinate.cpu().numpy(),
            self.encoder.get_inputs()[2].name: text_emb,
            self.encoder.get_inputs()[3].name: src_key_padding_mask.cpu().numpy(),
        })[0]
        B, N, T = mask.shape
        
        text_mask = torch.logical_not(src_key_padding_mask).byte()
        x_gcn = self._aggregate_avg_pooling(torch.tensor(x), text_mask)
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
        
        x_gcn = x_gcn * graph_node_mask.float()
        
        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N))
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)
        
        x_gcn = x_gcn.reshape(B, N, -1)
        x_gcn, soft_adj = self.graph.run(None, {self.graph.get_inputs()[0].name: x_gcn.cpu().numpy(),
                                            self.graph.get_inputs()[1].name: relation_features.cpu().numpy(),
                                            self.graph.get_inputs()[2].name: init_adj.cpu().numpy(),
                                            self.graph.get_inputs()[3].name: boxes_num.cpu().numpy()})
        
        adj = torch.tensor(soft_adj)*init_adj
        logits, new_mask = self.decoder.run(None, {self.decoder.get_inputs()[0].name: x.reshape(B, N, T, -1),
                                                   self.decoder.get_inputs()[1].name: x_gcn,
                                                   self.decoder.get_inputs()[2].name: mask.cpu().numpy(),
                                                   self.decoder.get_inputs()[3].name: text_length.cpu().numpy(),
                                                   })
        
        #self.decoder.get_inputs()[4].name: iob_tags_label.cpu().numpy()
        output = {
            'logits': logits,
            'new_mask': new_mask,
            'adj': adj,
        }
        
        return output
       
    
    def _aggregate_avg_pooling(self, input, text_mask):
        '''
        Apply mean pooling over time (text length), (B*N, T, D) -> (B*N, D)
        :param input: (B*N, T, D)
        :param text_mask: (B*N, T)
        :return: (B*N, D)
        '''
        print("=============")
        print(input.shape)
        print(text_mask.shape)
        print("=============")
        # filter out padding value, (B*N, T, D)
        input = input * text_mask.detach().unsqueeze(2).float()
        # (B*N, D)
        sum_out = torch.sum(input, dim=1)
        # (B*N, )
        text_len = text_mask.float().sum(dim=1)
        # (B*N, D)
        print("=============")
        print(text_len.shape)
        print(sum_out.shape)
        print("=============")
        text_len = text_len.unsqueeze(1).expand_as(sum_out)
        text_len = text_len + text_len.eq(0).float()  # avoid divide zero denominator
        # (B*N, D)
        mean_out = sum_out.div(text_len)
        return mean_out
    
    
    @staticmethod
    def compute_mask(mask: torch.Tensor):
        '''
        :param mask: (B, N, T)
        :return: True for masked key position according to pytorch official implementation of Transformer
        '''
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        print(type(mask))
        mask_sum = mask.sum(dim=-1)  # (B*N,)

        # (B*N,)
        graph_node_mask = mask_sum != 0
        # (B * N, T)
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)  # True for valid node
        # If src key are all be masked (indicting text segments is null), atten_weight will be nan after softmax
        # in self-attention layer of Transformer.
        # So we do not mask all padded sample. Instead we mask it after Transformer encoding.
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  # True for padding mask position
        return src_key_padding_mask, graph_node_mask
        
        
                          
if __name__ == "__main__":
    
    model = PICKOnnxExporter()
    #print(model.pickmodel)
    model.encoder_export()
    #test = torch.randint(size=(1, 10, 245), high=243)
    #embed = torch.nn.Embedding(243, 512)
    #print(embed(test).shape)