import torch
from torchvision.transforms import transforms
from torch.nn.functional import log_softmax
import numpy as np
from PIL import Image

from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model
from vietocr.model.beam import Beam

import onnxruntime as ort


class VietOcrExporter:
    
    def __init__(self, model_path, onnx_path, image_size) -> None:
        self.config = Cfg.load_config_from_name('vgg_seq2seq')
        self.config['weights'] = model_path
        self.config['cnn']['pretrained']=False
        self.config['device'] = 'cpu'
        self.image_size = image_size
        self.onnx_path = onnx_path
        
        self.model, self.vocab = build_model(self.config)
        self.model.load_state_dict(torch.load(self.config['weights'], map_location=torch.device(self.config['device'])))
        self.model.eval() 
        

    @staticmethod
    def export_vietocr_cnn(model, onnx_path, image_size):
        dummpy_input = torch.rand(image_size)
        input_name = ['img']
        output_name = ['src']
        
        src = model(dummpy_input)
        torch.onnx.export(model,
                        dummpy_input,
                        onnx_path,
                        verbose=True,
                        input_names=input_name,
                        output_names=output_name, 
                        export_params=True)
        return src
    
    @staticmethod
    def export_vietocr_encoder(model, onnx_path, inputs):
        dummpy_src_input = inputs
        input_src_name = ['src']
        output_src_name = ['encoder_outputs', 'hidden']
        
        encoder_outputs, hidden = model(dummpy_src_input)
        torch.onnx.export(model,
                        dummpy_src_input,
                        onnx_path,
                        verbose=True,
                        input_names=input_src_name,
                        output_names=output_src_name,
                        export_params=True)
        
        
        return encoder_outputs, hidden
    
    @staticmethod
    def export_vietocr_decoder(model, onnx_path, inputs):
        
        input_name = ['tgt', 'hidden', 'encoder_outputs']
        output_name = ['output', 'hidden', 'attn']
        output, hidden, _ = model(inputs[0], inputs[1], inputs[2])
        
        torch.onnx.export(model,
                        (inputs[0], inputs[1], inputs[2]),
                        onnx_path, 
                        verbose=True, 
                        input_names=input_name,
                        output_names=output_name,
                        export_params=True)
        
        return output, hidden
    
    def eport_vietocr_seq2seq(self):
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = self.model_path
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        
        
        
        model, vocab = build_model(self.config)
        model.load_state_dict(torch.load(config['weights'], map_location=torch.device(config['device'])))
        model.eval() 
        
        inputs = torch.rand(self.image_size)
        
        src = self.export_vietocr_cnn(model.cnn, self.onnx_path[0], inputs.shape)
        encoder_outputs, hidden = self.export_vietocr_encoder(model.transformer.encoder, self.onnx_path[1], src)
        
        tgt = torch.LongTensor([[1] * len(inputs)])[-1]
        decoder_inputs = [tgt, hidden, encoder_outputs]
        
        self.export_vietocr_decoder(model.transformer.decoder, self.onnx_path[2], decoder_inputs)


class VietOcrOnnx:
    
    def __init__(self, onnx_path) -> None:
        self.onnx_path = onnx_path
        self.cnn_session = ort.InferenceSession(onnx_path[0])
        self.encoder_session = ort.InferenceSession(onnx_path[1])
        self.decoder_session = ort.InferenceSession(onnx_path[2])
        
        self.config = Cfg.load_config_from_name('vgg_seq2seq')
        self.config['weights'] =  '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/text_recognition/vietocr/weights/transformerocr.pth'
        self.config['cnn']['pretrained']=False
        self.config['device'] = 'cpu'
        
        _, self.vocab = build_model(self.config)
        
        self.transforms  = transforms.Compose([
            transforms.Resize((32, 500)),
            transforms.ToTensor(),
        ])
        
    def predict_greedy(self, image):
        
        max_seq_length=128
        sos_token=1
        eos_token=2
        
        
        image = self.transforms(image)
        image = image.unsqueeze(0)
        image = np.asarray(image)
        
        cnn_input = {self.cnn_session.get_inputs()[0].name: image}
        
        src = self.cnn_session.run(None, cnn_input)
        
        encoder_input = {self.encoder_session.get_inputs()[0].name: src[0]}
        encoder_outputs, hidden = self.encoder_session.run(None, encoder_input)
        translated_sentence = [[sos_token]*len(image)]

        max_length = 0
        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
        
            tgt_inp = translated_sentence
        
            decoder_input = {
                self.decoder_session.get_inputs()[0].name: tgt_inp[-1],
                self.decoder_session.get_inputs()[1].name: hidden,
                self.decoder_session.get_inputs()[2].name: encoder_outputs
            }
            
            output, hidden, _ = self.decoder_session.run(None, decoder_input)
            output = torch.tensor(output)
            output = output.unsqueeze(1)
            output = torch.softmax(output, dim=-1)
            
            _, indices  = torch.topk(output, 5)
            
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            
            translated_sentence.append(indices)   
            max_length += 1

            del output
                
        translated_sentence = np.asarray(translated_sentence).T
        s = self.vocab.decode(translated_sentence[0].tolist())
        return s
    
    def beamsearch(self, memory, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):    
        beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token, end_token_id=eos_token)
        hidden, encoder_outputs = memory
        hidden = torch.tensor(hidden)
        encoder_outputs = torch.tensor(encoder_outputs)
        hidden = hidden.repeat(beam_size, 1)
        encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)
        memory = (hidden, encoder_outputs)
        
        hidden = np.array(hidden)
        encoder_outputs = np.array(encoder_outputs)

        for _ in range(max_seq_length):
            
            tgt_inp = beam.get_current_state().transpose(0,1).to('cpu')  # TxN
            
            decoder_outputs, hidden, _ = self.decoder_session.run(None, {
                self.decoder_session.get_inputs()[0].name: np.array(tgt_inp)[-1],
                self.decoder_session.get_inputs()[1].name: hidden,
                self.decoder_session.get_inputs()[2].name: encoder_outputs
            })
            
            decoder_outputs = torch.tensor(decoder_outputs)
            decoder_outputs = decoder_outputs.unsqueeze(1)
            log_prob = log_softmax(decoder_outputs[:,-1, :].squeeze(0), dim=-1)
            beam.advance(log_prob)
            
            if beam.done():
                break
                
        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

        return [1] + [int(i) for i in hypothesises[0][:-1]]
    
    
    def predict(self, image):

        image = self.transforms(image)
        image = image.unsqueeze(0)
        image = np.asarray(image)
        
        beam_size=4
        candidates=1
        max_seq_length=128
        sos_token=1
        eos_token=2

        src = self.cnn_session.run(None, {self.cnn_session.get_inputs()[0].name: image})[0]
        encoder_outputs, hidden = self.encoder_session.run(None, {self.encoder_session.get_inputs()[0].name: src})
        memory = (hidden, encoder_outputs)
        sent = self.beamsearch(memory, beam_size, candidates, max_seq_length, sos_token, eos_token)
        sent = self.vocab.decode(sent)

        return sent
        

def translate_onnx(image_path, onnx_path, max_seq_length=128, sos_token=1, eos_token=2):
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['weights'] =  '/home/huycq/OCR/Onnx/vietocr/weights/transformerocr.pth'
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    
    _, vocab = build_model(config)
    cnn_session = ort.InferenceSession(onnx_path[0])
    encoder_session = ort.InferenceSession(onnx_path[1])
    decoder_session = ort.InferenceSession(onnx_path[2])
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 475)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    image = np.asarray(image)
    
    
    cnn_input = {cnn_session.get_inputs()[0].name: image}
    
    src = cnn_session.run(None, cnn_input)
    
    encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
    encoder_outputs, hidden = encoder_session.run(None, encoder_input)
    translated_sentence = [[sos_token]*len(image)]
    
    char_probs = [[1]*len(image)]
    max_length = 0
    while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
    
        tgt_inp = translated_sentence
        
        decoder_input = {
            decoder_session.get_inputs()[0].name: tgt_inp[-1],
            decoder_session.get_inputs()[1].name: hidden,
            decoder_session.get_inputs()[2].name: encoder_outputs
        }
        
        output, hidden, _ = decoder_session.run(None, decoder_input)
        
        output = torch.tensor(output)
        
        output = output.unsqueeze(1)
        
        
        output = torch.softmax(output, dim=-1)
        values, indices  = torch.topk(output, 5)
        
        indices = indices[:, -1, 0]
        indices = indices.tolist()
        
        values = values[:, -1, 0]
        values = values.tolist()
        char_probs.append(values)

        translated_sentence.append(indices)   
        max_length += 1

        del output
            
    translated_sentence = np.asarray(translated_sentence).T
    
    char_probs = np.asarray(char_probs).T
    char_probs = np.multiply(char_probs, translated_sentence>3)
    char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)
    
    
    #print(list(translated_sentence)[0])
    s = vocab.decode(translated_sentence[0].tolist())
    return s, char_probs


if __name__ == '__main__':
    model_path = '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/text_recognition/vietocr/weights/transformerocr.pth'
    onnx_path = [
        '/home/huycq/OCR/Onnx/vietocr/weights/vietocr_cnn_1.onnx',
        '/home/huycq/OCR/Onnx/vietocr/weights/vietocr_encoder_1.onnx',
        '/home/huycq/OCR/Onnx/vietocr/weights/vietocr_decoder_1.onnx',
    ]
    exporter = VietOcrExporter(model_path, onnx_path, image_size=(1, 1, 32, 475))
    #inputs = torch.rand(1, 1, 32, 475)
    #image_path = "/home/huycq/OCR/Onnx/vietocr/image/samples.png"
    #translate_onnx(image_path, onnx_path, max_seq_length=128, sos_token=1, eos_token=2)
    #eport_vietocr_seq2seq(model_path, onnx_path, inputs)
    
    #cnn_session = ort.InferenceSession(onnx_path[0])
    exporter.eport_vietocr_seq2seq()