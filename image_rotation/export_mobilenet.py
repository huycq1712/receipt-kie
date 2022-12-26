import onnx
import onnxruntime as rt

from upsidedowndetector import UpsideDowndetector
import torch

def export_mobilenet():
    upside_down_detector = UpsideDowndetector()
    upside_down_detector.load_state_dict(torch.load('/home/huycq/OCR/Project/KIE/invoice/invoice_kie/image_rotation/weights/model_4.pt', map_location='cpu'))
    upside_down_detector.eval()
    
    input_dummy = torch.randn(1, 3, 512, 512)
    input_names = ['image']
    output_names = ['output']
    
    torch.onnx.export(upside_down_detector
                      ,input_dummy,
                      '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/image_rotation/weights/model_4.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      verbose=True)
    
    
if __name__ == '__main__':
    export_mobilenet()