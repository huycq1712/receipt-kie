model_config = {
    'device': 'cuda',
    'background_remove': {'src': '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/background_remove/yolov5',
                          'model_path': '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/background_remove/weights/model_4.pt',
                          'use_onnx': True,
                          'onnx_path': ['/home/huycq/OCR/Project/KIE/invoice/invoice_kie/background_remove/weights/model_4.onnx']},
    'image_rotation': {'src':'',
                       'model_path': '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/image_rotation/weights/model_4.pt',
                       'use_onnx': True,
                       'onnx_path': ['/home/huycq/OCR/Project/KIE/invoice/invoice_kie/image_rotation/weights/model_4.onnx']},
    'text_detection': {'src':'',
                       'model_path': '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/image_rotation/weights/model_4.pt',
                       'use_onnx': True,
                       'onnx_path': ['/home/huycq/OCR/Project/KIE/invoice/invoice_kie/image_rotation/weights/model_4.onnx']},
    'key_information_extraction': {'src':'',
                                   'model_path': '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/image_rotation/weights/model_4.pt',
                                   'use_onnx': True,
                                   'onnx_path': ['/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/embedding.onnx',
                                                 '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/encoder.onnx',
                                                 '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/graph.onnx',
                                                 '/home/huycq/OCR/Project/KIE/invoice/invoice_kie/pick/weights/decoder.onnx']},
}