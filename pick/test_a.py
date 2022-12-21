import torch
from parse_config import ConfigParser

if __name__ == '__main__':
    a = torch.load('/home/huycq/OCR/invoice_kie/pick/weights/model_best.pth', map_location='cpu')
    print(a['config'])
