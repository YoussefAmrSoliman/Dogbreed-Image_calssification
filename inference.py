import torch
import os
import logging
import base64
import io
import json
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COMPILED_FILE = 'compiled.pt'
UNCOMPILED_FILE = 'model.pth'
logger = logging.getLogger()

def model_fn(model_dir):
    if os.path.exists(COMPILED_FILE):
        import neopytorch
        model_file = COMPILED_FILE
        neopytorch.config(model_dir=model_dir, neo_runtime=True)
    else:
        model_file = UNCOMPILED_FILE
    model = torch.jit.load(model_file, map_location=device)
    model.eval()
    model.to(device)
    return model 

def predict_fn(image, model):
    return model(image)

def input_fn(request_body, content_type):
    iobytes = io.BytesIO(request_body)
    img = Image.open(iobytes)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    return input_batch.to(device)

