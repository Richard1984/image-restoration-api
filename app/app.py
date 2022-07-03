import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image
from torch import nn
from torchvision import models

from .model import AutoEncoder

app = Flask(__name__)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = AutoEncoder()
dirname = os.path.dirname(__file__)
model.load_state_dict(torch.load(os.path.join(
    dirname, 'models/cpuModel2.pth'), map_location=device))
model.to(device)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(128),
                                        transforms.CenterCrop(128),
                                        transforms.ToTensor(),
                                        transforms.Lambda(
                                            lambda x: x.to(device)),
                                        # transforms.Normalize(
                                        #     [0.485, 0.456, 0.406],
                                        #     [0.229, 0.224, 0.225])
                                        ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def restore_image(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    return np.transpose(outputs[0].cpu().detach().numpy(), (1, 2, 0))


@app.route('/restore', methods=['POST'])
def restore():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        output_image = restore_image(image_bytes=img_bytes)
        # plt.imshow(output_image)
        # plt.show()

        # convert numpy array to PIL Image
        img = Image.fromarray((output_image * 256).astype('uint8'))

        # create file-object in memory
        file_object = io.BytesIO()

        # write PNG in file-object
        img.save(file_object, 'PNG')

        # move to beginning of file so `send_file()` it will read from start
        file_object.seek(0)

        return send_file(file_object, mimetype='image/PNG')


if __name__ == '__main__':
    app.run()
