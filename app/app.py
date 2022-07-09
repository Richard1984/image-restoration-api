import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image
from torch import nn
from torchvision import models

from .model import AutoEncoder, load_model

app = Flask(__name__)
CORS(app)

image_size = 512

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = AutoEncoder()
dirname = os.path.dirname(__file__)
model = load_model(device=device, model_path=os.path.join(
    dirname, 'models/'), forTraining=False)[0]

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Lambda(
                                            lambda x: x.to(device))
                                        ])
    image = Image.open(io.BytesIO(image_bytes))

    if (len(image.split()) > 3):
        # remove alpha channel
        image.load()  # required for png.split()
        result = Image.new("RGB", image.size, (255, 255, 255))
        result.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return my_transforms(result).unsqueeze(0)

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
