import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import accuracy as ac

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = ac(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(weights='DEFAULT')
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 7)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


loaded_model=ResNet()
loaded_model.load_state_dict(torch.load('resnet_model.pth'))
device = get_default_device()
loaded_model=to_device(loaded_model,device)
loaded_model.eval()


import matplotlib.pyplot as plt
import torchvision.transforms as transforms
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

from PIL import Image

classes=['cardboard', 'ewaste', 'glass', 'metal', 'paper', 'plastic', 'trash']


import requests
from io import BytesIO

def predict_url(url):
    response=requests.get(url)
    im=Image.open(BytesIO(response.content))
    img=transformations(im).unsqueeze(0)
    img=to_device(img,device)
    _,pred=torch.max(loaded_model(img),1)
    
    return classes[pred[0].item()]


@app.route('/')
def home():
    return "Welcome to the Image Classification API!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    urls = data['urls']

    prediction=[]
    for url in urls:
        pred = predict_url(url)
        prediction.append(pred)

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)