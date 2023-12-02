
import streamlit as st
import subprocess
subprocess.call(["pip", "install", "-r", "./requirements.txt"])
import torch
from skimage.io import imread as imread
from sklearn.utils import resample
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

with open('./data/allergies.txt', 'r') as file:
    allergies = [line.strip() for line in file]
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes=len(allergies)):
        super(FineTunedResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def Model():
    return FineTunedResNet()

model = Model()
model.load_state_dict(torch.load('./FineTunedResNet_allergies_model.pth', map_location='cpu'))
model.cpu()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("lxhyylian-Food Allergies Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred_title = ', '.join(['{} ({:2.1f}%)'.format(allergies[j], 100 * torch.sigmoid(output[0, j]).item())
                            for j, v in enumerate(output.squeeze())
                            if torch.sigmoid(v) > 0.5])

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Predicted allergies in food: ", pred_title)
