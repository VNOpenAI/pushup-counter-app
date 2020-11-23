import cv2
import numpy as np
import torch
import torch.nn as nn
from resnest.torch import resnest50, resnest101, resnest200, resnest269, resnest50_fast_4s2x40d
from .resnest_head import ResNeSt_head
import torchvision.transforms as transforms
from models.ttt_net import TTTnet
import onnx
from tensorflow.keras.models import load_model
import onnxruntime
from onnx2keras import onnx_to_keras

class KeypointHeatmapModel():

    def __init__(self, checkpoint, img_size=(225,225)):

        torch.set_num_threads(8)

        # Build model and load weight
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
        self.model = self.build_model()
        checkpoint = torch.load(checkpoint, map_location=map_location)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(checkpoint)
        # Create the right input shape (e.g. for an image)
        dummy_input = torch.randn(1, 3, img_size[1], img_size[0])
        torch.onnx.export(self.model, dummy_input, "tmp.onnx")
        onnx_model = onnx.load('tmp.onnx')
        k_model = onnx_to_keras(onnx_model, ['input'])

        self.img_size = img_size

        # Init transformation
        self.trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(img_size[:2]),
                                    transforms.ToTensor()])

        self.ort_session = onnxruntime.InferenceSession("tmp.onnx")


    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def build_model(self):
        model = TTTnet(21, 3)
        return model


    def predict(self, origin_img):
        # with torch.no_grad() as tng:
        oh, ow = origin_img.shape[:2]
        img = self.trans(origin_img)
        img = torch.unsqueeze(img, 0)

        ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(img)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        preds = ort_outs[0][0]

        # preds = preds.cpu().numpy()[0]
        coor_x = []
        coor_y = []
        for i,pred in enumerate(preds[:7]):
            cx = np.argmax(pred)%pred.shape[0]
            cy = np.argmax(pred)//pred.shape[0]
            ovx = preds[i+7][cy,cx]*15
            ovy = preds[i+14][cy,cx]*15
            coor_x.append(int((cx*15+ovx)*ow/self.img_size[1]))
            coor_y.append(int((cy*15+ovy)*oh/self.img_size[0]))
        preds = np.vstack([coor_x, coor_y]).T
        return preds