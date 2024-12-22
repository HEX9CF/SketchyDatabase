from utils.extractor import Extractor
from models.sketch_resnet import resnet34
import torch as t
from torch import nn
import os

# The script to extract sketches or photos' features using the trained model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_set_root = '../data1/zzl/dataset/sketch-triplet-train'
test_set_root = '../data1/zzl/dataset/sketch-triplet-test'

train_photo_root = '../data1/zzl/dataset/photo-train'
test_photo_root = '../data1/zzl/dataset/photo-test'

# The trained model root for resnet
SKETCH_RESNET = '../data1/zzl/model/caffe2torch/resnet34_triplet_loss/sketch/sketch_resnet34_10.pth'
PHOTO_RESNET = '../data1/zzl/model/caffe2torch/resnet34_triplet_loss/photo/photo_resnet34_10.pth'

FINE_TUNE_RESNET = 'data1/zzl/model/caffe2torch/fine_tune/model_270.pth'

device = 'cuda:1'

if __name__ == '__main__':
    resnet = resnet34()
    resnet.fc = nn.Linear(in_features=512 * 1, out_features=125)  # Adjust the output features as needed
    ext = Extractor(e_model=resnet, vis=False, dataloader=True)

    resnet.load_state_dict(t.load(PHOTO_RESNET, map_location=t.device('cpu'), weights_only=True), strict=False)
    resnet.cuda()
    ext.reload_model(resnet)
    photo_feature = ext.extract(test_photo_root, 'photo-resnet34-epoch.pkl')

    resnet.load_state_dict(t.load(SKETCH_RESNET, map_location=t.device('cpu'), weights_only=True), strict=False)
    resnet.cuda()
    ext.reload_model(resnet)
    sketch_feature = ext.extract(test_set_root, 'sketch-resnet34-epoch.pkl')

