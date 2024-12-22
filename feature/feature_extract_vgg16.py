from utils.extractor import Extractor
from models.vgg import vgg16
import torch as t
from torch import nn
import os

# The script to extract sketches or photos' features using the trained model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_set_root = '../data1/zzl/dataset/sketch-triplet-train'
test_set_root = '../data1/zzl/dataset/sketch-triplet-test'

train_photo_root = '../data1/zzl/dataset/photo-train'
test_photo_root = '../data1/zzl/dataset/photo-test'

# The trained model root for vgg
SKETCH_VGG = '../data1/zzl/model/caffe2torch/vgg_triplet_loss/sketch/sketch_vgg16_5.pth'
PHOTO_VGG = '../data1/zzl/model/caffe2torch/vgg_triplet_loss/photo/photo_vgg16_5.pth'

device = 'cuda:1'

if __name__ == '__main__':
    vgg16 = vgg16(pretrained=False)
    vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
    ext = Extractor(e_model=vgg16, vis=False, dataloader=True)

    vgg16.load_state_dict(t.load(PHOTO_VGG, map_location=t.device('cpu'), weights_only=True), strict=False)
    vgg16.cuda()
    ext.reload_model(vgg16)
    photo_feature = ext.extract(test_photo_root, 'photo-vgg16-epoch.pkl')

    vgg16.load_state_dict(t.load(SKETCH_VGG, map_location=t.device('cpu'), weights_only=True), strict=False)
    vgg16.cuda()
    ext.reload_model(vgg16)
    sketch_feature = ext.extract(test_set_root, 'sketch-vgg16-epoch.pkl')

