import sys
sys.path.append('./deep_person_reid')

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn

from efficientnet_pytorch import EfficientNet  # from https://github.com/lukemelas/EfficientNet-PyTorch

from AIC2018_iamai.ReID.ReID_CNN.models import FeatureResNet  # from https://github.com/cw1204772/AIC2018_iamai

import deep_person_reid.torchreid.models.osnet_ain as osnet_ain  # from https://kaiyangzhou.github.io/deep-person-reid/
import deep_person_reid.torchreid as torchreid  

# How to add a new appearance model : 
# - add settings and get_<model> in this file
# - add the model in main.py's arguments of --feature
# - add the model in the function get_feature() of dataset.py

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Your device is {device}")


# == Settings for VGG19 == 

vgg19_model = torchvision.models.vgg19(pretrained=True)
vgg19_model.classifier = nn.Sequential(*list(vgg19_model.classifier.children())[:-1])  # removes the last layer

vgg19_model = vgg19_model.to(device)
if device == 'cuda':
    vgg19_model = torch.nn.DataParallel(vgg19_model)
    cudnn.benchmark = True

vgg19_model.eval()


# == Settings for Resnet18 ==

resnet18_model = torchvision.models.resnet18(pretrained=True)
resnet18_model.fc = nn.Sequential(*list(resnet18_model.fc.children())[:-1])  # removes the last layer

resnet18_model = resnet18_model.to(device)
if device == 'cuda':
    resnet18_model = torch.nn.DataParallel(resnet18_model)
    cudnn.benchmark = True

resnet18_model.eval()


# == Settings for Densenet121 ==

densenet121_model = torchvision.models.densenet121(pretrained=True)
densenet121_model.classifier = nn.Sequential(*list(densenet121_model.classifier.children())[:-1])  # removes the last layer

densenet121_model = densenet121_model.to(device)
if device == 'cuda':
    densenet121_model = torch.nn.DataParallel(densenet121_model)
    cudnn.benchmark = True

densenet121_model.eval()


# == Settings for EfficientNetB0 ==

# From https://github.com/lukemelas/EfficientNet-PyTorch/
efficientnetb0_model = EfficientNet.from_pretrained('efficientnet-b0')
efficientnetb0_model._fc = nn.Sequential(*list(efficientnetb0_model._fc.children())[:-1])  # removes the last layer

efficientnetb0_model = efficientnetb0_model.to(device)
if device == 'cuda':
    efficientnetb0_model = torch.nn.DataParallel(efficientnetb0_model)
    cudnn.benchmark = True

efficientnetb0_model.eval()


# == Settings for Vehicle-ReID model ==

vehreid_model = FeatureResNet(n_layers=50)

state_dict = torch.load("../weights/model_880_base.ckpt")
for key in list(state_dict.keys()):
    if key.find('fc') != -1 and key.find('fc_c') == -1:
        del state_dict[key]
    elif key.find('fc_c') != -1:
        del state_dict[key]

vehreid_model.load_state_dict(state_dict)

vehreid_model = vehreid_model.to(device)
if device == 'cuda':
    vehreid_model = torch.nn.DataParallel(vehreid_model)
    cudnn.benchmark = True

vehreid_model.eval()


# == Settings for OSNet_AIN_Market model ==

osnetAIN_Market_model = osnet_ain.osnet_ain_x1_0()
torchreid.utils.torchtools.load_pretrained_weights(osnetAIN_Market_model, weight_path="../weights/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth")
if device == 'cuda':
    osnetAIN_Market_model.to(device)
    cudnn.benchmark = True

osnetAIN_Market_model.eval()


# == Creation of features ==


def get_colorHistogram_BW(image):
    """
    Returns a normalized histogram based on a black and white image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vect = cv2.calcHist([gray_image], channels=[0], mask=None, histSize=[32], ranges=[0, 256])

    cv2.normalize(src=vect, dst=vect)

    return vect.flatten()


def get_colorHistogram_BGR(image):
    """
    Returns a normalized histogram based on a color image
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bgr = cv2.split(image)

    b_hist = cv2.calcHist(bgr, channels=[0], mask=None, histSize=[32], ranges=[0, 256])
    g_hist = cv2.calcHist(bgr, channels=[1], mask=None, histSize=[32], ranges=[0, 256])
    r_hist = cv2.calcHist(bgr, channels=[2], mask=None, histSize=[32], ranges=[0, 256])

    cv2.normalize(b_hist, b_hist)
    cv2.normalize(g_hist, g_hist)
    cv2.normalize(r_hist, r_hist)

    vect = np.concatenate((b_hist, g_hist, r_hist), axis=0)

    return vect.flatten()


def get_HOG(image, dataset):
    """
    Returns a histogram of oriented gradients
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if dataset in ['WildTrack', 'MOT20', 'MOT17']:
        winSize = (64, 128)  # default : (64,128) in Dalal 2005
    elif dataset in ['UrbanTracker', 'DETRAC', 'UAVDT']:  # vehicles or all
        winSize = (64, 64)
    else:
        raise ValueError(f"Dataset {dataset} is not recognized !")

    blockSize = (16, 16)  # default : (16,16)
    blockStride = (8, 8)  # default : (8,8)
    cellSize = (8, 8)  # default : (8,8)
    nbins = 9  # default : 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64  # default : 64
    useSignedGradients = True
    hog_cv2 = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                                L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)

    mini_img = cv2.resize(src=image, dsize=winSize)

    vect = hog_cv2.compute(mini_img)

    return vect.flatten()


def get_OSNet_AIN_Market(image):
    """
    Retuns ReID features computed with OSNet
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(cv2.resize(image, (128, 256)))
    image /= 255.0
    image_torch = torch.from_numpy(np.transpose(image))
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_normalise = normalizer(image_torch)
    tensor_normalise.unsqueeze_(0)

    img_tensor = tensor_normalise.to(torch.device("cuda"))
    with torch.no_grad():
        osnetAIN_Market_model.eval()
        vect = osnetAIN_Market_model(img_tensor)

    return vect.cpu().numpy().flatten() 


def get_VehReid(image):
    """
    Returns deep features vectors computed with Resnet18 learned with Reid on vehicles
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resize = np.float32(cv2.resize(image, (224, 224)))
    image_resize /= 255.0
    image_torch = torch.from_numpy(np.transpose(image_resize))
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_normalise = normalizer(image_torch)

    vect = vehreid_model(torch.unsqueeze(tensor_normalise, 0))

    return vect.data.cpu().numpy().flatten()


def get_VGG19(image):
    """
    Retuns deep features vectors computed with VGG19
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resize = np.float32(cv2.resize(image, (224, 224)))
    image_resize /= 255.0
    image_torch = torch.from_numpy(np.transpose(image_resize))
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_normalise = normalizer(image_torch)

    vect = vgg19_model(torch.unsqueeze(tensor_normalise, 0))

    return vect.data.cpu().numpy().flatten()


def get_ResNet18(image):
    """
    Returns deep features vectors computed with Resnet18
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resize = np.float32(cv2.resize(image, (224, 224)))
    image_resize /= 255.0
    image_torch = torch.from_numpy(np.transpose(image_resize))
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_normalise = normalizer(image_torch)

    vect = resnet18_model(torch.unsqueeze(tensor_normalise, 0))

    return vect.data.cpu().numpy().flatten()


def get_Densenet121(image):
    """
    Returns deep features vectors computed with Densenet121
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resize = np.float32(cv2.resize(image, (224, 224)))
    image_resize /= 255.0
    image_torch = torch.from_numpy(np.transpose(image_resize))
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_normalise = normalizer(image_torch)

    vect = densenet121_model(torch.unsqueeze(tensor_normalise, 0))

    return vect.data.cpu().numpy().flatten()


def get_EfficientnetB0(image):
    """
    Returns deep features vectors computed with EfficientNet-B0
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resize = np.float32(cv2.resize(image, (224, 224)))
    image_resize /= 255.0
    image_torch = torch.from_numpy(np.transpose(image_resize))
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_normalise = normalizer(image_torch)

    vect = efficientnetb0_model(torch.unsqueeze(tensor_normalise, 0))

    return vect.data.cpu().numpy().flatten()
