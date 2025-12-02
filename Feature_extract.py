import argparse
from utils import *
import copy
from Data import *
import types
import os
import torch
import torch.nn as nn
from torchvision import models

parser = argparse.ArgumentParser(description="use pretraining net work for feature extract")
parser.add_argument("--dataset",
                    # required=True,
                    dest='dataset',
                    choices=('fashion_mnist',
                             'cifar100'),
                    help="Dataset to train")
parser.add_argument("--model",
                    choices=('efficientnet_b0',
                             'efficientnet_b1',
                             'efficientnet_b2',
                             'efficientnet_b3',
                             'efficientnet_b4'),
                    default='efficientnet_b3')
args = parser.parse_args()

if not os.path.exists(features_save_dir):
    os.mkdir(features_save_dir)

setup_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'cifar100':
    all_img = False
    num_clusters = 100
    img_size = 224
elif args.dataset == 'fashion_mnist':
    all_img = False
    num_clusters = 10
    img_size = 224



from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b4, EfficientNet_B4_Weights
)

def load_efficientnet_model(model_name):
    if model_name == 'efficientnet_b0':
        weight = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weight)
    elif model_name == 'efficientnet_b1':
        weight = EfficientNet_B1_Weights.DEFAULT
        model = efficientnet_b1(weights=weight)
    elif model_name == 'efficientnet_b2':
        weight = EfficientNet_B2_Weights.DEFAULT
        model = efficientnet_b2(weights=weight)
    elif model_name == 'efficientnet_b3':
        weight = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3(weights=weight)
    elif model_name == 'efficientnet_b4':
        weight = EfficientNet_B4_Weights.DEFAULT
        model = efficientnet_b4(weights=weight)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


model = load_efficientnet_model(args.model)
model.to(device)


# add method get_middle_features to EfficientNet model
def get_middle_feature(self, x):
    features = []

    x = self.features[0](x)  # stem

    block_indices = []
    total_blocks = len(self.features) - 1

    if 'b0' in args.model or 'b1' in args.model:
        extract_at = [2, 4, 6]
    else:
        extract_at = [2, 5, 8]
    for i in range(1, len(self.features) - 1):
        x = self.features[i](x)
        if i in extract_at:

            pooled_feature = nn.AdaptiveAvgPool2d(1)(x)
            features.append(pooled_feature.flatten(1))


    x = self.features[-1](x)
    x = self.avgpool(x)
    features.append(x.flatten(1))

    return features


model.get_middle_feature = types.MethodType(get_middle_feature, model)

dataset = load_raw_image(args.dataset, img_size)
dl = DataLoader(dataset, batch_size=128)


model.eval()
features = []
y_true = torch.empty((0,)).to(device)  # 确保标签也在GPU上，保持设备一致

for i, (X, y) in enumerate(dl):
    if args.dataset != 'cifar100' and not all_img and i == 10:
        break
    if args.dataset == 'cifar100' and not all_img and i == 30:
        break
    X = X.to(device)
    y = y.to(device)  # 将标签也移到GPU
    y_true = torch.cat((y_true, y), dim=0)
    with torch.no_grad():
        ls = model.get_middle_feature(X)
    if features == []:
        features = copy.deepcopy(ls)
    else:
        for j in range(len(features)):
            features[j] = torch.cat((features[j], ls[j]), dim=0)


features_cpu = [feat.cpu() for feat in features]
y_true_cpu = y_true.cpu()


torch.save({'data': features_cpu, 'label': y_true_cpu},
           os.path.join(features_save_dir, args.dataset + features_suffix))

print("Files already downloaded and verified")
