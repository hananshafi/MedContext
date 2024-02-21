# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import numpy as np
import torch
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils import get_loader
from monai.metrics import compute_hausdorff_distance
from monai.inferers import sliding_window_inference
from medpy import metric
import monai

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=32, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")

def hd95(pred,gt):
    pred[pred>0]=1
    gt[gt>0] =1
    if pred.sum()>0 and gt.sum()>0:
        hd95 = metric.binary.hd95(pred,gt)
        return hd95
    else:
        return 0


def asd(pred,gt):
    pred[pred>0]=1
    gt[gt>0] =1
    if pred.sum()>0 and gt.sum()>0:
        eval_asd = metric.binary.asd(pred,gt)
        return eval_asd
    else:
        return 0


def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )

        model_dict = torch.load(os.path.join(pretrained_pth))['state_dict']
        #model_dict = {k.replace(k,'vit.'+k):v for k,v in model_dict.items() if not 'encoder}

        if 'baseline' not in args.pretrained_dir:
            model_dict_new = {}
            keys_retain = ['encoder','decoder', 'out.']
            for k, v in model_dict.items():
                if not any(substring in k for substring in keys_retain):
                # if 'encoder' not in k:
                    model_dict_new['vit.'+k] = v
                else:
                    model_dict_new[k] = v
            msg = model.load_state_dict(model_dict_new, strict=False)
            print(msg)
            print("Use pretrained weights")
        else:
            msg = model.load_state_dict(model_dict, strict=False)
            print(msg)
            print("Use pretrained weights")

        # model_dict = torch.load(pretrained_pth)
        # model.load_state_dict(model_dict['state_dict'])
    model.eval()
    model.to(device)
    #organ_dice_dict = {1:[],2:[],3:[],4:[],6:[],7:[],8:[],11:[]}
    #organ_hd_dict = {1:[],2:[],3:[],4:[],6:[],7:[],8:[],11:[]}
    organ_dice_dict = {1:[],2:[],3:[],4:[],5: [],6:[],7:[],8:[],9:[], 10:[], 11:[], 12:[], 13:[]}
    organ_hd_dict = {1:[],2:[],3:[],4:[],5: [],6:[],7:[],8:[],9:[], 10:[], 11:[], 12:[], 13:[]}
    organ_asd_dict = {1:[],2:[],3:[],4:[],5: [],6:[],7:[],8:[],9:[], 10:[], 11:[], 12:[], 13:[]}
    with torch.no_grad():
        dice_list_case = []
        hdf_case = []
        asd_case =[]
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            print("Inference on case {}".format(img_name))
            #monai.transforms.SaveImage(output_dir=args.pretrained_dir, resample=True, output_postfix="input_test-"+img_name)(val_inputs[0])
            #monai.transforms.SaveImage(output_dir=args.pretrained_dir, resample=True, output_postfix="label_test-"+img_name)(val_labels[0])
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            monai.transforms.SaveImage(output_dir=args.pretrained_dir, resample=False, output_postfix="3dmsr_pred_test-"+img_name)(val_outputs)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            hd_list_sub = []
            asd_list_sub = []
            organ_dice_list = []
            hd_8 = []
            asd_8 = []
            for i in range(1, 14):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                organ_hd = hd95(val_outputs[0] == i, val_labels[0] == i)
                organ_asd = asd(val_outputs[0] == i, val_labels[0] == i)
                dice_list_sub.append(organ_Dice)
                hd_list_sub.append(organ_hd)
                asd_list_sub.append(organ_asd)
                if i in organ_dice_dict.keys():
                	organ_dice_dict[i].append(organ_Dice)
                	organ_hd_dict[i].append(organ_hd)
                	organ_asd_dict[i].append(organ_asd)
                	
            mean_dice = np.mean(dice_list_sub)
            mean_hdf5 = np.mean(hd_list_sub)
            mean_asd = np.mean(asd_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            print("Mean Organ Hausdorff: {}".format(mean_hdf5))
            print("Mean Organ ASD: {}".format(mean_asd))
            dice_list_case.append(mean_dice)
            hdf_case.append(mean_hdf5)
            asd_case.append(mean_asd)
        organ_dice_dict = {key:np.array(v).mean() for key, v in organ_dice_dict.items()}
        organ_hd_dict = {key:np.array(v).mean() for key, v in organ_hd_dict.items()}
        organ_asd_dict = {key:np.array(v).mean() for key, v in organ_asd_dict.items()}
        for key, value  in organ_dice_dict.items():
        	print('Dice organ- ' + str(key) , value)
        for key, value  in organ_hd_dict.items():
        	print('hd95 organ- ' + str(key) , value)
        for key, value  in organ_asd_dict.items():
        	print('asd organ- ' + str(key) , value)

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        print("Overall Mean Hausdorff: {}".format(np.mean(hdf_case)))
        print("Overall Mean ASD: {}".format(np.mean(asd_case)))
        print("Overall Mean Dice for 8 organs: {}".format(np.array(list(organ_dice_dict.values())).mean()))
        print("Overall Mean Hausdorff for 8 organs: {}".format(np.array(list(organ_hd_dict.values())).mean()))
        print("Overall Mean ASD for 8 organs: {}".format(np.array(list(organ_asd_dict.values())).mean()))


if __name__ == "__main__":
    main()