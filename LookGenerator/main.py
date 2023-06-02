import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from LookGenerator.datasets.utils import prepare_image_for_segmentation, clean_image_by_mask

import torch
from torchvision import transforms

import os
from PIL import Image

from typing import Dict

from LookGenerator.networks.encoder_decoder import EncoderDecoder
from LookGenerator.networks.segmentation import UNet
from LookGenerator.networks.utils import load_model
from LookGenerator.networks.clothes_feature_extractor import ClothAutoencoder
from LookGenerator.networks.bpgm.model.models import BPGM
from LookGenerator.datasets import transforms as custom_transforms

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer


def load_models(weights_dir: str, device=None) -> Dict:
    """
        Params:
                weights_dir: str:
                    Directory with "segmentation.pt" and "encoder_decoder.pt" weights. Weights for unsampler will be downloaded
        Returns:
            Dict of all pretrained models. 
            Keys:
                - "segmentation" 
                - "encoder_decoder"  
                - "unsampler"


    """


    models = dict()

    monochannel_segmentation_model = UNet(in_channels=3, out_channels=1,)
    multichannel_segmentation_model = UNet(in_channels=3, out_channels=12,
            final_activation=nn.Softmax(dim=1), 
            features=(16, 32, 64, 128, 256, 512))

    cfs = ClothAutoencoder(in_channels=3, out_channels=3,
            features=(8, 16, 32, 64),
            latent_dim_size=128,
            encoder_activation_func=nn.LeakyReLU(),
            decoder_activation_func=nn.ReLU())

    cfs = load_model(cfs, os.path.join(weights_dir, "cfs.pt"))

    encoder_decoder_model = EncoderDecoder(clothes_feature_extractor=cfs, in_channels=6, out_channels=3)
    
    tps = BPGM(in_channels=12)


    models["monosegmentation"] = load_model(monochannel_segmentation_model,
        os.path.join(weights_dir, "monosegmentation.pt"))
    models["multisegmentation"] = load_model(multichannel_segmentation_model,
        os.path.join(weights_dir, "multisegmentation.pt"))
    models["encoder_decoder"] = load_model(encoder_decoder_model,  os.path.join(weights_dir, "encoder_decoder.pt"))
    models["tps"] = load_model(tps, os.path.join(weights_dir,  "tps.pt"))
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    model_path = load_file_from_url(model_url, weights_dir, progress=True, file_name=None)

    models["unsampler"] = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
)
    

    return models



def process_image(human_image: Image, clothes_image: Image, models: Dict,
        upscale=False)-> Image:

    """
        Params:
            human_image: Source image of human
            clothes_image: Source image of clothes
            models: Dictitionary of all pretrained models. 
                    Keys:
                    - "segmentation" 
                    - "encoder_decoder"  
        Return:
            Human image with new clothes.
    """


    """"""

    from torchvision.utils import save_image
    
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    transform_human = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.Normalize(std=[0.5, 0.5, 0.5],
            mean=[0.5, 0.5, 0.5])
    ])

    transform_clothes = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.Normalize(std=[0.5, 0.5, 0.5],
            mean=[0.25, 0.25, 0.25])
    ])

    transform_segmented = transforms.Compose([
        custom_transforms.ThresholdTransform(threshold=0.5),
    ])

    human = to_tensor(human_image).unsqueeze(0)
    clothes = to_tensor(clothes_image).unsqueeze(0)

    human = transform_human(human)
    clothes = transform_clothes(clothes)

   

    monosegmented = models["monosegmentation"](human)
    monosegmented = transform_segmented(monosegmented)
    
    monosegmented = torch.tensor(monosegmented, dtype=torch.bool)

    without_torso = human * (~monosegmented) + monosegmented


    multisegmented = models["multisegmentation"](human)
    multisegmented = transform_segmented(multisegmented)

    theta = models["tps"](multisegmented, clothes)
    warped = F.grid_sample(clothes, theta, padding_mode='border',
            align_corners=True)

    torso_mask = torch.tensor(multisegmented[:, 8], dtype=torch.bool)

    warped = warped * torso_mask
    torso_mask = ~torso_mask
    person = without_torso * torso_mask + warped


    encoder_decoder_input = torch.cat((person, clothes), axis=1)

    encoded_image = models["encoder_decoder"](encoder_decoder_input)

    if upscale:
        super_resolution, _ = models["unsampler"].enhance(encoded_image.squeeze(0).permute(1, 2, 0).detach().numpy())
        result_image = super_resolution

    else:
        result_image = encoded_image
    
    result_image = to_pil(result_image[0])

    return result_image
    


    

def process_images_in_folder(model, image_folder: str,  clothes_folder: str, dist: str): 
    human_image_list = os.listdir(image_folder)
    clothes_list = os.listdir(clothes_folder)

    for human_path in human_image_list:
        for clothes_path in clothes_list:
            human_image = Image.open(os.path.join(image_folder, human_path))
            clothes_image = Image.open(os.path.join(clothes_folder, clothes_path))
            
            processed_image = process_image(human_image=human_image, clothes_image=clothes_image, model=model)
            dist_dir = os.path.join(dist, f"{human_path.split('.')[0]}_{clothes_path.split('.')[0]}")
            os.mkdir(dist_dir)
            human_image.save(os.path.join(dist_dir, "human.jpg"))
            clothes_image.save(os.path.join(dist_dir, "clothes.jpg"))
            processed_image.save(os.path.join(dist_dir, "human.jpg"))

