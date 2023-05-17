'''

from LookGenerator.datasets.utils import prepare_image_for_segmentation, clean_image_by_mask, to_image_from_decoder

import torch
from torchvision import transforms

import os
from PIL import Image

from typing import Dict

from LookGenerator.networks.encoder_decoder import EncoderDecoder
from LookGenerator.networks.segmentation import UNet
from LookGenerator.networks.utils import load_model
from LookGenerator.datasets import transforms as custom_transforms

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer


def load_models(weights_dir: str) -> Dict:
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

    segmentation_model = UNet(in_channels=3, out_channels=1)
    encoder_decoder_model = EncoderDecoder(in_channels=6, out_channels=3)

    models["segmentation"] = load_model(segmentation_model, os.path.join(weights_dir, "segmentation.pt"))
    models["encoder_decoder"] = load_model(encoder_decoder_model,  os.path.join(weights_dir, "encoder_decoder.pt"))

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



def process_image(human_image: Image, clothes_image: Image, models: Dict, upscale=True)-> Image:

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


    transform_human = transforms.Compose([
        transforms.Resize((256, 192)),
        custom_transforms.Normalize()
    ])

    transform_clothes = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 192)),
        custom_transforms.Normalize()
    ])

    transform_segmented = transforms.Compose([
        custom_transforms.ThresholdTransform(threshold=0.5),
    ])


    human = prepare_image_for_segmentation(human_image, transform=transform_human)

    segmented = models["segmentation"](human)
    segmented = transform_segmented(segmented)
    
    without_body = clean_image_by_mask(human.squeeze(0), segmented.squeeze(0))
    

    clothes = transform_clothes(clothes_image)


    encoder_decoder_input = torch.cat((without_body.unsqueeze(0), clothes.unsqueeze(0)), axis=1)

    encoded_image = models["encoder_decoder"](encoder_decoder_input)

    if upscale:
        super_resoltion, _ = models["unsampler"].enhance(encoded_image.squeeze(0).permute(1, 2, 0).detach().numpy())
        result_image = to_image_from_decoder(torch.tensor(super_resoltion.transpose(2, 0, 1)).unsqueeze(0))
    else:
        result_image = encoded_image

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

'''
