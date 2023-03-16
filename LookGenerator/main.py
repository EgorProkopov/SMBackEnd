from LookGenerator.datasets.utils import prepare_image_for_segmentation, clean_image_by_mask, to_image_from_decoder

import torch
from torchvision import transforms

from PIL import Image

from typing import Dict

from LookGenerator.networks.encoder_decoder import EncoderDecoder
from LookGenerator.networks.segmentation import UNet
from LookGenerator.networks.utils import load_model


def load_models(segmentation_weights_path: str, encoder_decoder_weights_path: str) -> Dict:
    """
        Params:
            segmentation_weights_path: str. Path to saved weights of Segmentation model.
            encoder_decoder_weights_path: str. Path to saved weights of Encoder-Decoder model.
        Returns:
            Dict of all pretrained models. 
            Keys:
                - "segmentation" 
                - "encoder_decoder"  
                -


    """


    models = dict()

    segmentation_model = UNet(in_channels=3, out_channels=1)
    encoder_decoder_model = EncoderDecoder(in_channels=6, out_channels=3)

    models["segmentation"] = load_model(segmentation_model, segmentation_weights_path)
    models["encoder_decoder"] = load_model(encoder_decoder_model, encoder_decoder_weights_path)

    return models



def process_image(human_image: Image, clothes_image: Image, models: Dict )-> Image:

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


    transform_human = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25]
        )
    ])

    transform_clothes = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 192)),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.25, 0.25, 0.25]
        )
    ])


    human = prepare_image_for_segmentation(human_image, transform=transform_human)

    segmented = models["segmentation"](human)
    
    without_body = clean_image_by_mask(human, segmented)

    clothes = transform_clothes(clothes_image)

    encoder_decoder_input = torch.cat((without_body, clothes), axis=0)

    result_tensor = models["encoder_decoder"](encoder_decoder_input)

    result_image = to_image_from_decoder(result_tensor)

    return result_image
    


    
    