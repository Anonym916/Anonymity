import cv2
import torch
import numpy as np
import argparse

import accelerate
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPTextModel
from diffusers.utils import load_image
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from models.dycontrolnet import ControlNetModel
from models.pipline_dycontrolnet import StableDiffusionControlNetPipeline
from unets.unet2d import UNet2DConditionModelDy as ctrlnet
from unets.backbone import UNet2DConditionModelDy as Backbone


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="the pretrained weights of backbone."
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="the weights of controlnet."
    )
    parser.add_argument(
        "--condition_image",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_sampling_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--save_generated_image",
        type=str,
        default="image/generated_image"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):

    # set ramdom seed if needed
    # set_seed(args.seed)

    # please modify it to the address of your conditional image
    # image = load_image("images/example.png")
    # base_model_path = "runwayml/stable-diffusion-v1-5"
    # controlnet_path = "ckpt/checkpoint-xxx/controlnet"
    image = load_image(args.condition_image)
    base_model_path = args.pretrained_model_name_or_path
    controlnet_path = args.controlnet_model_name_or_path

    backbone = Backbone.from_pretrained(
        base_model_path, subfolder="unet", revision=args.revision, variant=args.variant,
    )  # load diffusion backbone

    # init_ctrlnet = ctrlnet.from_pretrained(
    #     base_model_path, subfolder="unet", revision=None, variant=None,
    #     low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
    # )  # torch_dtype=torch.float16, only used to initialize controlnet

    # controlnet = ControlNetModel.from_unet(init_ctrlnet)

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, unet=backbone, safety_checker=None,
    )  # torch_dtype=torch.float16
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    caption = args.caption  # description of the picture.

    image = pipe(caption,
                 image,
                 num_inference_steps=args.num_sampling_steps,
                 ).images[0]
    # image.save('images/generated_img.png')
    image.save(args.save_generated_image)


if __name__ == "__main__":
    args = parse_args()
    main(args)
