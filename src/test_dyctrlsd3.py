import torch
import argparse
import numpy as np
from diffusers.utils import load_image
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

# from models.pipeline_dycontrolnet_sd3 import StableDiffusion3ControlNetPipeline
from models.inf.pipeline_dycontrolnet_sd3_inf import StableDiffusion3ControlNetPipeline

# from diffusers import SD3Transformer2DModel
from unets.inf.transformer_sd3_inf import SD3Transformer2DModel

# from models.dycontrolnet_sd3 import SD3ControlNetModel
from models.inf.dycontrolnet_sd3_inf import SD3ControlNetModel


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
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
        default=28,
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=int,
        default=1,
    )
    parser.add_argument(
        "guidance_scale",
        type=int,
        default=7.0,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--save_generated_image",
        type=str,
        default="image/generated_image",
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

    # base_model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    # controlnet_path = "ckpt/checkpoint"
    base_model_path = args.pretrained_model_name_or_path
    controlnet_path = args.controlnet_model_name_or_path

    # load pipeline
    transformer = SD3Transformer2DModel.from_pretrained(
        base_model_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )

    # initialize from original transformer
    controlnet = SD3ControlNetModel.from_transformer(
        transformer, num_extra_conditioning_channels=0, num_layers=23
    )
    # load weight from ckpt trained by deepspeed zero stage2
    controlnet = load_state_dict_from_zero_checkpoint(
        controlnet, controlnet_path
    )

    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, use_safetensors=None,
    )
    pipe.to("cuda", torch.float16)

    # config
    control_image = load_image(args.condition_image)
    prompt = args.caption

    image = pipe(
        prompt,
        # negative_prompt=n_prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.8,
        guidance_scale=4.5,
        height=512,
        width=512,
    ).images[0]
    # image.save('images/image1.jpg')
    image.save(args.save_generated_image)


if __name__ == "__main__":
    args = parse_args()
    main(args)
