import numpy as np
import torch
from src.utils import mul


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def _convert_list_to_tensor(list_convert, dim=0):
    if len(list_convert):
        result = torch.stack(list_convert, dim=dim)
    else:
        result = None
    return result


def convert_list_to_tensor(list_convert, dim=0):
    if len(list_convert):
        list = [torch.tensor(i) for i in list_convert]
        result = torch.stack(list, dim=dim)
        # result = torch.stack(list_convert, dim=dim)
    else:
        result = None
    return result


def costloss_block(mask):
    alpha = 0.05  # hyperparameter to adjust the influence between mse and cost loss
    sparsity = 0.5  # hyperparameter to limit the computation consumption of additional net
    # (layers, bs, mask) -> (bs, layers, mask)
    mask = convert_list_to_tensor(mask, dim=0)
    mask_bs = mask.permute(1, 0, 2)

    ratio_bs = []
    for mask in mask_bs:
        mask = mask.flatten()
        ratio = torch.sum(mask>0).item() / len(mask)
        ratio_bs.append(ratio)

    # ratio_bs = convert_list_to_tensor(ratio_bs, dim=0)
    # loss = ratio_bs.mean()
    loss = np.mean(ratio_bs)
    loss = alpha * ((loss - sparsity)**2)
    # loss = args.alpha * abs((loss - args.sparse))

    return loss


def costloss(mask):
    alpha = 0.05  # hyperparameter to adjust the influence between mse and cost loss
    sparsity = 0.5  # hyperparameter to limit the computation consumption of additional net

    flops_unet = 283567882240.0
    block_unet = np.array([47185920.0, 15992340480.0, 943718400.0, 15992340480.0, 14345011200.0, 943718400.0, 16022732800.0, 14417592320.0, 943718400.0, 16095313920.0, 1889075200.0, 1889075200.0, 6027673600.0, 1889075200.0, 1889075200.0, 1889075200.0, 3774873600.0, 16095313920.0, 16095313920.0, 16095313920.0, 15099494400.0, 20636467200.0, 16022732800.0, 16022732800.0, 15099494400.0, 20606074880.0, 15992340480.0, 15992340480.0])

    # (layers, bs, mask) -> (bs, layers, mask)
    mask = convert_list_to_tensor(mask, dim=0)
    mask_bs = mask.permute(1, 0, 2)
    # mask_bs = np.array(mask)
    # mask_bs = mask_bs.transpose((1, 0, 2))

    ratio_bs = []
    for mask in mask_bs:
        mask = mask.flatten()
        # ratio = torch.sum(mask>0).item() / len(mask)
        flops = mul(mask, block_unet)
        ratio = flops.sum() / flops_unet
        ratio_bs.append(ratio)

    loss = np.mean(ratio_bs)
    loss = alpha * ((loss - sparsity)**2)
    # loss = args.alpha * abs((loss - args.sparse))

    return loss
