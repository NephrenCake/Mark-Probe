from typing import Union

import torch

from steganography.models.MPNet import MPEncoder, MPDecoder


def model_import(model_path, model_name, device, msg_size=96, img_size=448):
    """
    import encoder or decoder
    """
    if model_name == "Encoder":
        model = MPEncoder(msg_size=msg_size, img_size=img_size).to(device)
    elif model_name == "Decoder":
        model = MPDecoder(msg_size=msg_size, img_size=img_size, decoder_type="conv", has_stn=True).to(device)
    else:
        raise Exception("error! no {} model".format(model_name))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint[model_name])
    model.eval()

    return model


def get_device(device):
    return torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
