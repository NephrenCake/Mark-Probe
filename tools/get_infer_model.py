import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + os.sep + '..')
os.chdir(sys.path[0])

import torch

from steganography.models.MPNet import MPEncoder, MPDecoder

pre_train = "weight/latest-30.pth"

Encoder = MPEncoder().to("cuda")
Decoder = MPDecoder(decoder_type="conv").to("cuda")

checkpoint = torch.load(pre_train, map_location="cuda")

for model in ["Encoder", "Decoder"]:
    net_state_dict = eval(f"{model}.state_dict()")
    for k, v in net_state_dict.items():
        if k in checkpoint[model]:
            net_state_dict[k] = checkpoint[model][k]
        else:
            print(f"{model}: {k} not found")
    eval(f"{model}.load_state_dict(net_state_dict)")

torch.save({"Encoder": Encoder.state_dict(),
            "Decoder": Decoder.state_dict()}, "weight/infer.pth")
