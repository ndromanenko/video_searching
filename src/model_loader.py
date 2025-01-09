import torch
from src.EncDecoder import EncDecCTCModel

def load_ctc_model(config_path, weights_path, device="cpu"):
    model = EncDecCTCModel.from_config_file(config_path)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model = model.to(device)
    model = model.half()
    return torch.compile(model) 