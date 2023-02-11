from model.df_net import DFNet, CDFNet
import torch
from utils.config_loader import CFG


cfg_file_path = "../experiments/dfnet_L50.yml"
cfg = CFG.read_from_yaml(cfg_file_path)
print("Load configuration from {}".format(cfg_file_path))
model = CDFNet(cfg)
if cfg.general.pretrained_weight:
    pretrained_model_path = cfg.general.pretrained_weight
    print("load weight from:{}".format(pretrained_model_path))
    snapshot = torch.load(pretrained_model_path)
    model.load_state_dict(snapshot.pop("model"), strict=False)
else:
    print("do not use pretrained weight")
    model.init_weights()

device = torch.device("cuda:0")
model = model.to(device)
model.eval()

export_module = torch.jit.script(model)
torch.jit.save(export_module, "../pretrained/df_L50_cpp_gpu.pth")
