import torch


# snapshot = torch.load("../pretrained/df_pt_sobel_loss2.pth")
snapshot = torch.load("../logs/checkpoints/ckpt-best.pth")
# for k in list(snapshot["model"].keys()):
#     if "context_network" in k:
#         snapshot["model"].pop(k)
# torch.save(snapshot, "../logs/checkpoints/ckpt-best.pth")
print(snapshot["metrics_loss"])