import torch
import torch.utils.data as data
import torch.nn as nn
import os
from data import GestureData
from network import GestureNetFCN
from solver import Solver


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    
    root = "/home/dshah/I6_Gestures"
    
    train_data = GestureData(root, mode="train", z=7)
    val_data = GestureData(root, mode="val", z=7)
    
    train_loader = data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)

    model = nn.DataParallel(GestureNetFCN())
    model = model.cuda()

    optim_args_SGD = {"lr": 1e-3, "weight_decay": 0.005, "momentum": 0.9, "nesterov": True}
    solver = Solver(optim_args=optim_args_SGD, optim=torch.optim.SGD)

    solver.train(model, train_loader, val_loader, log_nth=5, num_epochs=10)

    name = "gesturenet.pth"
    torch.save(model.state_dict(), name)


if __name__ == "__main__":
    main()
