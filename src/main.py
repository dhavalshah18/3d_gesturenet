import torch
import torch.utils.data as data
from src.data import GestureData
from src.network import GestureNetFCN
from src.solver import Solver


def main():
    root = "/home/dhaval/I6_Gestures"
    train_data = GestureData(root, mode="train")
    val_data = GestureData(root, mode="val")
    train_loader = data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=5)
    val_loader = data.DataLoader(val_data, batch_size=10, shuffle=False, num_workers=5)

    model = GestureNetFCN()
    model.cuda()

    optim_args_SGD = {"lr": 1e-3, "weight_decay": 0.0, "momentum": 0.9, "nesterov": True}
    solver = Solver(optim_args=optim_args_SGD, optim=torch.optim.SGD)

    solver.train(model, train_loader, val_loader, log_nth=5, num_epochs=20)

    name = "gesturenet.pth"
    torch.save(model.state_dict(), name)


if __name__ == "__main__":
    main()
