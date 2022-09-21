import torch
import numpy as np


def regAcc(Y, Y_pred, correct, th):
    # Y.to(device='cuda')
    Y_pred_class = torch.zeros_like(Y_pred)
    Y_true = torch.cuda.DoubleTensor((np.log2(2),
                           np.log2(10/2**0+2),
                           np.log2(10/2**1+2),
                           np.log2(10/2**2+2),
                           np.log2(10/2**3+2),
                           np.log2(10/2**4+2),
                           np.log2(10/2**5+2),
                           np.log2(10/2**6+2),
                           np.log2(10/2**7+2),
                           np.log2(10/2**8+2),
                           np.log2(10/2**9+2),
                           np.log2(1),
                           ))
    # Y_true.requires_grad=True

    for i in range(len(Y_pred)):
        Y_pred_class[i] = Y_true[torch.argmin(torch.abs(Y_true-Y_pred[i]))]

    correct += (torch.abs(Y - Y_pred_class) <= th).sum()

    return Y_pred_class, correct


    # torch.abs(torch.diff(Y_true[1:-1]))
    #
    # (torch.mean(torch.abs(torch.diff(Y_true[1:-1]))),torch.std(torch.abs(torch.diff(Y_true[1:-1]))),torch.min(torch.abs(
    #     torch.diff(Y_true[1:-1]))),torch.max(torch.abs(torch.diff(Y_true[1:-1]))))
    # # >> output
    # # (tensor(0.2857, device='cuda:0', dtype=torch.float64),
    # #  tensor(0.2827, device='cuda:0', dtype=torch.float64),
    # #  tensor(0.0139, device='cuda:0', dtype=torch.float64),
    # #  tensor(0.7776, device='cuda:0', dtype=torch.float64))