# TestPlotReg_helper.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf
from scipy import ndimage
from regAccuracy import regAcc

invTF = transforms.Compose([
                            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
                            transforms.ToPILImage()
                            ])

def show_testRegimages( model, test_loader, error, title='RDT_Q_test_images', nx=1, ny=8, reg=False,
                     th=1e-8, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # print(next(iter(test_loader)))

    count = testing_loss = test_correct = 0
    with torch.no_grad():
        model.eval()
        for Xt, Yt in test_loader:
            plt.subplots(nx, ny, figsize=(10, 6))
            Xt, Yt = Xt.to(device), Yt.to(device)
            # print(Xt.shape, Yt.shape)
            Yt_pred = model(Xt.to(device))
            # X = invTF(X)
            if reg == False:
                loss = error(Yt_pred, Yt)
            else:
                Y_pred_class, test_correct = regAcc(Yt, Yt_pred, test_correct, th)
                loss = error(Yt_pred, Yt)
            testing_loss += loss.item()  # *Yt.size(0)

            # Compute accuracy
            for j, i in enumerate(Yt_pred):
                plt.subplot(nx, ny, j + 1)
                # print(f'before TF: {Xt.shape}')
                # Timg = invTF(tf.autocontrast(Xt[j, :, :, :]).squeeze(0))
                Timg = ndimage.rotate(tf.autocontrast(Xt[j, :, :, :]).permute(1, 2, 0).cpu().squeeze(0), -90)
                # print(f'before TF: {Timg.shape}')
                plt.imshow(np.abs(Timg))  # .cpu().permute(1, 2, 0))
                if reg == False:  # for classification
                    plt.title(f"Actual: {Yt[j].item()}\n  Pred: {torch.argmax(i)}", fontsize=8, fontweight='bold')
                else:  # for regression
                    if j == 0:
                        plt.title(f"Actual: {2 ** Yt[j] - 2: .4f}\n  Pred:"
                                  f" {2 ** Y_pred_class[j] - 2: .4f}", fontsize=6, fontweight='bold')
                    # plt.title(f"Actual: {Yt[j]: .4f}\n  Pred: {Y_pred_class[j]: .4f}", fontsize=6, fontweight='bold')
                    plt.title(f"Actual: {2 ** Yt[j] - 2: .4f}\n  Pred: {2 ** Y_pred_class[j] - 2: .4f}", fontsize=6,
                              fontweight='bold')
                plt.xticks([])
                plt.yticks([])
            #     if j == (nx * ny - 1):
            #         break
            # if j == (nx * ny - 1):
            #     break

            plt.suptitle('Test Image\nConcentration Predictions (ng/mL)\nNormalized + Auto-Contrast', fontsize=18,
                         fontweight='bold')
            plt.tight_layout()
            plt.savefig('./figures/' + title + f'TestImages.{count}.png', edgecolor='none')
            plt.show()
            count += 1

    # plt.subplots(nx, ny, figsize=(4.5, 8))
    #
    # with torch.no_grad():
    #     model.eval()
    #     for Xt, Yt in test_loader:
    #         Xt, Yt = Xt.to(device), Yt.to(device)
    #         # print(Xt.shape, Yt.shape)
    #         Yt_pred = model(Xt.to(device))
    #         # X = invTF(X)
    #         if reg == False:
    #             loss = error(Yt_pred, Yt)
    #         else:
    #             Y_pred_class, test_correct = regAcc(Yt, Yt_pred, test_correct)
    #             loss = error(Yt_pred, Yt)
    #         testing_loss += loss.item()  # *Yt.size(0)
    #
    #         # Compute accuracy
    #         for j, i in enumerate(Yt_pred):
    #             plt.subplot(nx, ny, j+1)
    #             # print(f'before TF: {Xt.shape}')
    #             # Timg = invTF(tf.autocontrast(Xt[j, :, :, :]).squeeze(0))
    #             Timg = tf.autocontrast(Xt[j, :, :, :]).squeeze(0)
    #             # print(f'before TF: {Timg.shape}')
    #             plt.imshow(Timg.cpu().permute(1, 2, 0))
    #             if reg == False:   # for classification
    #                 plt.ylabel(f"Actual: {Yt[j].item()}\n  Pred: {torch.argmax(i)}", fontsize=8, fontweight='bold')
    #             else:   # for regression
    #                 plt.ylabel(f"Actual: {Y_pred_class: .4f}\n  Pred: {i.item(): .4f}", fontsize=6, fontweight='bold')
    #                 # plt.ylabel(f"Actual: {2**(Yt[j].item())-2: .4f}\n  Pred: {i.item(): .4f}", fontsize=6, fontweight='bold')
    #             plt.xticks([])
    #             plt.yticks([])
    #             if j == (nx*ny-1):
    #                 break
    #         if j == (nx*ny-1):
    #             break
    #
    # plt.suptitle('Test Image\nConcentration Predictions\nNormalized + Auto-Contrast', fontsize=18, fontweight='bold')
    # plt.tight_layout()
    # plt.savefig('./figures/' + title + 'TestImages.png', edgecolor='none')
    # plt.show()
