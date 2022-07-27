import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf
import torch
import numpy as np

invTF = transforms.Compose([
                            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
                            transforms.ToPILImage()
                            ])

def show_testimages( model, test_loader, error, title='RDT_Q_test_images', nx=8, ny=1, reg=False,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # print(next(iter(test_loader)))

    testing_loss = 0

    plt.subplots(nx, ny, figsize=(4.5, 8))

    with torch.no_grad():
        model.eval()
        for Xt, Yt in test_loader:
            Xt, Yt = Xt.to(device), Yt.to(device)
            # print(Xt.shape, Yt.shape)
            Yt_pred = model(Xt.to(device))
            # X = invTF(X)
            if reg == False:
                loss = error(Yt_pred, Yt)
            else:
                loss = error(Yt_pred, Yt)
            testing_loss += loss.item()  # *Yt.size(0)

            # Compute accuracy
            for j, i in enumerate(Yt_pred):
                plt.subplot(nx, ny, j+1)
                # print(f'before TF: {Xt.shape}')
                # Timg = invTF(tf.autocontrast(Xt[j, :, :, :]).squeeze(0))
                Timg = tf.autocontrast(Xt[j, :, :, :]).squeeze(0)
                # print(f'before TF: {Timg.shape}')
                plt.imshow(Timg.cpu().permute(1, 2, 0))
                if reg == False:   # for classification
                    plt.ylabel(f"Actual: {Yt[j].item()}\n  Pred: {torch.argmax(i)}", fontsize=8, fontweight='bold')
                else:   # for regression
                    plt.ylabel(f"Actual: {2**(Yt[j].item())-2: .4f}\n  Pred: {i.item(): .4f}", fontsize=6,
                               fontweight='bold')
                plt.xticks([])
                plt.yticks([])
                if j == (nx*ny-1):
                    break
            if j == (nx*ny-1):
                break

    plt.suptitle('Test Image\nConcentration Predictions\nNormalized + Auto-Contrast', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./figures/' + title + 'TestImages.png', edgecolor='none')
    plt.show()
