import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import numpy as np

# from transform_helper import invTF

invTF = transforms.Compose([
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
                                ])

def show_images(x, Y, title, nmax=8, iTF=True, imC=True, reg=False):
    fig, ax = plt.subplots(nmax, 1, figsize=(3.25, 8))
    for i in range(nmax):
        # option for plotting the original image (inverse normalization) or the normed version
        if iTF == True:
            image = invTF(TF.autocontrast(x[i, :, :, :])).squeeze(0)
            tit1 = 'Original image'
        else:
            image = TF.autocontrast(x[i, :, :, :]).squeeze(0)
            tit1 = 'Normalized image'

        # option for plotting uint8 or float
        if imC == True:
            ax[i].imshow((image * 255).permute(1, 2, 0).numpy().astype(np.uint8))
            tit2 = 'unit8 value'
        else:
            ax[i].imshow((image.permute(1, 2, 0)))
            tit2 = 'with autocontrast'  # 'float value'

        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if reg == False:  # for classification
            ax[i].set_ylabel(f"Class: {Y[i].item()}", fontsize=8, fontweight='bold')
        else:   # for regression
            ax[i].set_ylabel(f" Conc: \n{(2**(Y[i].item())-2): .4f} ng/mL", fontsize=7, fontweight='bold')

    
    plt.suptitle(tit1 + '\n ' + tit2, fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./figures/' + title + '_' + tit1 + '_' + tit2 + '.png', edgecolor='none')
    plt.show()

def show_batch(dl, title, nmax=8, iTF=True, imC=True, reg=False):
    for images, labels in dl:
        show_images(images, labels, title, nmax, iTF, imC, reg)
        break


# defunct attempts
# # Display image and label.
# train_features, train_labels = next(iter(dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# # print(f"Labels batch shape: {train_labels.size()}")
# print(f"Labels batch shape: {len(train_labels)}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(invTF(img))  #, cmap="gray")
# plt.show()
# print(f"Label: {label}")
#
# #%%
#
#
# # figure = plt.figure(figsize=(8, 8))
# # cols, rows = 3, 3
# # for i in range(1, cols * rows + 1):
# #     sample_idx = torch.randint(len(dataloader), size=(1,)).item()
# #     img, label = dataloader[sample_idx]
# #     figure.add_subplot(rows, cols, i)
# #     plt.title(labels_map[label])
# #     plt.axis("off")
# # plt.show()
# #     plt.imshow(img.squeeze(), cmap="gray")
#
# # Display image and label.
# train_features, train_labels = next(iter(dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {len(train_labels)}")
# img = train_features[0].squeeze()
# print(f"Squeezed feature batch shape: {img.size()}")
# label = train_labels[0]
# plt.imshow(invTF(img))  #, cmap="gray")
# plt.show()
# print(f"Label: {label}")