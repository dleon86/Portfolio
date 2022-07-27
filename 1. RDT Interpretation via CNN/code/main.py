import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from CNet import CNet
from RDT_trainer import RDT_trainer
from ImagePlot_helper import show_batch
from loadQuideldata import loadQuideldata
from TestPlot_helper import show_testimages

## %%

# title = 'RDTCNN_Quidel_04_class_'
title = 'RDTCNN_Quidel_09_reg_'
# title = 'RDTCNN_Quidel_09_reg_'
# title = 'RDTCNN_Quidel_03_state_e19'
# reg_05: switch to leakyrelu(), Lrate:, Mom: .95, Dropout: .4,
# reg_06: switch to tanh()
# reg_07: switch to tanh() switched initiallization and adjusting weight_decay
# reg_08: switch to sigmoid()
# reg_08: switch back to Leakyrelu(alpha=0.1), .9, Dropout: .4, Lrate: 7.5e-4
# reg_08: switch back to Leakyrelu(alpha=0.05), .98, Dropout: .5, Lrate: 2.5e-4

# define pytorch device - useful for device-agnostic execution
torch.cuda.empty_cache()  # clear gpu cache
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# choose classification (reg=False) or regression
# reg = False   #True
reg = True

# define model parameters
NUM_EPOCHS   = 40   # 20 #90  # original paper
BATCH_SIZE   = 2**2 #
DROPOUT      = 0.4
LR_INIT      = 0.00025
W_DECAY      = 0.00005   # 0.00005
MOMENTUM     = 0.95
LR_DECAY     = 0.8
LR_STEP      = 5
VAL_INTERVAL = 1
DEVICE_IDS   = [0]  # GPUs to use
TEST_BATCH_SIZE = BATCH_SIZE #128

if reg == False:
    NUM_CLASSES  = 12  # 12 Dilutions including null & invalid
else:
    NUM_CLASSES  = 1

# prepare csv files of data
train_loader, val_loader, test_loader = loadQuideldata(
                                                        BATCH_SIZE,
                                                        TEST_BATCH_SIZE,
                                                        # subet=True,
                                                        subet=False,
                                                        regression=reg
                                                        )

IMAGE_DIM = (350, 1500) #(4032, 3024)  #227  # pixels

# modify this to point to your data directory
# # TRAIN_IMG_DIR = 'data/2022-05-uw-quidel-covid-2021'
OUTPUT_DIR = 'E:\\GradSchool\\AMATH563\\Project\\CNet_data_out'
LOG_DIR = OUTPUT_DIR + '\\tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '\\models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# #%%


tbwriter = SummaryWriter(log_dir=LOG_DIR)
print('TensorboardX summary writer created')

# create model
model = CNet(num_classes=NUM_CLASSES, im_dim=IMAGE_DIM).to(device)

# convert ReLU to LeakyReLU

def convert_activation_fn(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU(negative_slope=0.05))   # nn.LeakyReLU(inplace=Tru e))
        else:
            convert_activation_fn(child)


# swap ReLU for another activation function
convert_activation_fn(model)

# train on multiple GPUs
model = torch.nn.parallel.DataParallel(model)  # , device_ids=DEVICE_IDS)
print(model)
print('CNN created')

## %%


# # # %% Plot some images
# itf, imc  = False, False
# nmax = 10
# if BATCH_SIZE<nmax:
#     nmax = BATCH_SIZE
# # show_batch(train_loader, title, nmax=nmax)                   # What the model sees
# # show_batch(train_loader, title, nmax=nmax, iTF=itf)
# # show_batch(train_loader, title, nmax=nmax, imC=imc)
# show_batch(train_loader, title, nmax=nmax, iTF=itf, imC=imc, reg=reg)

# create optimizer
optimizer = optim.SGD(params=model.parameters(), lr=LR_INIT, weight_decay=W_DECAY , momentum=MOMENTUM)
# optimizer = optim.RMSprop(params=model.parameters(), lr=LR_INIT, weight_decay=W_DECAY , momentum=MOMENTUM)
# optimizer = optim.Adam(params=model.parameters(), lr=LR_INIT, weight_decay=W_DECAY)
print('Optimizer created')

# create loss function
if reg == False:
    error = nn.CrossEntropyLoss()  # setup for classification
else:
    error = nn.MSELoss()             # setup for regression

# multiply LR by 1 / 10 after every 30 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_DECAY)
print('LR Scheduler created')

# # load former model and test accuracy
# model.load_state_dict(torch.load(CHECKPOINT_DIR+'\\' + title + '_state.pkl'))
# test_accuracy, test_loss, Yt_labs, Yt_plabs = TestAcc(model, test_loader, error, reg=reg, th=1e-7)

train_loss, cv_loss, accuracy_list = [], [], []
## %%
# paramInDict = {
#                 'model': model,
#                 'train_loader': train_loader,
#                 'val_loader': val_loader,
#                 'optimizer': optimizer,
#                 'error': error,
#                 'lr_scheduler': lr_scheduler,
#                 'tbwriter': tbwriter,
#                 'train_loss': train_loss,
#                 'cv_loss': cv_loss,
#                 'accuracy_list': accuracy_list,
#                 'NUM_EPOCHS': NUM_EPOCHS,
#                 'BATCH_SIZE': BATCH_SIZE,
#                 'VAL_INTERVAL': VAL_INTERVAL,
#                 'CHECKPOINT_DIR': CHECKPOINT_DIR,
#                 'title': title,
#                 'device': device
#               }

# # train RDTNet
# model, train_loss, cv_loss, accuracy_list, \
#     Y, Y_pred, Yv, Yv_pred = RDT_trainer(model, train_loader, val_loader,
#                                         optimizer, error, lr_scheduler, tbwriter,
#                                         train_loss, cv_loss, accuracy_list,
#                                         NUM_EPOCHS, BATCH_SIZE, VAL_INTERVAL, CHECKPOINT_DIR,
#                                         title, device, th=1e-7, cont_train=False, reg=reg)

# paramOutDict = RDT_trainer(paramInDict, cont_train=False, reg=reg)
# paramOutDict = RDT_trainer(cont_train=False, reg=reg, paramInDict)
# %%
from TestAccuracy import TestAcc
from TestPlotReg_helper import show_testRegimages
from TestCVloss_plot import TestCVloss_plot

# TestCVloss_plot(accuracy_list, train_loss, cv_loss, title)
## %%
# # load former model and test accuracy
# model.load_state_dict(torch.load(CHECKPOINT_DIR+'\\' + title + '.pkl'))
model.load_state_dict(torch.load(CHECKPOINT_DIR+'\\' + title + '_state.pkl'))
# model.load_state_dict(torch.load(CHECKPOINT_DIR+'\\' + title + 'state.pkl'))

# test_accuracy, test_loss = TestAcc(model, test_loader, error, reg=reg)
# show_testRegimages( model, test_loader, error, title, reg=reg)
## %%
test_accuracy, test_loss, Yt_labs, Yt_plabs = TestAcc(model, test_loader, error, reg=reg, th=1e-7)

# %% Full image
labels  = test_loader.dataset.classes
textstr = '     $n_{samples}$: ' + str(len(Yt_labs)) + \
          '\n  $Accuracy$: ' + f'{test_accuracy.item(): .2f} %' + \
            '\n$MSE_{test} loss$: ' + f'{test_loss: .5f}'
fig, ax = plt.subplots(figsize=(12, 12))
# p1 = plt.scatter(Yt_labs.cpu(), Yt_plabs.cpu(), c=Yt_plabs.cpu(),
#             linewidths=0.5, edgecolors='k', s=150, cmap='prism', label=labels, alpha=0.25)
# plt.plot((0, 4), (0, 4), '--', c='c')
plt.scatter(2**Yt_plabs.cpu()-2, 2**Yt_labs.cpu()-2, c=np.log2(Yt_labs.cpu()+.1),
            linewidths=0.5, edgecolors='k', s=150, cmap='prism', label=labels, alpha=0.5)
plt.plot((-1, 11), (-1, 11), '--', c='c')
ax.set_xlabel('Predicted Value (ng/mL)', fontsize=16, fontweight='bold')
ax.set_ylabel('True Value (ng/mL)', fontsize=16, fontweight='bold')
fig.suptitle('Test Set Predictions\nQuidel COVID-19 Dataset\n'+title, fontsize=24, fontweight='bold')
ax.text(-1, 8, textstr, fontsize=18, fontweight='bold')
for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
    lab.set_fontsize(20)
# for l1 in labels:
#     plt.legend([p1], l1)  #, test_loader.dataset.classes)
plt.savefig('.\\figures\\TestPredictions' + title + '.png', edgecolor='none')
plt.show()

# %% Zoomed in on low conc
labels = test_loader.dataset.classes
textstr = '     $n_{samples}$: ' + str(len(Yt_labs)) + \
          '\n  $Accuracy$: ' + f'{test_accuracy.item(): .2f} %' + \
            '\n$MSE_{test} loss$: ' + f'{test_loss: .5f}'

fig, ax = plt.subplots(figsize=(12, 15))
# p1 = plt.scatter(Yt_labs.cpu(), Yt_plabs.cpu(), c=Yt_plabs.cpu(),
#             linewidths=0.5, edgecolors='k', s=150, cmap='prism', label=labels, alpha=0.25)
# plt.plot((0, 4), (0, 4), '--', c='c')
plt.scatter(2**Yt_plabs.cpu()-2, 2**Yt_labs.cpu()-2, c=np.log2(Yt_labs.cpu()+.1),
            linewidths=0.5, edgecolors='k', s=150, cmap='prism', label=labels, alpha=0.5)   # 'prism'
plt.plot((-1, 11), (-1, 11), '--', c='c')
ax.set_xlabel('Predicted Value (ng/mL)', fontsize=16, fontweight='bold')
ax.set_ylabel('True Value (ng/mL)', fontsize=16, fontweight='bold')
ax.set_xlim(-1.05, .7), ax.set_ylim(-1.04, .7)
fig.suptitle('Test Set Predictions\nZoom in on Low Concentration\nQuidel COVID-19 Dataset\n' + title + '\n\n',
             fontsize=24, fontweight='bold')
ax.text(-1, .5, textstr, fontsize=18, fontweight='bold')
for lab in (ax.get_xticklabels() + ax.get_yticklabels()):
    lab.set_fontsize(20)
# for l1 in labels:
#     plt.legend([p1], l1)  #, test_loader.dataset.classes)
plt.savefig('.\\figures\\TestPredictions' + title + 'Zoomed.png', edgecolor='none')
plt.show()

# %%
threshold = (1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
thdict = {}
for th in  threshold:
    test_accuracy, test_loss, Yt_labs, Yt_plabs = TestAcc(model, test_loader, error, reg=reg, th=th)
    thdict[str(th)] = [test_accuracy.item()]

print(thdict)
# threshold with reg_05
# {'1.0':    [100.0],
#  '0.1':    [80.59701538085938],
#  '0.01':   [52.7363166809082],
#  '0.001':  [52.7363166809082],
#  '0.0001': [52.7363166809082],
#  '1e-05':  [52.7363166809082],
#  '1e-06':  [52.7363166809082],
#  '1e-07':  [52.7363166809082],
#  '1e-08':  [41.79104232788086],
#  '1e-09':  [37.810943603515625],
#  '1e-10':  [37.810943603515625]}
# %%

# %% Continue traing
accuracy_list, Y, Y_pred, Yv, Yv_pred = RDT_trainer(model, train_loader, val_loader,
                                                        optimizer, error, lr_scheduler, tbwriter,
                                                        train_loss, cv_loss, accuracy_list,
                                                        NUM_EPOCHS, BATCH_SIZE, VAL_INTERVAL, CHECKPOINT_DIR,
                                                        title, device, cont_train=True, reg=reg)

# %%
from regAccuracy import regAcc

test_correct = testing_loss = batches = 0

n_test = len(test_loader.sampler.indices)

Yt_labs, Yt_plabs = [], []

with torch.no_grad():
    model.eval()
    for Xt, Yt in test_loader:
        Xt, Yt = Xt.to(device), Yt.to(device)
        batches += 1

        Yt_pred = model(Xt)
        if reg == False:
            testing_loss += error(Yt_pred, Yt).item()
            # loss = error(Yt_pred, Yt)
            test_correct += (torch.argmax(Yt_pred.cpu(), dim=1) == Yt).sum()
        else:
            Y_pred_class, test_correct = regAcc(Yt, Yt_pred, test_correct)  # train_correct
            testing_loss += error(Yt_pred, Yt.float()).item()
            # loss =  error(Yt_pred.squeeze(1), Yt.float())
        # testing_loss += loss.item()

test_accuracy = test_correct / n_test * 100
testing_loss /= batches
if reg == False:
    print(f'Test set accuracy : {test_accuracy}')
else:
    print(f'Test set MSE loss : {testing_loss}')
    print(f'Test set accuracy : {test_accuracy}')

# %% Version of TestPlotReg_helper.py for batch size = 4
import torchvision.transforms.functional as tf
from scipy import ndimage
from regAccuracy import regAcc

count = idx = testing_loss = test_correct = 0

nx, ny = 1, 8

# plt.subplots(nx, ny, figsize=(10, 6))

with torch.no_grad():
    model.eval()
    for Xt, Yt in test_loader:
        if count % 2 == 0:
            plt.subplots(nx, ny, figsize=(10, 6))
            idx = 0
        Xt, Yt = Xt.to(device), Yt.to(device)
        # print(Xt.shape, Yt.shape)
        Yt_pred = model(Xt.to(device))
        # X = invTF(X)
        if reg == False:
            loss = error(Yt_pred, Yt)
        else:
            Y_pred_class, test_correct = regAcc(Yt, Yt_pred, test_correct, 1e-7)
            loss = error(Yt_pred, Yt)
        testing_loss += loss.item()  # *Yt.size(0)

        # Compute accuracy
        for j, i in enumerate(Yt_pred):
            plt.subplot(nx, ny, idx + 1)
            # print(f'before TF: {Xt.shape}')
            # Timg = invTF(tf.autocontrast(Xt[j, :, :, :]).squeeze(0))
            Timg = ndimage.rotate(tf.autocontrast(Xt[j, :, :, :]).permute(1, 2, 0).cpu().squeeze(0), -90)
            # print(f'before TF: {Timg.shape}')
            plt.imshow(np.abs(Timg))   # .cpu().permute(1, 2, 0))
            if reg == False:  # for classification
                plt.ylabel(f"Actual: {Yt[j].item()}\n  Pred: {torch.argmax(i)}", fontsize=8, fontweight='bold')
            else:  # for regression
                if j == 0:
                    plt.title(f"Actual: {2 ** Yt[j] - 2: .4f}\n  Pred:"
                              f" {2 ** Y_pred_class[j] - 2: .4f}", fontsize=6, fontweight='bold')
                # plt.title(f"Actual: {Yt[j]: .4f}\n  Pred: {Y_pred_class[j]: .4f}", fontsize=6, fontweight='bold')
                plt.title(f"Actual: {2**Yt[j]-2: .4f}\n  Pred: {2**Y_pred_class[j]-2: .4f}", fontsize=6,
                          fontweight='bold')
            plt.xticks([])
            plt.yticks([])
            idx += 1  # index for subplot
        count += 1    # index for file saving and figure generation
        if count % 2 == 0:
            plt.suptitle(title + 'Test Image\nConcentration Predictions (ng/mL)\nNormalized + Auto-Contrast', fontsize=18,
                         fontweight='bold')
            plt.tight_layout()
            plt.savefig('./figures/' + title + f'TestImages.{(count+1)/2}.png', edgecolor='none')
            plt.show()

# %%
#  # developing regression accuracy module
# # def regTestAc(Y, Y_pred, correct, loss, error):
#
# Y_true = torch.cuda.DoubleTensor((np.log2(2),
#                        np.log2(10/2**0+2),
#                        np.log2(10/2**1+2),
#                        np.log2(10/2**2+2),
#                        np.log2(10/2**3+2),
#                        np.log2(10/2**4+2),
#                        np.log2(10/2**5+2),
#                        np.log2(10/2**6+2),
#                        np.log2(10/2**7+2),
#                        np.log2(10/2**8+2),
#                        np.log2(10/2**9+2),
#                        np.log2(1),
#                        ))  #dtype=torch.float64)).to(device='cuda')
# correct = 0
# Y_pred_class = torch.zeros_like(Y)
# ntot = len(Y_pred)
#
# for i in range(len(Y)):
#     Y_pred_class[i] = Y_true[torch.argmin(torch.abs(Y_true-Y_pred[i]))]
#     # print(torch.argmin(torch.abs(Y-Y_pred[i])),'\n',Y_true[torch.argmin(torch.abs(Y_true-Y_pred[i]))])
#
# correct +=  ((Y - Y_pred_class)==0).sum()
# print(correct)
#     # return Y_pred_class

# %%
# https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
# we will save the conv layer weights in this list
model_weights =[]
#we will save the 5 conv layers in this list
conv_layers = []  # get all the model children as list

model_children = list(model.module.net.children())  #list(model.children())  #counter to keep count of the conv layers
counter = 0  #append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")
#%%
from scipy import ndimage
outputs = []
names = []
processed = []

nx, ny = 1, 4

# plt.subplots(nx, ny, figsize = (12,12))

with torch.no_grad():
    model.eval()
    for X, _ in train_loader:
        # Ycv_pred = model(X.to(device))
        # X1 = invTF(X
        k = 0
        for layer in conv_layers[0:]:
            if k==1:
                break
            for n in range(nx * ny):
                layer = layer.to(device)
                image = layer(X[n, :, :, :].to(device))

                # image = TF(image)
                print(f"Image shape before: {image.shape}")
                image = image.unsqueeze(0)
                print(f"Image shape after: {image.shape}")
                # image = image.to(device)

                outputs.append(image)
                names.append(str(layer))
            print(len(outputs))  # print feature_maps

            for feature_map in outputs:
                print(feature_map.shape)

            # for feature_map in outputs:
            #     feature_map = feature_map.squeeze(0)
            #     gray_scale = torch.sum(feature_map, 0)
            #     gray_scale = gray_scale / feature_map.shape[0]
            #     processed.append(gray_scale.data.cpu().numpy())

            for fm in processed:
                print(fm.shape)

            k += 1

        if k == 1:
            break


# %% plot features from convlayer 1
nx1 = int(np.floor(np.sqrt(feature_map.shape[0])))

nx1, ny1 = 1, 4

fig, ax = plt.subplots(nx1, ny1, figsize = (10,16))
plt.suptitle('Feature Map from 1st Convolution Layer:\nFist 16 Features', fontsize=28, fontweight='bold')
plt.savefig('./figures/' + title + f'TestImagesCNNLayer1Features16.png', edgecolor='none')
for q in range(nx1*ny):

    plt.subplot(nx1, ny1, q+1)
    plt.tight_layout()

    plt.imshow(ndimage.rotate(feature_map[0, q+12, :, :].squeeze(0).cpu(), -90), cmap='viridis', interpolation='none')
    # plt.imshow(ndimage.rotate(feature_map[0, q, :, :].squeeze(0).cpu(), -90), cmap='viridis', interpolation='none')

    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig('./figures/' + title + f'Feature Map Sample.{32}.png', edgecolor='none')
plt.show()

# %% extra code

for feature_map in outputs:
    print(feature_map.shape)

for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

for fm in processed:
    print(fm.shape)

# # %%
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

for fm in processed:
    print(fm.shape)


# # %%

nx1 = int(np.floor(np.sqrt(feature_map.shape[0])))

nx1, ny1 = 1, 8

fig ,ax = plt.subplots(nx1, ny1, figsize = (10,6))
plt.suptitle(f'Feature Map from 1st Convolution Layer:\n{nx1*ny1} images in first filter of CNN', fontsize=28,
             fontweight='bold')
for q in range(nx1*ny1):

    plt.subplot(nx1, ny1, q+1)
    plt.tight_layout()

    plt.imshow(ndimage.rotate(processed[q], -90), cmap='viridis', interpolation='none')

    plt.xticks([])
    plt.yticks([])

plt.show()
plt.savefig('./figures/' + title + f'TestImagesCNNLayer1Images8.png', edgecolor='none')


# %%