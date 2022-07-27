import random

import torch
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from RDTDataset_helper import RDTDataset
from transform_helper import trform

def loadQuideldata(BATCH_SIZE, TEST_BATCH_SIZE, subet=False, regression=False):
    # prep the transforms
    transform = trform()

    # #%% Loading full dataset
    df = pd.read_csv('data\\2022-05-uw-photos-metadata.csv')  # , sep=',')
    set = ['uw-quidel-covid-2021']
    dfq = df[df['project_id'].isin(set)].reset_index()  # choose only quidel data
    col_id = list(dfq.columns)

    # fix mislabeled data
    dfq.loc[dfq['rdt_id'] == '124121', 'rdt_id'] = "121"
    dfq.loc[dfq['rdt_id'] == '.214', 'rdt_id'] = "214"

    # set the invalid dilution value to 11
    invset = ['false']
    dfq.loc[dfq['control_line'].isin(invset), 'dilution'] = 11

    # this is for loading the full dataset
    if subet==False and regression==False:
        # mute this if you want to do subset (samp)
        dfq = dfq[['dilution', 'photo_uid']]
        dfq['loc'] = 'data\\2022-05-uw-quidel-covid-2021\\'  # './qwork/'
        dfq['photo_uid'] = dfq['loc'] + dfq['photo_uid']
        dfq = dfq.drop(['loc'], axis=1)
        dfq['dilution'] = dfq['dilution'].values
        dfq.to_csv('.\\dfq_labels.csv', columns=['dilution'], index=False, header=None)
        dfq.to_csv('.\\dfq_impath.csv', columns=['photo_uid'], index=False, header=None)

        dataset = RDTDataset(annotations_file='dfq_labels.csv', img_dir='dfq_impath.csv', transform=transform)

    elif subet==True and regression==False:
        # separate subset of data for figuring out dataloader
        dfq_samp = dfq.iloc[:256]

        rind = np.random.randint(0, 2000, 512)

        # # mute this if you want to do full set (above)
        dfq_samp = dfq.iloc[rind]
        dfq_samp = dfq_samp[['dilution', 'photo_uid']]
        dfq_samp['loc'] = 'data\\2022-05-uw-quidel-covid-2021\\'  #'./qwork/'
        dfq_samp['photo_uid'] = dfq_samp['loc'] + dfq_samp['photo_uid']
        dfq_samp = dfq_samp.drop(['loc'], axis=1)
        dfq_samp['dilution'] = dfq_samp['dilution'].values   #.astype(str)
        dfq_samp.to_csv('.\\dfq_samp_labels.csv', columns=['dilution'], index=False, header=None)
        dfq_samp.to_csv('.\\dfq_samp_impath.csv', columns=['photo_uid'], index=False, header=None)

        dataset = RDTDataset(annotations_file='dfq_samp_labels.csv', img_dir='dfq_samp_impath.csv', transform=transform)

    elif subet == True and regression == True:
        # print('Warning, no data prepared for this situation!!!') # invalid statement, remove later
        dfq_samp = dfq.iloc[:256]

        rind = np.random.randint(0, 2000, 512)
        dfq_samp = dfq.iloc[rind]
        dfq_samp = dfq_samp[['dilution', 'photo_uid']]
        dfq_samp['loc'] = 'data\\2022-05-uw-quidel-covid-2021\\'  # './qwork/'
        dfq_samp['photo_uid'] = dfq_samp['loc'] + dfq_samp['photo_uid']
        dfq_samp = dfq_samp.drop(['loc'], axis=1)
        dfq_samp['dilution'] = dfq_samp['dilution'].values  # .astype(str)
        # dfq_samp.to_csv('.\\dfq_samp_labels.csv', columns=['dilution'], index=False, header=None)
        # dfq_samp.to_csv('.\\dfq_samp_impath.csv', columns=['photo_uid'], index=False, header=None)

        dfq_samp.loc[(dfq['dilution'] == 0), 'dilution_reg_log'] = np.log2(2)
        dfq_samp.loc[(dfq['dilution'] == 1), 'dilution_reg_log'] = np.log2(10 / 2 ** 0 + 2)
        dfq_samp.loc[(dfq['dilution'] == 2), 'dilution_reg_log'] = np.log2(10 / 2 ** 1 + 2)
        dfq_samp.loc[(dfq['dilution'] == 3), 'dilution_reg_log'] = np.log2(10 / 2 ** 2 + 2)
        dfq_samp.loc[(dfq['dilution'] == 4), 'dilution_reg_log'] = np.log2(10 / 2 ** 3 + 2)
        dfq_samp.loc[(dfq['dilution'] == 5), 'dilution_reg_log'] = np.log2(10 / 2 ** 4 + 2)
        dfq_samp.loc[(dfq['dilution'] == 7), 'dilution_reg_log'] = np.log2(10 / 2 ** 5 + 2)
        dfq_samp.loc[(dfq['dilution'] == 6), 'dilution_reg_log'] = np.log2(10 / 2 ** 6 + 2)
        dfq_samp.loc[(dfq['dilution'] == 8), 'dilution_reg_log'] = np.log2(10 / 2 ** 7 + 2)
        dfq_samp.loc[(dfq['dilution'] == 9), 'dilution_reg_log'] = np.log2(10 / 2 ** 8 + 2)
        dfq_samp.loc[(dfq['dilution'] == 10), 'dilution_reg_log'] = np.log2(10 / 2 ** 9 + 2)
        dfq_samp.loc[(dfq['dilution'] == 11), 'dilution_reg_log'] = np.log2(1)

        dfq_samp.to_csv('.\\dfq_samp_labels_regression.csv', columns=['dilution_reg_log'], index=False, header=None)
        dfq_samp.to_csv('.\\dfq_samp_impath.csv', columns=['photo_uid'], index=False, header=None)

        dataset = RDTDataset(annotations_file='dfq_samp_labels_regression.csv', img_dir='dfq_samp_impath.csv',
                             transform=transform)

    else:
        # mute this if you want to do subset (samp)
        dfq = dfq[['dilution', 'photo_uid']]
        dfq['loc'] = 'data\\2022-05-uw-quidel-covid-2021\\'  # './qwork/'
        dfq['photo_uid'] = dfq['loc'] + dfq['photo_uid']
        dfq = dfq.drop(['loc'], axis=1)
        dfq['dilution'] = dfq['dilution'].values
        dfq['dilution_reg'] = dfq['dilution'].values

        # dfq.loc[(dfq['dilution'] == 0), 'dilution_reg_log'] = np.log2(2)
        # dfq.loc[(dfq['dilution'] == 1), 'dilution_reg_log'] = np.log2(12)
        # dfq.loc[(dfq['dilution'] == 2), 'dilution_reg_log'] = np.log2(7)
        # dfq.loc[(dfq['dilution'] == 3), 'dilution_reg_log'] = np.log2(3.5)
        # dfq.loc[(dfq['dilution'] == 4), 'dilution_reg_log'] = np.log2(3.25)
        # dfq.loc[(dfq['dilution'] == 5), 'dilution_reg_log'] = np.log2(2.625)
        # dfq.loc[(dfq['dilution'] == 6), 'dilution_reg_log'] = np.log2(2.3125)
        # dfq.loc[(dfq['dilution'] == 7), 'dilution_reg_log'] = np.log2(2.15625)
        # dfq.loc[(dfq['dilution'] == 8), 'dilution_reg_log'] = np.log2(2.078125)
        # dfq.loc[(dfq['dilution'] == 9), 'dilution_reg_log'] = np.log2(2.0390625)
        # dfq.loc[(dfq['dilution'] == 10), 'dilution_reg_log'] = np.log2(2.01953125)
        # dfq.loc[(dfq['dilution'] == 11), 'dilution_reg_log'] = np.log2(1)

        dfq.loc[(dfq['dilution'] == 0), 'dilution_reg_log'] = np.log2(2)
        dfq.loc[(dfq['dilution'] == 1), 'dilution_reg_log'] = np.log2(10/2**0+2)
        dfq.loc[(dfq['dilution'] == 2), 'dilution_reg_log'] = np.log2(10/2**1+2)
        dfq.loc[(dfq['dilution'] == 3), 'dilution_reg_log'] = np.log2(10/2**2+2)
        dfq.loc[(dfq['dilution'] == 4), 'dilution_reg_log'] = np.log2(10/2**3+2)
        dfq.loc[(dfq['dilution'] == 5), 'dilution_reg_log'] = np.log2(10/2**4+2)
        dfq.loc[(dfq['dilution'] == 7), 'dilution_reg_log'] = np.log2(10/2**5+2)
        dfq.loc[(dfq['dilution'] == 6), 'dilution_reg_log'] = np.log2(10/2**6+2)
        dfq.loc[(dfq['dilution'] == 8), 'dilution_reg_log'] = np.log2(10/2**7+2)
        dfq.loc[(dfq['dilution'] == 9), 'dilution_reg_log'] = np.log2(10/2**8+2)
        dfq.loc[(dfq['dilution'] == 10), 'dilution_reg_log'] = np.log2(10/2**9+2)
        dfq.loc[(dfq['dilution'] == 11), 'dilution_reg_log'] = np.log2(1)

        dfq.to_csv('.\\dfq_labels_regression.csv', columns=['dilution_reg_log'], index=False, header=None)
        dfq.to_csv('.\\dfq_impath.csv', columns=['photo_uid'], index=False, header=None)

        dataset = RDTDataset(annotations_file='dfq_labels_regression.csv', img_dir='dfq_impath.csv',
                             transform=transform)

    print('Datasets created')

    # # print the seed value # pytorch.org/docs/master/notes/randomness.html#dataloader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    g = torch.Generator()
    g.manual_seed(58008)
    # print('Used seed : {}'.format(seed))
    dataloader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        num_workers= 16,   # 3,
        worker_init_fn=seed_worker,  # added these
        generator=g,                 #  two lines last
    )

    print('Dataloader created')

    #
    # # set up for validation set
    # np.random.seed(seed)
    idx = list(range(len(dataset)))
    np.random.shuffle(idx)
    # split 20% of training set for validation and test set
    split = int(np.floor(0.2 * len(dataset)))
    cv_samp    = SubsetRandomSampler(idx[:int(split / 2)])
    test_samp  = SubsetRandomSampler(idx[int(split / 2):split])
    train_samp = SubsetRandomSampler(idx[split:])

    # Use Data Loader
    train_loader = torch.utils.data.DataLoader(dataloader.dataset, sampler=train_samp, batch_size=BATCH_SIZE)
    val_loader   = torch.utils.data.DataLoader(dataloader.dataset, sampler=cv_samp, batch_size=BATCH_SIZE)
    test_loader  = torch.utils.data.DataLoader(dataloader.dataset, sampler=test_samp, batch_size=TEST_BATCH_SIZE)

    print('Dataloaders created')

    return train_loader, val_loader, test_loader