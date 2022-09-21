from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class RDTDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = pd.read_csv(img_dir, header=None)
        # # version for classification
        # self.samples = list(zip(
        #     list(self.img_dir.values.squeeze().astype(str)),
        #     list(self.img_labels.values.squeeze().astype(int))
        # ))
        # version for regression
        self.samples = list(zip(
            list(self.img_dir.values.squeeze().astype(str)),
            list(self.img_labels.values.squeeze().astype(float))
        ))
        self.transform = transform

        self.targets = [s[1] for s in self.samples]
        # set-up for classification
        self.classes = [
                        'negative',
                        '10.00 ng/mL',
                        '5.00 ng/mL',
                        '2.50 ng/mL',
                        '1.25 ng/mL',
                        '0.625 ng/mL',
                        '0.3125 ng/mL',
                        '0.1563 ng/mL',
                        '0.0781 ng/mL',
                        '0.0391 ng/mL',
                        '0.0195 ng/mL',
                        'invalid'
                        ]
        self.class_to_idx = dict([
                        ('negative',      0),
                        ('10.00 ng/mL',   1),
                        ('5.00 ng/mL',    2),
                        ('2.50 ng/mL',    3),
                        ('1.25 ng/mL',    4),
                        ('0.625 ng/mL',   5),
                        ('0.3125 ng/mL',  6),
                        ('0.1563 ng/mL',  7),
                        ('0.0781 ng/mL',  8),
                        ('0.0391 ng/mL',  9),
                        ('0.0195 ng/mL', 10),
                        ('invalid',      11)
                        ])
        # set-up for regression
        # self.classes = [
        #                 '0.0',
        #                 '10.0',
        #                 '5.0',
        #                 '2.5',
        #                 '1.25',
        #                 '0.625',
        #                 '0.3125',
        #                 '0.15625',
        #                 '0.078125',
        #                 '0.0390625',
        #                 '0.01953125',
        #                 '-1.0'
        #                 ]
        # self.class_to_idx = dict([
        #                 ('negative',      0.0),
        #                 ('10.00 ng/mL',  10.0),
        #                 ('5.00 ng/mL',    5.0),
        #                 ('2.50 ng/mL',    2.5),
        #                 ('1.25 ng/mL',    1.25),
        #                 ('0.625 ng/mL',   0.625),
        #                 ('0.3125 ng/mL',  0.3125),
        #                 ('0.1563 ng/mL',  0.15625),
        #                 ('0.0781 ng/mL',  0.078125),
        #                 ('0.0391 ng/mL',  0.0390625),
        #                 ('0.0195 ng/mL',  0.01953125),
        #                 ('invalid',      -1.0)
        #                 ])
        # self.class_to_idx = dict([
        #                 ('negative',      '0.0'),
        #                 ('10.00 ng/mL',  '10/2^0'),
        #                 ('5.00 ng/mL',    '10/2^1'),
        #                 ('2.50 ng/mL',    '10/2^2'),
        #                 ('1.25 ng/mL',    '10/2^3'),
        #                 ('0.625 ng/mL',   '10/2^4'),
        #                 ('0.3125 ng/mL',  '10/2^5'),
        #                 ('0.1563 ng/mL',  '10/2^6'),
        #                 ('0.0781 ng/mL',  '10/2^7'),
        #                 ('0.0391 ng/mL',  '10/2^8'),
        #                 ('0.0195 ng/mL',  '10/2^9'),
        #                 ('invalid',      '10/2^10')
        #                 ])
        # self.class_to_idx = dict([
        #                 ('negative',      0.0),
        #                 ('10.00 ng/mL',  10/2**0),
        #                 ('5.00 ng/mL',    10/2**1),
        #                 ('2.50 ng/mL',    10/2**2),
        #                 ('1.25 ng/mL',    10/2**3),
        #                 ('0.625 ng/mL',   10/2**4),
        #                 ('0.3125 ng/mL',  10/2**5),
        #                 ('0.1563 ng/mL',  10/2**6),
        #                 ('0.0781 ng/mL',  10/2**7),
        #                 ('0.0391 ng/mL',  10/2**8),
        #                 ('0.0195 ng/mL',  10/2**9),
        #                 ('invalid',      10/2**10)
        #                 ])
        self.imgs = self.samples

    def __len__(self):
        return len(list(self.img_labels.values.squeeze().astype(int)))

    def __getitem__(self, idx):
        image = Image.open(str(self.img_dir.iloc[idx, 0])).convert('RGB')  #[idx])
        label = (self.img_labels.iloc[idx, 0])   #[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
