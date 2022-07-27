from torchvision import transforms

def trform(IMAGE_DIM=(350, 1500)):

    transformer = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        #  transforms.RandomHorizontalFlip(),
        # transforms.Resize(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    return transformer

def invTF():

    inverse_transformer = transforms.Compose([
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
                                ])
    return inverse_transformer
