import torch.nn as nn
import torchvision.transforms.functional as TF

class CNet(nn.Module):
    """
    Neural network model originally consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.5, im_dim: (int, int) = (350, 1500)) -> None:
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 350 x 1500)  # olde (b x 3 x 227 x 227)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 22), stride=4),  # (b x 96 x 85 x 370)
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.AdaptiveMaxPool2d(output_size=(42, 184)),  # try this adaptive max pool to fix everything
            # nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 42 x 184) [42x184]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(6, 10), padding=(2, 0), stride=(2, 4)),
                     # (b x 256 x 20 x 44) [21x44]
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # [21x44]
            nn.Conv2d(256, 384, kernel_size=(4, 8), stride=(1, 2)),  # (b x 384 x 16 x 18) [18x19]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(4, 6)),  # (b x 384 x 12 x 12) [15x14]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),  # (b x 256 x 9 x 9) [13x12]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=1, padding=0),  # (b x 256 x 6 x 6) [10x9]
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=(256 * 10 * 9), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=num_classes),
            # nn.LogSoftmax(dim=0)
        )
        # self.init_bias()  # initialize bias
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight)
            # nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            # nn.init.kaiming_normal_(module.weight)
            # nn.init.xavier_normal_(module.weight)  # didn't work well for classification, vanishing w's
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    # def init_bias(self):
    #     for layer in self.net:
    #         if isinstance(layer, nn.Conv2d):
    #             nn.init.normal_(layer.weight, mean=0, std=0.01)
    #             nn.init.constant_(layer.bias, 0.5) #0
    #     # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
    #     nn.init.constant_(self.net[4].bias, 0.5)#1
    #     nn.init.constant_(self.net[10].bias, 0.5)
    #     nn.init.constant_(self.net[12].bias, 0.5)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        # add autograd to pipeline
        x = TF.autocontrast(x)
        x = self.net(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x.squeeze(1)

        # return self.classifier(x)