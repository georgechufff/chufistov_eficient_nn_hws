import torch
import torch.nn as nn


class CustomModel(nn.Module):

    def __init__(self):

        super(CustomModel, self).__init__()

        def get_padding(x):
            return x // 2
        
        self.convs = nn.Sequential(
    
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=get_padding(7),
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    
            # 2. Conv5x5 32->64
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=get_padding(5),
                bias=False,
            ),
            nn.ReLU(inplace=True),
    
            # 3. Conv3×3 s2 64->128
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=get_padding(3),
                bias=False,
            ),
            nn.ReLU(inplace=True),
    
            # 4. Conv1×1 128->256
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                padding=get_padding(1),
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
    
            # 5. Conv3×3 s2 256->256
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=get_padding(3),
                bias=False,
            ),
            nn.ReLU(inplace=True),

            # 6. Conv1×1 256->512
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=get_padding(1),
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
    
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x