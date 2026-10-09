import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two-convolution feature extraction block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class GalaxyCNNV2(nn.Module):
    """
    GalaxAI Custom CNN v2

    Multi-task galaxy morphology classifier.

    Outputs:
        8 binary morphology attributes
        1 four-class bulge prediction
        1 three-class roundedness prediction

    Total morphology tasks: 10
    Total output logits: 15
    """

    def __init__(self):
        super().__init__()

        
        # SHARED CNN BACKBONE
        

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        
        # SHARED GALAXY REPRESENTATION
        

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.40),
        )

        
        # BINARY MORPHOLOGY HEADS
        

        self.featured_head = nn.Linear(256, 1)

        self.edge_on_head = nn.Linear(256, 1)

        self.bar_head = nn.Linear(256, 1)

        self.spiral_arms_head = nn.Linear(256, 1)

        self.disturbed_head = nn.Linear(256, 1)

        self.merger_head = nn.Linear(256, 1)

        self.clumpy_head = nn.Linear(256, 1)

        self.symmetry_head = nn.Linear(256, 1)

        
        # MULTI-CLASS MORPHOLOGY HEADS
        

        # Bulge:
        # 0 = None
        # 1 = Just noticeable
        # 2 = Obvious
        # 3 = Dominant

        self.bulge_head = nn.Linear(256, 4)

        # Roundedness:
        # 0 = Completely
        # 1 = In-between
        # 2 = Cigar-shaped

        self.roundedness_head = nn.Linear(256, 3)

    def forward(self, x):

        # Shared visual feature extraction
        x = self.features(x)

        # Convert spatial feature maps into
        # a compact galaxy representation
        x = self.global_pool(x)

        # Shared feature vector
        x = self.shared(x)

        
        # OUTPUTS
        

        outputs = {
            "featured": self.featured_head(x).squeeze(1),

            "edge_on": self.edge_on_head(x).squeeze(1),

            "bar": self.bar_head(x).squeeze(1),

            "spiral_arms": self.spiral_arms_head(x).squeeze(1),

            "bulge": self.bulge_head(x),

            "roundedness": self.roundedness_head(x),

            "disturbed": self.disturbed_head(x).squeeze(1),

            "merger": self.merger_head(x).squeeze(1),

            "clumpy": self.clumpy_head(x).squeeze(1),

            "symmetry": self.symmetry_head(x).squeeze(1),
        }

        return outputs