from torch import nn

class LeNet5(nn.Module):
    def __init__(self, num_classes, pool_1, pool_2, in_channels=1, padding=2):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=padding),
            nn.Tanh(),
            pool_1,
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            pool_2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))
    
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, pool_list, num_classes=1000, init_weights=False, initial_image_size=224, dropout=False, version='VGG11'):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.pool_list = pool_list
        self.features = self._make_layers(cfg[version], initial_image_size)
        self.classifier = self._make_classifier(initial_image_size, num_classes, dropout=dropout)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, initial_image_size):
        layers = []
        in_channels = 3
        # pool_size_7x7 = 7

        i = 0
        for x in cfg:
            if x == 'M':
                layers += [self.pool_list[i]]
                i += 1
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x

        # final_pool_size = min(pool_size_7x7, initial_image_size // (2**(len([x for x in cfg if x == 'M']))) )

        # layers += [nn.AdaptiveAvgPool2d((final_pool_size, final_pool_size))]
        return nn.Sequential(*layers)

    def _make_classifier(self, initial_image_size, num_classes, dropout=False):
        final_pool_size = initial_image_size // (2**(len([x for x in cfg['VGG11'] if x == 'M'])))
        linear_num = 4096 if final_pool_size>=3 else 256
        if dropout:
            return nn.Sequential(
                nn.Linear(512 * final_pool_size * final_pool_size, linear_num),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(linear_num, linear_num),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(linear_num, num_classes),
            )
        else:
            return nn.Sequential(
                nn.Linear(512 * final_pool_size * final_pool_size, linear_num),
                nn.ReLU(True),
                nn.Linear(linear_num, linear_num),
                nn.ReLU(True),
                nn.Linear(linear_num, num_classes),
            )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
