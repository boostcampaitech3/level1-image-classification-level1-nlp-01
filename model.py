import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from torchensemble.utils.logging import set_logger
from torchensemble import BaggingClassifier

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class EnsembleModel(nn.Module):
    def __init__(self, model, n_estimators, cuda, criterion, optimizer, scheduler):
        super().__init__()

        self.model = model
        self.n_estimators = n_estimators
        self.cuda = cuda
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ensemble_classifier = BaggingClassifier(estimator=self.model,\
                                                     n_estimators=self.n_estimators,
                                                     cuda=self.cuda)
    
    def forward(self, x):
        x = self.ensemble_classifier(x)
    
# timm library models
# Refactoring Needed

class Efficientnet_B4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class ThreeWayNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.ModuleList([models.resnet50(pretrained=True),
                                    models.resnet50(pretrained=True),
                                    models.resnet50(pretrained=True)])
        self.model[0] = change_last_layer(self.model[0], 3)
        self.model[1] = change_last_layer(self.model[1], 2)
        self.model[2] = change_last_layer(self.model[2], 3)

    def forward(self, x):
        labels = [self.model[i](x) for i in range(3)]
        return labels

class ThreeWayEfficientnet_B0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.ModuleList([EfficientNet.from_pretrained('efficientnet-b0', num_classes=3),
                                    EfficientNet.from_pretrained('efficientnet-b0', num_classes=2),
                                    EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)])
        
    def forward(self, x):
        labels = [self.model[i](x) for i in range(3)]
        return labels

class ThreeWayEfficientnet_B4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.ModuleList([timm.create_model('efficientnet_b4', pretrained=True, num_classes=3),
                                    timm.create_model('efficientnet_b4', pretrained=True, num_classes=2),
                                    timm.create_model('efficientnet_b4', pretrained=True, num_classes=3)])
        
    def forward(self, x):
        labels = [self.model[i](x) for i in range(3)]
        return labels

class CoatMini(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('coat_mini', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class CoatLiteMini(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('coat_lite_mini', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class ConvNext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('convnext_large', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class ConvNext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('convnext_large', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
class ConvNextIn22ft1k(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('convnext_base_in22ft1k', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class Efficientnet_B0(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_B0, self).__init__()
        self.model = timm.create_model('efficientnet_b3a', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class Efficientnet_B7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class VitBase(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_B0, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x 

class VitLarge(nn.Module):
    def __init__(self, num_classes):
        super(VitLarge, self).__init__()
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class SWSResnext(nn.Module):
    def __init__(self, num_classes):
        super(SWSResnext, self).__init__()
        self.model = timm.create_model('swsl_resnext50_32x4d', pretrained = True, num_classes = num_classes)
        
    def forward(self, x):
        x = self.model
        return x
    
class Mobilenet(nn.Module):
    def __init__(self, num_classes):
        super(Mobilenet, self).__init__()
        self.model = timm.create_model('mobilenetv2_100', pretrained = True, num_classes = num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class SwinLarge(nn.Module):
    def __init__(self, num_classes):
        super(SwinLarge, self).__init__()
        self.model = timm.create_model('swin_large_patch4_window7_224', pretrained = True, num_classes = num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class CaiT(nn.Module):
    def __init__(self, num_classes):
        super(CaiT, self).__init__()
        self.model = timm.create_model('cait_xxs36_224', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x 
    
# torchvision.models
# Refactoring Needed

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu') 

def change_last_layer(model, num_classes):
    name_last_layer = list(model.named_modules())[-1][0]
    
    if name_last_layer == 'classifier':
        model.classifier = nn.Linear(in_features = model.classifier.in_features,
                                              out_features = num_classes, bias = True)
        initialize_weights(model.classifier)
        return model
    
    elif name_last_layer == 'fc':
        model.fc = nn.Linear(in_features = model.fc.in_features,
                                      out_features = num_classes, bias = True)
        initialize_weights(model.fc)
        return model
    
    else:
        raise Exceptionception('last layer should be nn.Linear Module named as either fc or classifier')
    
    
class DenseNet201(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201, self).__init__()
        self.model = models.densenet201(pretrained = True)
        self.model = change_last_layer(self.model, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    
class DenseNet161(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet161, self).__init__()
        self.model = models.densenet161(pretrained = True)
        self.model = change_last_layer(self.model, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

    
class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(pretrained = True)
        self.model = change_last_layer(self.model, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    
class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.inception_v3(pretrained = True)
        self.model = change_last_layer(self.model, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    
class Resnet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet152(pretrained = True)
        self.model = change_last_layer(self.model, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    
class ResNext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnext50_32x4d(pretrained = True)
        self.model = change_last_layer(self.model, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class CoralNet(nn.Module):
    def __init__(self, num_classes, grayscale):
        self.num_classes = num_classes
        super(CoralNet, self).__init__()
        self.model = models.resnet50(pretrained = True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float(), requires_grad=True)

    def forward(self, x):
        logits = self.model(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas

class CoralNet_EfficientNet(nn.Module):
    def __init__(self, num_classes, grayscale):
        self.num_classes = num_classes
        super(CoralNet_EfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes - 1)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float(), requires_grad=True)

    def forward(self, x):
        logits = self.model(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas

class CoralNet_Pretrained(nn.Module):
    def __init__(self, num_classes, grayscale):
        self.num_classes = num_classes
        super(CoralNet_Pretrained, self).__init__()
        self.model = resnet34(55, False)
        self.model.load_state_dict(torch.load('./model/Coral_ResNet_morph.pt', map_location=torch.device('cuda:0')))
        self.model.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float(), requires_grad=True)

    def forward(self, x):
        logits, probas = self.model(x)
        return logits, probas

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


if __name__ == "__main__":
    m = ThreeWayNet(18)
    print(m.model[2].linear_1_bias)