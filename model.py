import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

# effiecientnet 'efficientnet-b4' test
class effiecientnet_test(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


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
    
# timm library models
# Refactoring Needed

class ConvNextLIn22ft1kCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_large_in22ft1k', pretrained=True, num_classes = 1000)
        self.dropout = nn.Dropout(0.5)
        self.dropouts = nn.ModuleList([
                    nn.Dropout(0.5) for _ in range(5)])
        self.age_layer = nn.Linear(in_features=1000, out_features=3, bias=True)
        self.mask_layer = nn.Linear(in_features=1000, out_features=3, bias=True)
        self.sex_layer = nn.Linear(in_features=1000, out_features=2, bias=True)
    
    def forward(self, x):
        x = self.model(x)
        x_ = self.dropout(x)
        
        for i, dropout in enumerate(self.dropouts):
            if i==0:
                x_age = self.age_layer(dropout(x_))
                x_mask = self.mask_layer(dropout(x_))
                x_sex = self.sex_layer(dropout(x_))
            else:
                x_age += self.age_layer(dropout(x_))
                x_mask += self.mask_layer(dropout(x_))
                x_sex += self.sex_layer(dropout(x_))
        else:
            x_age /= len(self.dropouts)
            x_mask /= len(self.dropouts)
            x_sex /= len(self.dropouts)
        
        return x_age, x_mask, x_sex
    
class ConvNextLIn22Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('convnext_large_in22k', pretrained=True, num_classes = 1536)
        self.dropout = nn.Dropout(0.5)
        self.dropouts = nn.ModuleList([
                    nn.Dropout(0.5) for _ in range(5)])
        self.age_layer = nn.Linear(in_features=1536, out_features=3, bias=True)
        self.mask_layer = nn.Linear(in_features=1536, out_features=3, bias=True)
        self.sex_layer = nn.Linear(in_features=1536, out_features=2, bias=True)
    
    def forward(self, x):
        x = self.model(x)
        x_ = self.dropout(x)
        
        for i, dropout in enumerate(self.dropouts):
            if i==0:
                x_age = self.age_layer(dropout(x_))
                x_mask = self.mask_layer(dropout(x_))
                x_sex = self.sex_layer(dropout(x_))
            else:
                x_age += self.age_layer(dropout(x_))
                x_mask += self.mask_layer(dropout(x_))
                x_sex += self.sex_layer(dropout(x_))
        else:
            x_age /= len(self.dropouts)
            x_mask /= len(self.dropouts)
            x_sex /= len(self.dropouts)
        
        return x_age, x_mask, x_sex

class CoatLiteMini(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('coat_lite_mini', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class ConvNextLIn22ft1k(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('convnext_large_in22ft1k', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class ConvNextLIn22(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('convnext_large_in22k', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x    
 
class ConvNextBIn22ft1k(nn.Module):

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
        super(SWSRexnext, self).__init__()
        self.model = timm.create_model('swsl_resnext50_32x4d', pretrained = True, num_classes = num_classes)
        
    def forward(self, x):
        x = self.model
        return x
    
class SWSResnext(nn.Module):
    def __init__(self, num_classes):
        super(SWSRexnext, self).__init__()
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
        self.model = timm.create_model('cait_s24_224', pretrained=True, num_classes=num_classes)
        
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
    