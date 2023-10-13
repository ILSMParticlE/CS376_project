import sys
import torch
import torch.nn as nn
from torchvision import models
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TLmodel(LightningModule):
    def __init__(self, 
                model_name='resnet34', 
                pretrained=False, 
                num_classes=10,
                optimizer='adam', 
                learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        if model_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            num_filters = backbone.fc.in_features
            self.classifier = nn.Linear(num_filters, num_classes)
        elif model_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            num_filters = backbone.fc.in_features
            self.classifier = nn.Linear(num_filters, num_classes)
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            num_filters = backbone.fc.in_features
            self.classifier = nn.Linear(num_filters, num_classes)
        elif model_name == 'resnext':
            backbone = models.resnext50_32x4d(pretrained=pretrained)
            num_filters = backbone.fc.in_features
            self.classifier = nn.Linear(num_filters, num_classes)
        elif model_name == 'vgg16':
            backbone = models.vgg16_bn(pretrained=pretrained)
            num_filters = backbone.classifier[6].in_features
            backbone.classifier[6] = nn.Linear(num_filters, num_classes)
            self.classifier = backbone.classifier[:]
        else:
            print('wrong model')
            sys.exit()

        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        
        if optimizer == 'adam':
            self.configure_optimizers = self.set_adam
        elif optimizer == 'sgd_pure':
            self.configure_optimizers = self.set_sgd_pure
        elif optimizer == 'sgd_schedule':
            self.configure_optimizers = self.set_sgd_schedule
        else:
            print('wrong optimizer')
            sys.exit()

        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = torch.sum(preds == y)/x.size(0)
        self.log('train_loss', loss)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_loss.append(avg_loss)
        self.train_acc.append(avg_acc)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        with torch.no_grad():
            out = self.forward(x)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = torch.sum(preds == y)/x.size(0)
        self.log('val_loss', loss)
        return {'val_loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.val_loss.append(avg_loss)
        self.val_acc.append(avg_acc)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, fname = batch
        with torch.no_grad():
            output = self.forward(x)
        result = dict({'output': output, 'fname': fname})
        return result

    def set_adam(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def set_sgd_pure(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer
    
    def set_sgd_schedule(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = {
            'scheduler': lr_scheduler, 
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }  
        return [optimizer], [scheduler]
    


# def load_model(model_name='resnet34', num_classes=10, pretrained=False):
#     if model_name == 'vgg11':
#         model = models.vgg11_bn(pretrained=pretrained)
#         if pretrained:
#             for param in model.parameters():
#                 param.requires_grad = False
#         num_ftrs = model.classifier[6].in_features
#         model.classifier[6] = nn.Linear(num_ftrs, num_classes)
#     elif model_name == 'resnet18':
#         model = models.resnet18(pretrained=pretrained) 
#         if pretrained:
#             for param in model.parameters():
#                 param.requires_grad = False
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, 10) 
#     elif model_name == 'resnet34':
#         model = models.resnet34(pretrained=pretrained) 
#         if pretrained:
#             for param in model.parameters():
#                 param.requires_grad = False
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, 10) 
#     else:
#         print('Wrong model')
#         sys.exit()
#     return model