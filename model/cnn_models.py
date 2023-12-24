import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch import nn
from torchvision import models

from model.resnet import deeplabv3_resnet101


class RoomLayoutUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=11):
        super(RoomLayoutUNet, self).__init__()
        self.model = deeplabv3_resnet101(num_classes=4, progress=False)
        self.model.backbone.conv1 = nn.Conv2d(n_channels, 64,
                                              kernel_size=(7, 7),
                                              stride=(2, 2), padding=(3, 3),
                                              bias=False)
        self.model.classifier[4] = nn.Conv2d(256, n_classes,
                                             kernel_size=(1, 1), stride=(1, 1))
        self.embedding = nn.Embedding(5, 1)

    def forward(self, x):
        output = self.model(x)['out']
        return output


class DecoderCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2560, 2048, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2048)

        self.deconv1 = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2,
                                          padding=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,
                                          padding=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
                                          padding=1, output_padding=0)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn3(self.deconv1(x)))
        x = F.relu(self.bn4(self.deconv2(x)))
        x = F.relu(self.bn5(self.deconv3(x)))
        x = torch.sigmoid(self.conv3(x))

        return x


class Encoder(nn.Module):
    def __init__(self, inch=3, outch=512):
        super().__init__()
        self.conv1 = nn.Conv2d(inch, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, outch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


class CentroidRegressor(pl.LightningModule):
    def __init__(self, inch=12, state=None):
        super().__init__()

        self.example_input_2 = None
        self.example_input_1 = None
        self.state = state

        model = RoomLayoutUNet()
        model.model.backbone.conv1 = nn.Conv2d(inch, 64,
                                               kernel_size=(7, 7), bias=False,
                                               stride=(2, 2), padding=(3, 3))
        cent_model_encoder = torch.nn.Sequential(
            *(list(model.model.children())[:-2]))

        self.encoder_main = cent_model_encoder
        self.encoder_mini = Encoder(inch=3)

        self.decoder_aio = DecoderCNN()
        self.decoder_bed = DecoderCNN()
        self.decoder_bath = DecoderCNN()
        self.decoder_kit = DecoderCNN()

        self.loss_func = nn.MSELoss()

    def set_state(self, state):
        self.state = state

    def forward_encoder_main(self, x):
        return self.encoder_main(x)["out"]

    def forward_encoder_mini(self, x):
        return self.encoder_mini(x)

    def forward_decoder(self, x1, x2, state=None):
        x = torch.cat([x1, x2], dim=1)
        if state == None:
            x = self.decoder_aio(x)
        if state == 1:
            x = self.decoder_bed(x)
        elif state == 2:
            x = self.decoder_bath(x)
        elif state == 3:
            x = self.decoder_kit(x)
        return x

    def forward(self, x1, x2):
        x1 = self.encoder_main(x1)["out"]
        x2 = self.encoder_mini(x2)
        x = torch.cat([x1, x2], dim=1)

        if self.state is None:
            x = self.decoder_aio(x)
        if self.state == 1:
            x = self.decoder_bed(x)
        elif self.state == 2:
            x = self.decoder_bath(x)
        elif self.state == 3:
            x = self.decoder_kit(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                              gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }

    def training_step(self, batch, batch_idx):
        inputs_1, inputs_2, labels, state = batch
        outputs = self(inputs_1, inputs_2)
        loss = self.loss_func(outputs, labels)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs_1, inputs_2, labels, state = batch
        outputs = self(inputs_1, inputs_2)
        loss = self.loss_func(outputs, labels)
        self.log("val_loss", loss, on_epoch=True)

        # store an example output and label
        self.example_input_1 = inputs_1[0]
        self.example_input_2 = inputs_2[0]
        self.example_output = outputs[0]
        self.example_label = labels[0]

        return loss

    def on_train_epoch_end(self):
        print(f"Epoch: {self.current_epoch}")
        print(f"Train Loss: {self.trainer.callback_metrics['train_loss']}")
        print(f"Validation Loss: {self.trainer.callback_metrics['val_loss']}")

        plt.imshow(self.example_input_1.detach().cpu().permute((1, 2, 0))[:, :,
                   [1, 2, 3]])
        plt.show()
        plt.imshow(self.example_input_2.detach().cpu().permute((1, 2, 0))[:, :,
                   [0, 1, 2]])
        plt.show()
        plt.imshow(
            self.example_label.detach().cpu().permute((1, 2, 0))[:, :, :])
        plt.show()
        plt.imshow(
            self.example_output.detach().cpu().permute((1, 2, 0))[:, :, 0])
        plt.show()


class XXYYRegressor(torch.nn.Module):
    def __init__(self, model=None, criterion=None, use_res=True):
        super().__init__()

        self.example_label = None
        self.example_output = None
        if model is None:
            if use_res:
                model = models.resnet50(pretrained=False)
                model.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7),
                                        stride=(2, 2), padding=(3, 3),
                                        bias=False)
                model.fc = nn.Linear(2048, 4)
            else:
                model = models.mobilenet_v3_small(pretrained=False)
                model.features[0][0] = nn.Conv2d(5, 16, kernel_size=(3, 3),
                                                 stride=(2, 2), padding=(1, 1),
                                                 bias=False)
                model.classifier[3] = nn.Linear(in_features=1024,
                                                out_features=4, bias=True)
        self.model = model
        if criterion is None:
            criterion = nn.MSELoss()
        self.loss_func = criterion

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                              gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }

    def training_step(self, batch, batch_idx):
        try:
            inputs, labels = batch
            labels = labels.float() / 256.0
            outputs = self(inputs)
            loss = self.loss_func(outputs, labels)
            self.log("train_loss", loss, on_epoch=True)
            return loss
        except:
            return 0

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        labels = labels.float() / 256.0
        outputs = self(inputs)
        loss = self.loss_func(outputs, labels)
        self.log("val_loss", loss, on_epoch=True)

        # store an example output and label
        self.example_output = outputs[0]
        self.example_label = labels[0]

        return loss

    def on_train_epoch_end(self):
        print(f"Epoch: {self.current_epoch}")
        print(f"Train Loss: {self.trainer.callback_metrics['train_loss']}")
        print(f"Validation Loss: {self.trainer.callback_metrics['val_loss']}")

        # print the example output and label
        print(f"Example Output: {self.example_output * 256}")
        print(f"Example Label: {self.example_label * 256}")


class BedBathCountsRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet34(num_classes=2)
        self.model.conv1 = nn.Conv2d(2, 64,
                                     kernel_size=(7, 7), stride=(2, 2),
                                     padding=(3, 3), bias=False)
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                              gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
            }
        }

    def training_step(self, batch, batch_idx):
        inputs_1, labels = batch
        outputs = self(inputs_1)
        loss = self.loss_func(outputs, labels)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs_1, labels = batch
        outputs = self(inputs_1)
        loss = self.loss_func(outputs, labels)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        print(f"Epoch: {self.current_epoch}")
        print(f"Train Loss: {self.trainer.callback_metrics['train_loss']}")
        print(f"Validation Loss: {self.trainer.callback_metrics['val_loss']}")

        plt.imshow(self.example_input_1.detach().cpu().permute((1, 2, 0))[:, :,
                   [0, 1, 1]])
        plt.show()

        print("label: ", self.example_label)
        print("output: ", self.example_output)
