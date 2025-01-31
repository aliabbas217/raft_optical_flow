import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d, activation=nn.ReLU, affine=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if norm_layer is not None:
            self.norm = norm_layer(out_channels, eps=1e-05, momentum=0.1, affine=affine, track_running_stats=False)
        else:
            self.norm = nn.Identity()
        self.activation = activation(inplace=True) if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.convnormrelu1 = Conv2dNormActivation(in_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
        self.convnormrelu2 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.convnormrelu1(x)
        x = self.convnormrelu2(x)
        x = self.downsample(residual) + x 
        x = self.relu(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.convnormrelu = Conv2dNormActivation(3, 64, kernel_size=7, stride=2, padding=3, norm_layer=norm_layer)
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, norm_layer=norm_layer),
            ResidualBlock(64, 64, norm_layer=norm_layer)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 96, downsample=Conv2dNormActivation(64, 96, kernel_size=1, stride=2, norm_layer=norm_layer), norm_layer=norm_layer),
            ResidualBlock(96, 96, norm_layer=norm_layer)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(96, 128, downsample=Conv2dNormActivation(96, 128, kernel_size=1, stride=2, norm_layer=norm_layer), norm_layer=norm_layer),
            ResidualBlock(128, 128, norm_layer=norm_layer)
        )
        self.conv = nn.Conv2d(128, 256, kernel_size=1)

    def forward(self, x):
        x = self.convnormrelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv(x)
        return x


class CorrBlock(nn.Module):
    def __init__(self, num_levels=4, radius=4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

    def forward(self, f1, f2):
        b, c, h, w = f1.shape
        max_disp = self.radius
        corr_pyramid = []

        for i in range(self.num_levels):
            stride = 2 ** i
            padded_f2 = F.pad(f2, (0, max_disp * stride, 0, max_disp * stride))
            corr = F.conv2d(f1, padded_f2.view(b, c, 1, 1), groups=b, stride=1)
            corr = corr.view(b, 1, h, w, (2 * max_disp + 1) ** 2)
            corr = F.softmax(corr, dim=-1)
            corr_pyramid.append(corr)

        return corr_pyramid



class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=(3,3)):
        super().__init__()
        self.convz = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=tuple((k-1)//2 for k in kernel_size))
        self.convr = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=tuple((k-1)//2 for k in kernel_size))
        self.convq = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=tuple((k-1)//2 for k in kernel_size))

    def forward(self, x, h):
        z = torch.sigmoid(self.convz(torch.cat([x, h], dim=1)))
        r = torch.sigmoid(self.convr(torch.cat([x, h], dim=1)))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        h = (1 - z) * h + z * q
        return h


class RecurrentBlock(nn.Module):
    def __init__(self, in_channels=126, hidden_channels=128):
        super().__init__()
        self.convgru1 = ConvGRU(in_channels, hidden_channels, kernel_size=(1,5))
        self.convgru2 = ConvGRU(in_channels, hidden_channels, kernel_size=(5,1))

    def forward(self, x, h):
         h = self.convgru1(x, h)
         h = self.convgru2(x, h)
         return h


class MotionEncoder(nn.Module):
    def __init__(self, in_channels=324):
        super().__init__()
        self.convcorr1 = Conv2dNormActivation(in_channels, 256, kernel_size=1)
        self.convcorr2 = Conv2dNormActivation(256, 192, kernel_size=3, padding=1)
        self.convflow1 = Conv2dNormActivation(2, 128, kernel_size=7, padding=3)
        self.convflow2 = Conv2dNormActivation(128, 64, kernel_size=3, padding=1)
        self.conv = Conv2dNormActivation(256, 126, kernel_size=3, padding=1)

    def forward(self, corr, flow):
        out_corr = self.convcorr1(corr)
        out_corr = self.convcorr2(out_corr)
        out_flow = self.convflow1(flow)
        out_flow = self.convflow2(out_flow)
        out = torch.cat([out_corr, out_flow], dim=1)
        out = self.conv(out)
        return out


class FlowHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class UpdateBlock(nn.Module):
    def __init__(self, in_channels=126, hidden_dim=128import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d, activation=nn.ReLU, affine=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if norm_layer is not None:
            self.norm = norm_layer(out_channels, eps=1e-05, momentum=0.1, affine=affine, track_running_stats=False)
        else:
            self.norm = nn.Identity()
        self.activation = activation(inplace=True) if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.convnormrelu1 = Conv2dNormActivation(in_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
        self.convnormrelu2 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.convnormrelu1(x)
        x = self.convnormrelu2(x)
        x = self.downsample(residual) + x
        x = self.relu(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.convnormrelu = Conv2dNormActivation(3, 64, kernel_size=7, stride=2, padding=3, norm_layer=norm_layer)
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, norm_layer=norm_layer),
            ResidualBlock(64, 64, norm_layer=norm_layer)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 96, downsample=Conv2dNormActivation(64, 96, kernel_size=1, stride=2, norm_layer=norm_layer), norm_layer=norm_layer),
            ResidualBlock(96, 96, norm_layer=norm_layer)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(96, 128, downsample=Conv2dNormActivation(96, 128, kernel_size=1, stride=2, norm_layer=norm_layer), norm_layer=norm_layer),
            ResidualBlock(128, 128, norm_layer=norm_layer)
        )
        self.conv = nn.Conv2d(128, 256, kernel_size=1)

    def forward(self, x):
        x = self.convnormrelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv(x)
        return x


class CorrBlock(nn.Module):
    def __init__(self, num_levels=4, radius=4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

    def forward(self, f1, f2):
        b, c, h, w = f1.shape
        max_disp = self.radius
        corr_pyramid = []

        for i in range(self.num_levels):
            stride = 2 ** i
            padded_f2 = F.pad(f2, (0, max_disp * stride, 0, max_disp * stride))
            corr = F.conv2d(f1, padded_f2.view(b, c, 1, 1), groups=b, stride=1)
            corr = corr.view(b, 1, h, w, (2 * max_disp + 1) ** 2)
            corr = F.softmax(corr, dim=-1)
            corr_pyramid.append(corr)

        return corr_pyramid



class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=(3,3)):
        super().__init__()
        self.convz = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=tuple((k-1)//2 for k in kernel_size))
        self.convr = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=tuple((k-1)//2 for k in kernel_size))
        self.convq = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=tuple((k-1)//2 for k in kernel_size))

    def forward(self, x, h):
        z = torch.sigmoid(self.convz(torch.cat([x, h], dim=1)))
        r = torch.sigmoid(self.convr(torch.cat([x, h], dim=1)))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        h = (1 - z) * h + z * q
        return h


class RecurrentBlock(nn.Module):
    def __init__(self, in_channels=126, hidden_channels=128):
        super().__init__()
        self.convgru1 = ConvGRU(in_channels, hidden_channels, kernel_size=(1,5))
        self.convgru2 = ConvGRU(in_channels, hidden_channels, kernel_size=(5,1))

    def forward(self, x, h):
         h = self.convgru1(x, h)
         h = self.convgru2(x, h)
         return h


class MotionEncoder(nn.Module):
    def __init__(self, in_channels=324):
        super().__init__()
        self.convcorr1 = Conv2dNormActivation(in_channels, 256, kernel_size=1)
        self.convcorr2 = Conv2dNormActivation(256, 192, kernel_size=3, padding=1)
        self.convflow1 = Conv2dNormActivation(2, 128, kernel_size=7, padding=3)
        self.convflow2 = Conv2dNormActivation(128, 64, kernel_size=3, padding=1)
        self.conv = Conv2dNormActivation(256, 126, kernel_size=3, padding=1)

    def forward(self, corr, flow):
        out_corr = self.convcorr1(corr)
        out_corr = self.convcorr2(out_corr)
        out_flow = self.convflow1(flow)
        out_flow = self.convflow2(out_flow)
        out = torch.cat([out_corr, out_flow], dim=1)
        out = self.conv(out)
        return out


class FlowHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class UpdateBlock(nn.Module):
    def __init__(self, in_channels=126, hidden_dim=128):
        super().__init__()
        self.motion_encoder = MotionEncoder(in_channels)
        self.recurrent_block = RecurrentBlock(in_channels=in_channels, hidden_channels=hidden_dim)
        self.flow_head = FlowHead(hidden_dim)

        self.mask_predictor = MaskPredictor(hidden_dim)

    def forward(self, corr_pyramid, flow, hidden):
        corr = torch.cat(corr_pyramid, dim=1)
        motion_features = self.motion_encoder(corr, flow)
        hidden = self.recurrent_block(motion_features, hidden)
        flow_delta = self.flow_head(hidden)
        flow = flow + flow_delta

        mask = self.mask_predictor(hidden)
        return flow, hidden, mask


class MaskPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.convrelu = Conv2dNormActivation(in_channels, 256, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(256, 6, kernel_size=1)

    def forward(self, x):
        x = self.convrelu(x)
        x = self.conv(x)
        return x


class RAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder(norm_layer=nn.InstanceNorm2d)
        self.context_encoder = FeatureEncoder(norm_layer=nn.BatchNorm2d)
        self.corr_block = CorrBlock()
        self.update_block = UpdateBlock()

    def forward(self, image1, image2, iters=12, flow_init=None):
        f1 = self.feature_encoder(image1)
        f2 = self.feature_encoder(image2)
        c = self.context_encoder(image1)

        corr_pyramid = self.corr_block(f1, f2)

        b, _, h, w = f1.shape
        if flow_init is None:
            flow = torch.zeros((b, 2, h, w)).to(image1.device)
        else:
            flow = flow_init

        hidden = torch.zeros((b, 128, h, w)).to(image1.device)

        flow_predictions = []
        mask_predictions = []

        for i in range(iters):
            flow, hidden, mask = self.update_block(corr_pyramid, flow, hidden)
            flow_predictions.append(flow)
            mask_predictions.append(mask)


        return flow_predictions, mask_predictions
