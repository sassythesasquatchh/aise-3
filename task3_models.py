import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import ipdb

torch.manual_seed(0)
np.random.seed(0)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]
        # hint: use torch.fft library torch.fft.rfft
        # use DFT to approximate the fourier transform

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 1  # pad the domain if input is non-periodic
        self.linear_p = nn.Linear(1, self.width)

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # ipdb.set_trace()
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)

        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        # x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)

        x = self.linear_layer(x, self.linear_q)
        x = self.output_layer(x)
        return x


# ---------------------
# Activation Function:
# ---------------------


class CNO_LReLu(nn.Module):
    def __init__(self, in_size, out_size):
        super(CNO_LReLu, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x):

        x = F.interpolate(
            x.unsqueeze(2), size=(1, 2 * self.in_size), mode="bicubic", antialias=True
        )
        x = self.act(x)
        x = F.interpolate(x, size=(1, self.out_size), mode="bicubic", antialias=True)

        return x[:, :, 0]


# --------------------
# CNO Block:
# --------------------


class CNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, out_size, use_bn=True):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size

        # -----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        self.convolution = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )

        if use_bn:
            self.batch_norm = nn.BatchNorm1d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()
        self.act = CNO_LReLu(in_size=self.in_size, out_size=self.out_size)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)


# --------------------
# Lift/Project Block:
# --------------------


class LiftProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, latent_dim=64):
        super(LiftProjectBlock, self).__init__()

        self.inter_CNOBlock = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=size,
            out_size=size,
            use_bn=False,
        )

        self.convolution = torch.nn.Conv1d(
            in_channels=latent_dim, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x


# --------------------
# Residual Block:
# --------------------


class ResidualBlock(nn.Module):
    def __init__(self, channels, size, use_bn=True):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size = size

        # -----------------------------------------

        # We apply Conv -> BN (optional) -> Activation -> Conv -> BN (optional) -> Skip Connection
        # Up/Downsampling happens inside Activation

        self.convolution1 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
        )
        self.convolution2 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
        )

        if use_bn:
            self.batch_norm1 = nn.BatchNorm1d(self.channels)
            self.batch_norm2 = nn.BatchNorm1d(self.channels)

        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        self.act = CNO_LReLu(in_size=self.size, out_size=self.size)

    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out


# --------------------
# ResNet:
# --------------------


class ResNet(nn.Module):
    def __init__(self, channels, size, num_blocks, use_bn=True):
        super(ResNet, self).__init__()

        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks

        self.res_nets = []
        for _ in range(self.num_blocks):
            self.res_nets.append(
                ResidualBlock(channels=channels, size=size, use_bn=use_bn)
            )

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.res_nets[i](x)
        return x


# --------------------
# CNO:
# --------------------


class CNO1d(nn.Module):
    def __init__(
        self,
        in_dim,  # Number of input channels.
        out_dim,  # Number of input channels.
        size,  # Input and Output spatial size (required )
        N_layers,  # Number of (D) or (U) blocks in the network
        N_res=4,  # Number of (R) blocks per level (except the neck)
        N_res_neck=4,  # Number of (R) blocks in the neck
        channel_multiplier=16,  # How the number of channels evolve?
        use_bn=True,  # Add BN? We do not add BN in lifting/projection layer
    ):

        super(CNO1d, self).__init__()

        self.N_layers = int(N_layers)  # Number od (D) & (U) Blocks
        self.lift_dim = (
            channel_multiplier // 2
        )  # Input is lifted to the half of channel_multiplier dimension
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_multiplier = channel_multiplier  # The growth of the channels

        ######## Num of channels/features - evolution ########

        self.encoder_features = [
            self.lift_dim
        ]  # How the features in Encoder evolve (number of features)
        for i in range(self.N_layers):
            self.encoder_features.append(2**i * self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[
            1:
        ]  # How the features in Decoder evolve (number of features)
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = (
                2 * self.decoder_features_in[i]
            )  # Pad the outputs of the resnets (we must multiply by 2 then)

        ######## Spatial sizes of channels - evolution ########

        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(size // 2**i)
            self.decoder_sizes.append(size // 2 ** (self.N_layers - i))

        ######## Define Lift and Project blocks ########

        self.lift = LiftProjectBlock(
            in_channels=in_dim, out_channels=self.encoder_features[0], size=size
        )

        self.project = LiftProjectBlock(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=out_dim,
            size=size,
        )

        ######## Define Encoder, ED Linker and Decoder networks ########

        self.encoder = nn.ModuleList(
            [
                (
                    CNOBlock(
                        in_channels=self.encoder_features[i],
                        out_channels=self.encoder_features[i + 1],
                        in_size=self.encoder_sizes[i],
                        out_size=self.encoder_sizes[i + 1],
                        use_bn=use_bn,
                    )
                )
                for i in range(self.N_layers)
            ]
        )

        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion = nn.ModuleList(
            [
                (
                    CNOBlock(
                        in_channels=self.encoder_features[i],
                        out_channels=self.encoder_features[i],
                        in_size=self.encoder_sizes[i],
                        out_size=self.decoder_sizes[self.N_layers - i],
                        use_bn=use_bn,
                    )
                )
                for i in range(self.N_layers + 1)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                (
                    CNOBlock(
                        in_channels=self.decoder_features_in[i],
                        out_channels=self.decoder_features_out[i],
                        in_size=self.decoder_sizes[i],
                        out_size=self.decoder_sizes[i + 1],
                        use_bn=use_bn,
                    )
                )
                for i in range(self.N_layers)
            ]
        )

        ####################### Define ResNets Blocks ################################################################

        # Here, we define ResNet Blocks.

        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet networks (before the neck)
        for l in range(self.N_layers):
            self.res_nets.append(
                ResNet(
                    channels=self.encoder_features[l],
                    size=self.encoder_sizes[l],
                    num_blocks=self.N_res,
                    use_bn=use_bn,
                )
            )

        self.res_net_neck = ResNet(
            channels=self.encoder_features[self.N_layers],
            size=self.encoder_sizes[self.N_layers],
            num_blocks=self.N_res_neck,
            use_bn=use_bn,
        )

        self.res_nets = torch.nn.Sequential(*self.res_nets)

    def forward(self, x):

        x = self.lift(x)  # Execute Lift
        skip = []

        # Execute Encoder
        for i in range(self.N_layers):

            # Apply ResNet & save the result
            y = self.res_nets[i](x)
            skip.append(y)

            # Apply (D) block
            x = self.encoder[i](x)

        # Apply the deepest ResNet (bottle neck)
        x = self.res_net_neck(x)

        # Execute Decode
        for i in range(self.N_layers):

            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)  # BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])), 1)

            # Apply (U) block
            x = self.decoder[i](x)

        # Cat & Execute Projetion
        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)
        x = self.project(x)

        return x
