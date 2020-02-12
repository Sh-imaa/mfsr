""" Pytorch implementation of HRNet, a neural network for multi-frame super resolution (MFSR) by recursive fusion. """

import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3, dr=0.0):
        '''
        Args:
            channel_size : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
            dr : float, dropout rate
        '''
        
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.Dropout(p=dr),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.Dropout(p=dr),
            nn.PReLU()
        )

    def forward(self, x):
        '''
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        '''
        
        residual = self.block(x)
        return x + residual


class Encoder(nn.Module):
    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''
        
        super(Encoder, self).__init__()

        in_channels = config["in_channels"]
        num_layers = config["num_layers"]
        kernel_size = config["kernel_size"]
        channel_size = config["channel_size"]
        dr = config["dropout"]
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.Dropout(p=dr),
            nn.PReLU())

        res_layers = [ResidualBlock(channel_size, kernel_size, dr) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        '''
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        '''
        
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x


class RecuversiveNet(nn.Module):

    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''
        
        super(RecuversiveNet, self).__init__()

        self.input_channels = config["in_channels"]
        self.num_layers = config["num_layers"]
        self.alpha_residual = config["alpha_residual"]
        dr = config["dropout"]
        kernel_size = config["kernel_size"]
        padding = kernel_size // 2
        self.attention = config["attention"]
        if self.attention:
            in_channels = 2 * self.input_channels + 2
        else:
            in_channels = 2 * self.input_channels

        self.fuse = nn.Sequential(
            ResidualBlock(in_channels , kernel_size),
            nn.Conv2d(in_channels=in_channels, out_channels=self.input_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.Dropout(p=dr),
            nn.PReLU())

    def forward(self, x, alphas):
        '''
        Fuses hidden states recursively.
        Args:
            x : tensor (B, L, C, W, H), hidden states
            alphas : tensor (B, L, 1, 1, 1), boolean indicator (0 if padded low-res view, 1 otherwise)
        Returns:
            out: tensor (B, C, W, H), fused hidden state
        '''
        
        batch_size, nviews, channels, width, heigth = x.shape
        parity = nviews % 2
        half_len = nviews // 2
        
        while half_len > 0:
            alice = x[:, :half_len] # first half hidden states (B, L/2, C, W, H)
            bob = x[:, half_len:nviews - parity] # second half hidden states (B, L/2, C, W, H)
            bob = torch.flip(bob, [1])

            if self.attention == True:
                alphas_alice = alphas[:, :half_len]
                alphas_bob = alphas[:, half_len:nviews - parity]
                alphas_bob = torch.flip(alphas_bob, [1])

                alice_and_bob = torch.cat([alice, alphas_alice, bob, alphas_bob], 2)
                alice_and_bob = alice_and_bob.view(-1, (2 * channels + 2), width, heigth)
                x = self.fuse(alice_and_bob)
                x = x.view(batch_size, half_len, channels, width, heigth)
                alphas_ = torch.stack([alphas_alice, alphas_bob])
                alphas = torch.max(alphas_, dim=0)[0]
            if self.alpha_residual == 'padding':
                x = alice + alphas_bob * x

            else:
                alice_and_bob = torch.cat([alice, bob], 2)  # concat hidden states accross channels (B, L/2, 2*C, W, H)
                alice_and_bob = alice_and_bob.view(-1, 2 * channels, width, heigth)
                x = self.fuse(alice_and_bob)
                x = x.view(batch_size, half_len, channels, width, heigth)  # new hidden states (B, L/2, C, W, H)

                if self.alpha_residual == 'padding': # skip connect padded views (alphas_bob = 0)
                    alphas_alice = alphas[:, :half_len]
                    alphas_bob = alphas[:, half_len:nviews - parity]
                    alphas_bob = torch.flip(alphas_bob, [1])
                    x = alice + alphas_bob * x
                    alphas = alphas_alice

                elif self.alpha_residual == 'weighted':
                    alphas_alice = alphas[:, :half_len]
                    alphas_bob = alphas[:, half_len:nviews - parity]
                    alphas_bob = torch.flip(alphas_bob, [1])
                    alphas_ = torch.stack([alphas_alice, alphas_bob])
                    x = alice * alphas_alice + bob * alphas_bob + x * torch.min(alphas_, dim=0)[0]
                    alphas = torch.max(alphas_, dim=0)[0]

                elif self.alpha_residual == 'weight_maps':
                    alphas_alice = alphas[:, :half_len]
                    alphas_bob = alphas[:, half_len:nviews - parity]
                    alphas_bob = torch.flip(alphas_bob, [1])
                    alphas_ = torch.stack([alphas_alice, alphas_bob])
                    x = alice * alphas_alice + bob * alphas_bob + x * torch.min(alphas_, dim=0)[0]
                    alphas = torch.max(alphas_, dim=0)[0]

            nviews = half_len
            parity = nviews % 2
            half_len = nviews // 2

        return torch.mean(x, 1)



class Decoder(nn.Module):
    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''
        
        super(Decoder, self).__init__()

        dr = config["dropout"]
        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels=config["deconv"]["in_channels"],
                                                       out_channels=config["deconv"]["out_channels"],
                                                       kernel_size=config["deconv"]["kernel_size"],
                                                       stride=config["deconv"]["stride"]),
                                    nn.Dropout(p=dr),
                                    nn.PReLU())

        self.final = nn.Conv2d(in_channels=config["final"]["in_channels"],
                               out_channels=config["final"]["out_channels"],
                               kernel_size=config["final"]["kernel_size"],
                               padding=config["final"]["kernel_size"] // 2)

    def forward(self, x):
        '''
        Decodes a hidden state x.
        Args:
            x : tensor (B, C, W, H), hidden states
        Returns:
            out: tensor (B, C_out, 3*W, 3*H), fused hidden state
        '''
        
        x = self.deconv(x)
        x = self.final(x)
        return x


class HRNet(nn.Module):
    ''' HRNet, a neural network for multi-frame super resolution (MFSR) by recursive fusion. '''

    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''

        super(HRNet, self).__init__()
        self.encode = Encoder(config["encoder"])
        self.fuse = RecuversiveNet(config["recursive"])
        self.decode = Decoder(config["decoder"])

    def forward(self, lrs, alphas, weight_maps=None, pixels_weights=False, extra_channel=False):
        '''
        Super resolves a batch of low-resolution images.
        Args:
            lrs : tensor (B, L, W, H), low-resolution images
            alphas : tensor (B, L), boolean indicator (0 if padded low-res view, 1 otherwise)
        Returns:
            srs: tensor (B, C_out, W, H), super-resolved images
        '''

        batch_size, seq_len, heigth, width = lrs.shape
        lrs = lrs.view(-1, seq_len, 1, heigth, width)
        if pixels_weights:
            alphas = alphas.view(-1, seq_len, 1, heigth, width)
        else:
            alphas = alphas.view(-1, seq_len, 1, 1, 1)


        if extra_channel:
            weight_maps = weight_maps.view(-1, seq_len, 1, heigth, width)
            refs = torch.mul(weight_maps, lrs).mean(dim=1)
            refs = refs.repeat(1, seq_len, 1, 1).unsqueeze(2)
            stacked_input = torch.cat([lrs, refs, weight_maps], 2)
            stacked_input = stacked_input.view(batch_size * seq_len, 3, width, heigth)
        else:
            refs, _ = torch.median(lrs[:, :9], 1, keepdim=True)  # reference image aka anchor, shared across multiple views
            refs = refs.repeat(1, seq_len, 1, 1, 1)
            stacked_input = torch.cat([lrs, refs], 2) # tensor (B, L, 2*C_in, W, H)
            stacked_input = stacked_input.view(batch_size * seq_len, 2, width, heigth)
        
        
        layer1 = self.encode(stacked_input) # encode input tensor
        layer1 = layer1.view(batch_size, seq_len, -1, width, heigth) # tensor (B, L, C, W, H)

        # fuse, upsample
        recursive_layer = self.fuse(layer1, alphas)  # fuse hidden states (B, C, W, H)
        srs = self.decode(recursive_layer)  # decode final hidden state (B, C_out, 3*W, 3*H)
        return srs
