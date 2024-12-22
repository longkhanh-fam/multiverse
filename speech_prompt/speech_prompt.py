import torch
import torch.nn as nn
from Modules import *
class SpeechPromptEncoder(nn.Module):
    """
    Speech Prompt Encoder
    - Mel-Spec Pre-net:
        3x
            -  Conv1D
            -  ReLU
            -  LayerNorm
            -  Dropout
    - Positional Encoding
    - 4x FFT Blocks


    """
    def __init__(self, config):
        
        super(SpeechPromptEncoder, self).__init__()
        self.config = config

        """Conv1D params
        """
        self.mel_channels = config.mel_channels
        self.conv_channels = config.conv_channels
        self.conv_kernel = config.conv_kernels
        self.dropout = config.dropout
        self.hidden_embed_dim = config.hidden_embed_dim

        """FFT Blocks params
        """
        self.nb_block = config.nb_block

        """"""
        self.pre_convs = nn.Sequential(
            ConvNorm1D(in_channels=self.mel_channels, out_channels=self.conv_channels, 
                       kernel_size=self.conv_kernel, padding=int((self.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(self.conv_channels),
            nn.Dropout(self.dropout),

            ConvNorm1D(in_channels=self.conv_channels, out_channels=self.conv_channels, 
                       kernel_size=self.conv_kernel, padding=int((self.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(self.conv_channels),
            nn.Dropout(self.dropout),
            
            ConvNorm1D(in_channels=self.conv_channels, out_channels=self.conv_channels, 
                       kernel_size=self.hidden_embed_dim, padding=int((self.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.ReLU(),
            nn.LayerNorm(self.conv_channels),
            nn.Dropout(self.dropout)
        )

        self.pos_enc = PositionalEncoding(self.hidden_embed_dim)
        fft_blocks = []
        for  _ in range(self.nb_block):
            fft_blocks.append(FFTBlock(config))
    def forward(self, mel_specs: torch.Tensor):
        
        ''' Forward function of Prompt Speech Encoder:
            
            mel_specs = (B, nb_mels, T_max)
            speaker_ids = (B, )
            output_lengths = (B, )
        '''
        mel_specs = self.pre_convs(mel_specs.transpose(1,2)).transpose(1,2)
        pass

