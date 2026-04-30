import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    """
    BasicBlock: A fundamental building block for 3D Convolutional Neural Networks (CNNs).
    This block consists of two 3D convolutional layers, each followed by batch normalization and a ReLU activation.
    It is designed to extract hierarchical features from 3D input volumes.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the convolutional kernel.
    stride : int or tuple, optional (default=1)
        Stride of the convolution.
    padding : str or int or tuple, optional (default='same')
        Padding added to the input.
    dilation : int or tuple, optional (default=1)
        Spacing between kernel elements.
    groups : int, optional (default=1)
        Number of blocked connections from input to output channels.
    bias : bool, optional (default=True)
        If True, adds a learnable bias to the output.
    padding_mode : str, optional (default='zeros')
        Type of padding.
    device : torch.device, optional (default=None)
        Device to place the module on.
    dtype : torch.dtype, optional (default=None)
        Data type for the module.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First 3D convolutional layer
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode, device, dtype)

        # Second 3D convolutional layer
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride,
                  padding, dilation, groups, bias, padding_mode, device, dtype)

        # Batch normalization layers for stabilizing training
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # ReLU activation for non-linearity
        self.relu = nn.ReLU()

    def forward(self, input):
        """
        Forward pass of the BasicBlock.

        Parameters:
        -----------
        input : torch.Tensor
            Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        # Apply first convolution, batch normalization, and ReLU activation
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)

        # Apply second convolution, batch normalization, and ReLU activation
        input = self.conv2(input)
        input = self.bn2(input)
        input = self.relu(input)

        return input

class UNet3D(nn.Module):
    """
    UNet3D: A 3D U-Net architecture for volumetric segmentation tasks.
    This model consists of an encoder (downsampling path) and a decoder (upsampling path),
    with skip connections between corresponding layers to preserve spatial information.

    Parameters:
    -----------
    depth : int
        Number of layers in the encoder/decoder.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    between_channels : int, optional (default=64)
        Number of channels in the first layer; doubles with each downsampling.
    deep_supervision : int, optional (default=0)
        If > 0, enables deep supervision with intermediate outputs.
    size : int, optional (default=32)
        Initial size of the input volume (used for calculating padding in transposed convolutions).
    """

    def __init__(self, depth, in_channels, out_channels, between_channels=64, deep_supervision=0, size=32):
        super().__init__()
        self.depth = depth
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision

        # Calculate padding for transposed convolutions to ensure correct output size
        weights = []
        for i in range(depth):
            weights.append(int(size % 2))
            size = (size - size % 2) / 2
        weights.reverse()

        # Encoder path: downsampling with BasicBlocks and max pooling
        self.blocks_encoder = [BasicBlock(in_channels, between_channels, 3)]
        for i in range(1, self.depth):
            between_channels *= 2
            self.blocks_encoder.append(BasicBlock(between_channels // 2, between_channels, 3))

        self.blocks_encoder = nn.ModuleList(self.blocks_encoder)

        # Decoder path: upsampling with transposed convolutions and BasicBlocks
        self.blocks_decoder = []
        self.conv_transpose = []
        for i in range(1, self.depth):
            between_channels //= 2
            self.conv_transpose.append(nn.ConvTranspose3d(2 * between_channels, between_channels, 2, stride=2, output_padding=weights[i]))
            self.blocks_decoder.append(BasicBlock(2 * between_channels, between_channels, 3))

        self.blocks_decoder = nn.ModuleList(self.blocks_decoder)
        self.conv_transpose = nn.ModuleList(self.conv_transpose)

        # Max pooling for downsampling
        self.max_pool = nn.MaxPool3d(2)

        # Final 1x1 convolution to produce the output segmentation map
        self.conv_final = nn.Conv3d(between_channels, self.out_channels, 1)

        # Deep supervision: additional 1x1 convolutions for intermediate outputs
        if self.deep_supervision:
            self.conv_DS = []
            for i in range(self.deep_supervision):
                between_channels *= 2
                self.conv_DS.append(nn.Conv3d(between_channels, self.out_channels, 1))
            self.conv_DS = nn.ModuleList(self.conv_DS)

    def forward(self, input):
        """
        Forward pass of the UNet3D.

        Parameters:
        -----------
        input : torch.Tensor
            Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
        --------
        list of torch.Tensor
            List of output tensors. If deep supervision is enabled, includes intermediate outputs.
        """
        # Encoder path: apply BasicBlocks and store intermediate outputs for skip connections
        stack_input = []
        for block in self.blocks_encoder[:-1]:
            input = block(input)
            stack_input.append(input)
            input = self.max_pool(input)
        input = self.blocks_encoder[-1](input)

        # Deep supervision: store intermediate decoder outputs
        if self.deep_supervision:
            stack_DS = []

        # Decoder path: upsample, concatenate with skip connections, and apply BasicBlocks
        for i, block in enumerate(self.blocks_decoder):
            input = self.conv_transpose[i](input)
            input = torch.cat([input, stack_input.pop()], dim=1)
            input = block(input)

            # Store intermediate outputs for deep supervision
            if self.deep_supervision and i != len(self.blocks_decoder) - 1:
                stack_DS.append(input)
                if len(stack_DS) > self.deep_supervision:
                    stack_DS.pop(0)

        # Final 1x1 convolution
        input = self.conv_final(input)

        # Deep supervision: apply 1x1 convolutions to intermediate outputs
        if self.deep_supervision:
            stack_DS.reverse()
            for i, block in enumerate(self.conv_DS):
                stack_DS[i] = block(stack_DS[i])
            return [input, *stack_DS]

        return [input]