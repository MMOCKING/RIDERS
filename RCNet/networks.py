import torch
from utils import net_utils
import torchvision
from linear_attention import LocalFeatureTransformer
'''
Encoders
'''


class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections
    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(ResNetEncoder, self).__init__()

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        # Resolution 1/1 -> 1/2
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]

        self.blocks2 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]

        self.blocks3 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]

        self.blocks4 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]

        self.blocks5 = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]

            self.blocks6 = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            self.blocks6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]

            self.blocks7 = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            self.blocks7 = None

    def _make_layer(self,
                    network_block,
                    n_block,
                    in_channels,
                    out_channels,
                    stride,
                    weight_initializer,
                    activation_func,
                    use_batch_norm):
        '''
        Creates a layer
        Arg(s):
            network_block : Object
                block type
            n_block : int
                number of blocks to use in layer
            in_channels : int
                number of channels
            out_channels : int
                number of output channels
            stride : int
                stride of convolution
            weight_initializer : str
                kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
            activation_func : func
                activation function after convolution
            use_batch_norm : bool
                if set, then applied batch normalization
        '''

        blocks = []

        for n in range(n_block):

            if n == 0:
                stride = stride
            else:
                in_channels = out_channels
                stride = 1

            block = network_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks.append(block)

        blocks = torch.nn.Sequential(*blocks)

        return blocks

    def forward(self, x):
        '''
        Forward input x through the ResNet model
        Arg(s):
            x : torch.Tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))

        return layers[-1], layers[1:-1]


class FullyConnectedEncoder(torch.nn.Module):
    '''
    Fully connected encoder
    Arg(s):
        input_channels : int
            number of input channels
        n_neurons : list[int]
            number of filters to use per layer
        latent_size : int
            number of output neuron
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function after convolution
    '''

    def __init__(self,
                 input_channels=3,
                 n_neurons=[32, 64, 96, 128, 256],
                 latent_size=29 * 10,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(FullyConnectedEncoder, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        self.mlp = torch.nn.Sequential(
            net_utils.FullyConnected(
                in_features=input_channels,
                out_features=n_neurons[0],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[0],
                out_features=n_neurons[1],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[1],
                out_features=n_neurons[2],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[2],
                out_features=n_neurons[3],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[3],
                out_features=n_neurons[4],
                weight_initializer=weight_initializer,
                activation_func=activation_func),
            net_utils.FullyConnected(
                in_features=n_neurons[4],
                out_features=latent_size,
                weight_initializer=weight_initializer,
                activation_func=activation_func))

    def forward(self, x):
        return self.mlp(x)


class RCNetEncoder(torch.nn.Module):
    '''
    Radar association network
    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        n_filters_encoder_image : int
            number of filters for image (RGB) branch
        n_neurons_encoder_depth : int
            number of neurons for depth (radar) branch
        latent_size_depth : int
            size of latent vector
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''
    def __init__(self,
                 input_channels_image=3,
                 input_channels_depth=3,
                 input_patch_size_image=(900, 288),
                 n_filters_encoder_image=[32, 64, 128, 128, 128],
                 n_neurons_encoder_depth=[32, 64, 128, 128, 128],
                 latent_size_depth=128 * 29 * 10,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(RCNetEncoder, self).__init__()

        self.n_neuron_latent_depth = n_neurons_encoder_depth[-1]

        self.encoder_image = ResNetEncoder(
            n_layer=18,
            input_channels=input_channels_image,
            n_filters=n_filters_encoder_image,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.attention = LocalFeatureTransformer(['self','cross'], n_layers=4, d_model=self.n_neuron_latent_depth)

        self.encoder_depth = FullyConnectedEncoder(
            input_channels=input_channels_depth,
            n_neurons=n_neurons_encoder_depth,
            latent_size=latent_size_depth,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.input_patch_size_image =input_patch_size_image

    def forward(self, image, points, b_boxes):
        # Image shape: (B, C, H, W) # Should be (B, 3, 768, 288)
        # points shape: (B*K, X)
        # b_boxes: [(K, 4) * B], this should be a list with B elements, and each element is (K, 4) size
        # K is the number of radar points per image
        # X is the radar dimension


        # Define dimensions
        shape = self.input_patch_size_image
        latent_height = int(shape[-2] // 32.0)
        latent_width = int(shape[-1] // 32.0)
        batch_size = image.shape[0]

        # Define scales and feature sizes
        skip_scales = [ 1 /2.0, 1/ 4.0, 1 / 8.0, 1 / 16.0, 1 / 32.0, 1 / 64.0, 1 / 128.0]
        skip_feature_sizes = [
            (int(shape[-2] * skip_scale),
             int(shape[-1] * skip_scale))
            for skip_scale in skip_scales
        ]  # Should be [(384, 144), (192, 72), (96, 36), (48, 18)]

        latent_scale = 1 / 32.0
        latent_feature_size = (latent_height, latent_width)  # Should be (24, 9)

        # Forward the entire image
        latent_image, skips_image = self.encoder_image(image)

        # ROI pooling on latent images
        latent_image_pooled = torchvision.ops.roi_pool(
            latent_image, b_boxes,
            spatial_scale=latent_scale,
            output_size=latent_feature_size
        )  # (N*K, C, H, W)

        # ROI pooling on the skips
        skips_image_pooled = []
        for skip_image_idx in range(len(skips_image)):
            skips_image_pooled.append(
                torchvision.ops.roi_pool(
                    skips_image[skip_image_idx], b_boxes,
                    spatial_scale=skip_scales[skip_image_idx],
                    output_size=skip_feature_sizes[skip_image_idx]
                )  # (N*K, C, H, W)
            )

        # Radar points size: (bath_size * total_points_sampled, 3)
        # latent_depth size: (batch_size * total_points_sampled, n_neuron_latent_depth, patch_w//32, patch_h//32)
        # latent_image_pooled size = latent_depth size
        latent_depth = self.encoder_depth(points)
        latent_depth = latent_depth.view(points.shape[0], self.n_neuron_latent_depth, -1, latent_width)

        latent_depth_reshape = latent_depth.view(latent_depth.shape[0], latent_depth.shape[1], -1).permute(0, 2, 1)
        latent_image_pooled_reshape = latent_image_pooled.view(latent_image_pooled.shape[0],
                                                               latent_image_pooled.shape[1], -1).permute(0, 2, 1)
        latent_depth_tf, latent_image_pooled_tf = self.attention(latent_depth_reshape, latent_image_pooled_reshape)
        latent_depth_tf = latent_depth_tf.permute(0, 2, 1).view(latent_depth.shape)
        latent_image_pooled_tf = latent_image_pooled_tf.permute(0, 2, 1).view(latent_image_pooled.shape)

        # Concatenate the features
        # latent = torch.cat([latent_image_pooled, latent_depth], dim=1)
        latent = torch.cat([latent_image_pooled_tf, latent_depth_tf], dim=1)
        return latent, skips_image_pooled

'''
Decoder
'''


class MultiScaleDecoder(torch.nn.Module):
    '''
    Multi-scale decoder with skip connections
    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_resolution : int
            number of output resolutions (scales) for multi-scale prediction
        n_filters : int list
            number of filters to use at each decoder block
        n_skips : int list
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        output_func : func
            activation function for output
        use_batch_norm : bool
            if set, then applied batch normalization
        deconv_type : str
            deconvolution types available: transpose, up
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_resolution=1,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 deconv_type='up'):
        super(MultiScaleDecoder, self).__init__()

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert n_resolution > 0 and n_resolution < network_depth

        self.n_resolution = n_resolution
        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Upsampling from lower to full resolution requires multi-scale
        if 'upsample' in self.output_func and self.n_resolution < 2:
            self.n_resolution = 2

        filter_idx = 0

        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[filter_idx], n_filters[filter_idx]
        ]

        # Resolution 1/128 -> 1/64
        if network_depth > 6:
            self.deconv6 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv6 = None

        # Resolution 1/64 -> 1/32
        if network_depth > 5:
            self.deconv5 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv5 = None

        # Resolution 1/32 -> 1/16
        self.deconv4 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        self.deconv3 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_resolution > 3:
            self.output3 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/8 -> 1/4
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 3:
            skip_channels = skip_channels + output_channels

        self.deconv2 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_resolution > 2:
            self.output2 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/4 -> 1/2
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 2:
            skip_channels = skip_channels + output_channels

        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        if self.n_resolution > 1:
            self.output1 = net_utils.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=output_func,
                use_batch_norm=False)

        # Resolution 1/2 -> 1/1
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 1:
            skip_channels = skip_channels + output_channels

        self.deconv0 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        self.output0 = net_utils.Conv2d(
            out_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False)

    def forward(self, x, skips, shape=None):
        '''
        Forward latent vector x through decoder network
        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
            shape : tuple[int]
                (height, width) tuple denoting output size
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        '''

        layers = [x]
        outputs = []

        # Start at the end and walk backwards through skip connections
        n = len(skips) - 1

        # Resolution 1/128 -> 1/64
        if self.deconv6 is not None:
            layers.append(self.deconv6(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/64 -> 1/32
        if self.deconv5 is not None:
            layers.append(self.deconv5(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/32 -> 1/16
        layers.append(self.deconv4(layers[-1], skips[n]))

        # Resolution 1/16 -> 1/8
        n = n - 1

        layers.append(self.deconv3(layers[-1], skips[n]))

        if self.n_resolution > 3:
            output3 = self.output3(layers[-1])
            outputs.append(output3)

            upsample_output3 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/8 -> 1/4
        n = n - 1

        skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_resolution > 3 else skips[n]
        layers.append(self.deconv2(layers[-1], skip))

        if self.n_resolution > 2:
            output2 = self.output2(layers[-1])
            outputs.append(output2)

            upsample_output2 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/4 -> 1/2
        n = n - 1

        skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_resolution > 2 else skips[n]
        layers.append(self.deconv1(layers[-1], skip))

        if self.n_resolution > 1:
            output1 = self.output1(layers[-1])
            outputs.append(output1)

            upsample_output1 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/2 -> 1/1
        n = n - 1

        if 'upsample' in self.output_func:
            output0 = upsample_output1
        else:
            if self.n_resolution > 1:
                # If there is skip connection at layer 0
                skip = torch.cat([skips[n], upsample_output1], dim=1) if n == 0 else upsample_output1
                layers.append(self.deconv0(layers[-1], skip))
            else:

                if n == 0:
                    layers.append(self.deconv0(layers[-1], skips[n]))
                else:
                    layers.append(self.deconv0(layers[-1], shape=shape[-2:]))

            output0 = self.output0(layers[-1])

        outputs.append(output0)

        return outputs