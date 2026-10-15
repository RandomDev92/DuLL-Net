import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        self._init_weights()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y

    def _init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out

    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)

        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)

        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)

        output = self.combine_heads(attention)

        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)

class JointHVDenoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3):
        super(JointHVDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act4 = nn.PReLU()
        self._init_weights()

    def forward(self, input, in2):
        x = torch.cat([input, in2], dim=1)
        x1 = self.act1(self.conv1(x))
        x2 = self.act2(self.conv2(x1))
        x3 = self.act3(self.conv3(x2))
        x4 = self.act4(self.conv4(x3))
        x = self.bottleneck(x4)
        x = self.up4(x)
        x = self.up3(x + x3)
        x = self.up2(x + x2)
        x = x + x1
        x = torch.tanh(self.res_layer(x))
        return torch.tanh(self.output_layer(x + input))

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class DuLL(nn.Module):
    def __init__(self, filters=32):
        super(DuLL, self).__init__()
        #hvi
        self.process_i = self._create_processing_layers(filters)
        self.process_h = self._create_processing_layers_hv(filters)
        self.process_v = self._create_processing_layers_hv(filters)
        self.hvi_t = self.HVIT
        self.hvi_i = self.PHVIT
        self.denoiser_h = JointHVDenoiser(filters//2)
        self.denoiser_v = JointHVDenoiser(filters//2)

        self.i_pool = nn.MaxPool2d(8)
        self.i_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.i_up = nn.Upsample(scale_factor=8, mode='bicubic', align_corners=False)
        self.i_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0)
        self.hv_conv = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)
        self.msef_hvi = MSEFBlock(filters)
        self.recombine_hvi = nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1)
        self.recombine_act = nn.PReLU()
        self.final_adjustments_hvi = nn.Conv2d(filters, 3, kernel_size=3, padding=1)

        # #ycb
        self.process_y = self._create_processing_layers(filters)
        self.lum_pool = nn.MaxPool2d(8)
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.lum_up = nn.Upsample(scale_factor=8, mode='nearest')

        self.combine_light = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2= False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0
        self._init_weights()
        pass

    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * np.pi).sin() + eps).pow(k)
        ch = (2.0 * np.pi * hue).cos()
        cv = (2.0 * np.pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I],dim=1)
        return xyz

    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]

        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * np.pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V + eps,H + eps) / (2*np.pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

    def _create_processing_layers(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _create_processing_layers_hv(self, filters):
        return nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.PReLU()
        )


    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5

        yuv = torch.stack((y, u, v), dim=1)
        return yuv

    def _ycbcr_to_rgb(self, yuv):
        # Split the channels (B, 3, H, W)
        y, u, v = yuv[:, 0, :, :], yuv[:, 1, :, :], yuv[:, 2, :, :]

        # Subtract the 0.5 offset applied during the forward transform
        u_scaled = u - 0.5
        v_scaled = v - 0.5

        # Inverse matrix multiplication derived from your constants
        r = y + 1.13983 * v_scaled
        g = y - 0.39465 * u_scaled - 0.58060 * v_scaled
        b = y + 2.03211 * u_scaled

        # Restack back into (B, 3, H, W)
        rgb = torch.stack((r, g, b), dim=1)
        return rgb

    def forward(self, inputs):
        ycbcr = self._rgb_to_ycbcr(inputs)

        y, cb, cr = torch.split(ycbcr, 1, dim=1)

        y_processed = self.process_y(y)

        lum = y_processed
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum = lum + lum_1

        hvi = self.hvi_t(inputs)

        h, v, i = torch.split(hvi, 1, dim=1)
        # hv = torch.cat([h, v], dim=1)
        # hv = self.denoiser_hv(hv)
        # h_denoised, v_denoised = torch.split(hv, 1, dim=1)
        # h = h + h_denoised
        # v = v + v_denoised
        h = h + self.denoiser_h(h, torch.cat([cb, cr], dim=1))
        v = v + self.denoiser_v(v, torch.cat([cb, cr], dim=1))

        i_processed = self.process_i(i)
        h_processed = self.process_h(h)
        v_processed = self.process_v(v)

        hv = torch.cat([h_processed, v_processed], dim=1)
        i = i_processed
        i_1 = self.i_pool(i)
        i_1 = self.i_mhsa(i_1)
        i_1 = self.i_up(i_1)
        i = i + i_1

        y_i = i + lum
        i = i + self.combine_light(y_i)

        hv = self.hv_conv(hv)
        shortcut = hv
        hv = hv + 0.2 * self.i_conv(i)
        hv = self.msef_hvi(hv)
        hv = hv + shortcut

        recombined = self.recombine_act(self.recombine_hvi(torch.cat([hv, i], dim=1)))
        output1 = self.final_adjustments_hvi(recombined)
        h, v, i = torch.split(output1, 1, dim=1)
        bounded_h = torch.tanh(h)    # Tanh bounds H to [-1, 1]
        bounded_v = torch.tanh(v)    # Tanh bounds V to [-1, 1]
        bounded_i = torch.sigmoid(i) # Sigmoid bounds Intensity to [0, 1]
        output1_hvi = torch.cat([bounded_h, bounded_v, bounded_i], dim=1)
        output1 = self.hvi_i(output1_hvi)
        #output1 = self._rgb_to_ycbcr(output1)
        # output = torch.clamp(output, 0.0, 1.0)
        return output1

    def _init_weights(self):
        # Loop over immediate children only if you want to avoid overwriting grandchild blocks
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.kaiming_uniform_(module.weight, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)