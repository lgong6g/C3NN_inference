import sys
import escnn.gspaces
import escnn.nn
import torch
import torch.nn as nn
import torch.nn.functional as F

bandlimit = 6
gspace = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=bandlimit)


""" Implements the non-linear convolution operation using the recursive formula."""
class NonlinearConv(nn.Module):
    def __init__(self, in_channels, num_filts, filter_size, order):
        super().__init__()
        self.order = order
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size

        """
        # Initialize filter similar as to how Pytorch initializes nn.Conv2d, but all positive
        # This is the old way of initializing C3NN filters without rotational equivariance/invariance
        weight_sample_k = 1.0 / np.sqrt(filter_size**2 * in_channels)
        init_filt = (2 * weight_sample_k * torch.rand(num_filts, in_channels, filter_size, filter_size, dtype=torch.float32))
        self.conv_filt = torch.nn.Parameter(init_filt)
        # Initialize some slices of the filter to be 0
        with torch.no_grad():
            self.conv_filt[:, :, int(self.filter_size/2)-2:int(self.filter_size/2)+3, int(self.filter_size/2)-2:int(self.filter_size/2)+3]=0.0
        """

        # Initialize the escnn filter
        feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.trivial_repr])
        feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.trivial_repr])
        # set initialization to be False in order to directly load weights from a pre-trained model
        self.conv = escnn.nn.R2Conv(in_type=feat_type_in, out_type=feat_type_out, bias=False, kernel_size=self.filter_size)

        # Precompute constant padding (avoids recomputing int() on every forward call)
        self.pad = (filter_size - 1) // 2
        # Precompute alternating signs (-1)^k for k=0,...,order-1
        self._signs = [1.0 if k % 2 == 0 else -1.0 for k in range(order)]

    def train(self, mode=True):
        super().train(mode)
        # escnn caches non-leaf tensors during eval() which breaks PyTorch deepcopy.
        # Since we extract parameters manually, force escnn to stay in train mode to prevent caching!
        if hasattr(self, 'conv'):
            self.conv.train(True)
        return self

    def forward(self, x):
        # Expand equivariant parameters to a dense filter; use a local variable
        # (not stored on self) so the tensor is released after the forward pass.
        abs_filt = self.conv.expand_parameters()[0].abs()

        # Build conv_powers incrementally: x_pow = x^k, w_pow = |w|^k.
        # Avoids recomputing each power from scratch (saves ~2 element-wise
        # multiplications over the independent x**k / abs_filt**k approach).
        x_pow, w_pow = x, abs_filt
        conv_powers = []
        for _ in range(self.order):
            conv_powers.append(F.conv2d(x_pow, w_pow, stride=1, padding=self.pad))
            x_pow, w_pow = x_pow * x, w_pow * abs_filt

        # C_0 is defined to be one everywhere
        corrs = [torch.ones_like(conv_powers[0])]

        # Compute each order of C_n recursively.
        # addcmul_ fuses sign * P_k * C_{n-k-1} into a single kernel,
        # eliminating the two intermediate tensors created by the original
        # (-1)**k * conv_powers[k] * corrs[n-k-1] expression.
        for n in range(1, self.order + 1):
            Cn = torch.zeros_like(conv_powers[0])
            for k in range(n):
                #Cn.addcmul_(conv_powers[k], corrs[n - k - 1], value=self._signs[k])
                Cn = torch.addcmul(Cn, conv_powers[k], corrs[n - k - 1], value=self._signs[k])
            Cn.div_(n)
            corrs.append(Cn)

        # Stack correlator maps along the channel dimension, dropping C_0
        return torch.cat(corrs[1:], dim=-3)
        #return torch.cat(corrs[2:], dim=-3) # cut away C_1 and start from C_2

class NonlinearConv_nopadding(nn.Module):
    def __init__(self, in_channels, num_filts, filter_size, order):
        super().__init__()
        self.order = order
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size

        """
        # Initialize filter similar as to how Pytorch initializes nn.Conv2d, but all positive
        # This is the old way of initializing C3NN filters without rotational equivariance/invariance
        weight_sample_k = 1.0 / np.sqrt(filter_size**2 * in_channels)
        init_filt = (2 * weight_sample_k * torch.rand(num_filts, in_channels, filter_size, filter_size, dtype=torch.float32))
        self.conv_filt = torch.nn.Parameter(init_filt)
        # Initialize some slices of the filter to be 0
        with torch.no_grad():
            self.conv_filt[:, :, int(self.filter_size/2)-2:int(self.filter_size/2)+3, int(self.filter_size/2)-2:int(self.filter_size/2)+3]=0.0
        """

        # Initialize the escnn filter
        feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.trivial_repr])
        feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.trivial_repr])
        self.conv = escnn.nn.R2Conv(in_type=feat_type_in, out_type=feat_type_out, bias=False, kernel_size=self.filter_size)

        self._signs = [1.0 if k % 2 == 0 else -1.0 for k in range(order)]

    def forward(self, x):
        abs_filt = self.conv.expand_parameters()[0].abs()

        x_pow, w_pow = x, abs_filt
        conv_powers = []
        for _ in range(self.order):
            conv_powers.append(F.conv2d(x_pow, w_pow, stride=1, padding=0))
            x_pow, w_pow = x_pow * x, w_pow * abs_filt

        corrs = [torch.ones_like(conv_powers[0])]

        for n in range(1, self.order + 1):
            Cn = torch.zeros_like(conv_powers[0])
            for k in range(n):
                #Cn.addcmul_(conv_powers[k], corrs[n - k - 1], value=self._signs[k])
                Cn = torch.addcmul(Cn, conv_powers[k], corrs[n - k - 1], value=self._signs[k])
            Cn.div_(n)
            corrs.append(Cn)

        return torch.cat(corrs[1:], dim=-3)

class NonlinearConv_periodic(nn.Module):
    def __init__(self, in_channels, num_filts, filter_size, order):
        super().__init__()
        self.order = order
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size

        """
        # Initialize filter similar as to how Pytorch initializes nn.Conv2d, but all positive
        # This is the old way of initializing C3NN filters without rotational equivariance/invariance
        weight_sample_k = 1.0 / np.sqrt(filter_size**2 * in_channels)
        init_filt = (2 * weight_sample_k * torch.rand(num_filts, in_channels, filter_size, filter_size, dtype=torch.float32))
        self.conv_filt = torch.nn.Parameter(init_filt)
        # Initialize some slices of the filter to be 0
        with torch.no_grad():
            self.conv_filt[:, :, int(self.filter_size/2)-2:int(self.filter_size/2)+3, int(self.filter_size/2)-2:int(self.filter_size/2)+3]=0.0
        """

        # Initialize the escnn filter
        feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.trivial_repr])
        feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.trivial_repr])
        self.conv = escnn.nn.R2Conv(in_type=feat_type_in, out_type=feat_type_out, bias=False, kernel_size=self.filter_size)

        self._pad_size = filter_size // 2
        self._signs   = [1.0 if k % 2 == 0 else -1.0 for k in range(order)]

    def forward(self, x):
        abs_filt = self.conv.expand_parameters()[0].abs()
        p = self._pad_size
        x = F.pad(x, pad=(p, p, p, p), mode="circular")

        x_pow, w_pow = x, abs_filt
        conv_powers = []
        for _ in range(self.order):
            conv_powers.append(F.conv2d(x_pow, w_pow, stride=1, padding=0))
            x_pow, w_pow = x_pow * x, w_pow * abs_filt

        corrs = [torch.ones_like(conv_powers[0])]

        for n in range(1, self.order + 1):
            Cn = torch.zeros_like(conv_powers[0])
            for k in range(n):
                #Cn.addcmul_(conv_powers[k], corrs[n - k - 1], value=self._signs[k])
                Cn = torch.addcmul(Cn, conv_powers[k], corrs[n - k - 1], value=self._signs[k])
            Cn.div_(n)
            corrs.append(Cn)

        return torch.cat(corrs[1:], dim=-3)


class NonlinearConv_initload(nn.Module):
    def __init__(self, in_channels, num_filts, filter_size, order, init_weight):
        super().__init__()
        self.order = order
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.init_weight = init_weight

        """
        # Initialize filter similar as to how Pytorch initializes nn.Conv2d, but all positive
        # This is the old way of initializing C3NN filters without rotational equivariance/invariance
        weight_sample_k = 1.0 / np.sqrt(filter_size**2 * in_channels)
        init_filt = (2 * weight_sample_k * torch.rand(num_filts, in_channels, filter_size, filter_size, dtype=torch.float32))
        self.conv_filt = torch.nn.Parameter(init_filt)
        # Initialize some slices of the filter to be 0
        with torch.no_grad():
            self.conv_filt[:, :, int(self.filter_size/2)-2:int(self.filter_size/2)+3, int(self.filter_size/2)-2:int(self.filter_size/2)+3]=0.0
        """
        
        # Initialize the escnn filter
        feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.trivial_repr])
        feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.trivial_repr])
        # set initialization to be False in order to directly load weights from a pre-trained model
        self.conv = escnn.nn.R2Conv(in_type=feat_type_in, out_type=feat_type_out, bias=False, kernel_size=self.filter_size, initialize=False) 

        self.pad     = (filter_size - 1) // 2
        self._signs  = [1.0 if k % 2 == 0 else -1.0 for k in range(order)]

        # Force the expanded filter to equal init_weight
        with torch.no_grad():
            expanded_filter, _ = self.conv.expand_parameters()
            # Ensure shapes match
            assert expanded_filter.shape == init_weight.shape, \
                   f"Shape mismatch: expanded {expanded_filter.shape}, init {init_weight.shape}"
            # Setup a hack: compute the required weights such that expanded filter == init_weight
            # Here we simply set the internal expanded filter attribute (not officially supported)
            self.conv.filter = nn.Parameter(init_weight.clone().detach())
            # Alternatively: assign self.conv.weights = ??? may need to match internal param
            # Mark as non‐trainable or trainable depending on your usage
            self.conv.filter.requires_grad = self.conv.weights.requires_grad
        
    def forward(self, x):
        # Use expand_parameters to get the filter we forced
        abs_filt = self.conv.expand_parameters()[0].abs()

        x_pow, w_pow = x, abs_filt
        conv_powers = []
        for _ in range(self.order):
            conv_powers.append(F.conv2d(x_pow, w_pow, stride=1, padding=self.pad))
            x_pow, w_pow = x_pow * x, w_pow * abs_filt

        corrs = [torch.ones_like(conv_powers[0])]

        for n in range(1, self.order + 1):
            Cn = torch.zeros_like(conv_powers[0])
            for k in range(n):
                #Cn.addcmul_(conv_powers[k], corrs[n - k - 1], value=self._signs[k])
                Cn = torch.addcmul(Cn, conv_powers[k], corrs[n - k - 1], value=self._signs[k])
            Cn.div_(n)
            corrs.append(Cn)

        return torch.cat(corrs[1:], dim=-3)



""" Does a simple average over the last two dimensions of the input."""
class Mean2d(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.mean((-2, -1))


""" The full CCNN is Input -> NonlinearConv -> Spatial Avg -> BatchNorm -> Logistic. The way it is constructed
    here assumes a binary classification task, but this can be extended to the multiclass case."""
class C3NN_w_std(nn.Module):
    """C3NN embedding that concatenates per-map mean and std summaries.

    Output dimension is ``2 * num_filts * order``.
    """

    def __init__(self, hparams, absbeta=False):
        """Build the C3NN summary network with optional absolute beta usage."""
        super().__init__()
        self.in_channels = hparams.in_channels
        self.num_filts = hparams.num_filts
        self.filter_size = hparams.filter_size
        self.order = hparams.order
        self.absbeta = absbeta

        # This flag is to avoid error reporting for the batch normalization at the beginning when sbi package only inputs one cosmology
        self.flag = False

        # the model architecture with escnn
        self.corr_conv = NonlinearConv(self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d()
        # I preserve the batch normalization layer here so that we can scale the compressed summary statistics to the same order of magnitude
        # This may help the numerical stability of inference in the 2nd part
        self.bn = nn.BatchNorm1d(2*hparams.num_filts * hparams.order, affine=False, eps=3.185289e-35, dtype=torch.float32) #3.185289e-35
        self.parallel = torch.vmap(self.forward_single, in_dims=0, out_dims=0)
    
    def forward_single(self, x):    
        """Encode a single map into moment summaries before realization pooling."""
        x = self.corr_conv(x)
        x = self.spatial_mean(x)
        return x
    
    def forward(self, x, chop=False):
        """Encode ``x`` and return concatenated mean/std summaries per cosmology."""
        x = self.parallel(x)
        mean = torch.mean(x, 1)
        std = torch.std(x, 1)
        x = torch.cat((mean, std), 1)
        if self.flag != False:
            if chop == False:
                x = self.bn(x)
        self.flag = True
        return x

""" Same as C3NN_w_std but with an additional learnable compression to hparams.out_dim"""
class C3NN_w_std_fixed_dim(nn.Module):
    """C3NN mean/std summary with an additional learnable linear compression."""

    def __init__(self, hparams, absbeta=False):
        """Build the compressed C3NN summary network."""
        super().__init__()
        self.in_channels = hparams.in_channels
        self.num_filts = hparams.num_filts
        self.filter_size = hparams.filter_size
        self.order = hparams.order
        self.absbeta = absbeta

        # This flag is to avoid error reporting for the batch normalization at the beginning when sbi package only inputs one cosmology
        self.flag = False

        # the model architecture with escnn
        self.corr_conv = NonlinearConv(self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d()
        # I preserve the batch normalization layer here so that we can scale the compressed summary statistics to the same order of magnitude
        # This may help the numerical stability of inference in the 2nd part
        self.bn = nn.BatchNorm1d(hparams.out_dim, affine=False, eps=3.185289e-35, dtype=torch.float32) #3.185289e-35
        #self.parallel = torch.vmap(self.forward_single, in_dims=0, out_dims=0)
        
        # additional layer of compression
        self.linear = nn.Linear(2*hparams.num_filts * hparams.order, hparams.out_dim)
    
    def forward_single(self, x):    
        """Encode a single map into moment summaries before realization pooling."""
        x = self.corr_conv(x)
        x = self.spatial_mean(x)
        return x
    
    def forward(self, x, chop=False):
        '''
        """Encode ``x`` and project the summary to ``hparams.out_dim``."""
        x = self.parallel(x)
        mean = torch.mean(x, 1)
        std = torch.std(x, 1)
        x = torch.cat((mean, std), 1)
        x = self.linear(x)
        if self.flag != False:
            if chop == False:
                x = self.bn(x)
        self.flag = True
        return x
        '''
        B, R, C, H, W = x.shape
        x_flat = x.reshape(B * R, C, H, W)
        x_encoded = self.forward_single(x_flat)
        x = x_encoded.reshape(B, R, -1)
        
        mean = torch.mean(x, 1)
        std = torch.std(x, 1)
        x = torch.cat((mean, std), 1)
        
        # CRITICAL: Restore the linear compression!
        x = self.linear(x)
        
        if self.flag != False:
            if chop == False:
                x = self.bn(x)
        self.flag = True
        return x


class C3NN_Nptonly(nn.Module):
    """C3NN variant that keeps only the highest-order moment block."""

    def __init__(self, hparams, absbeta=False):
        """Build a C3NN summary extractor restricted to selected N-point features."""
        super().__init__()
        self.in_channels = hparams.in_channels
        self.num_filts = hparams.num_filts
        self.filter_size = hparams.filter_size
        self.order = hparams.order
        self.absbeta = absbeta
        self.Npt_selection = self.num_filts*(self.order-1)

        # This flag is to avoid error reporting for the batch normalization at the beginning when sbi package only inputs one cosmology
        self.flag = False

        # the model architecture with escnn
        self.corr_conv = NonlinearConv(self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d()
        # I preserve the batch normalization layer here so that we can scale the compressed summary statistics to the same order of magnitude
        # This may help the numerical stability of inference in the 2nd part
        self.bn = nn.BatchNorm1d(2*hparams.num_filts * hparams.order, affine=False, eps=3.185289e-35, dtype=torch.float32) #3.185289e-35
        self.parallel = torch.vmap(self.forward_single, in_dims=0, out_dims=0)
    
    def forward_single(self, x):    
        """Encode a single map into moment summaries before realization pooling."""
        x = self.corr_conv(x)
        x = self.spatial_mean(x)
        return x
    
    def forward(self, x, chop=False):
        """Encode ``x`` and return only selected N-point mean/std features."""
        x = self.parallel(x)
        mean = torch.mean(x, 1)[:,self.Npt_selection:]
        std = torch.std(x, 1)[:,self.Npt_selection:]
        x = torch.cat((mean, std), 1)
        if self.flag != False:
            if chop == False:
                x = self.bn(x)
        self.flag = True
        return x



class C3NN_Nptonly_Nptbn(nn.Module):
    """Highest-order-only C3NN with preloaded filter initialization and BN."""

    def __init__(self, hparams, init_weight, absbeta=False):
        """Build the N-point-selected C3NN initialized from existing filters."""
        super().__init__()
        self.in_channels = hparams.in_channels
        self.num_filts = hparams.num_filts
        self.filter_size = hparams.filter_size
        self.order = hparams.order
        self.absbeta = absbeta
        self.Npt_selection = self.num_filts*(self.order-1)
        self.init_weight = init_weight

        # This flag is to avoid error reporting for the batch normalization at the beginning when sbi package only inputs one cosmology
        self.flag = False

        # the model architecture with escnn
        self.corr_conv = NonlinearConv_initload(self.in_channels, self.num_filts, self.filter_size, self.order, self.init_weight)
        self.spatial_mean = Mean2d()
        # I preserve the batch normalization layer here so that we can scale the compressed summary statistics to the same order of magnitude
        # This may help the numerical stability of inference in the 2nd part
        self.bn = nn.BatchNorm1d(2*self.num_filts, affine=False, eps=3.185289e-35, dtype=torch.float32) #3.185289e-35
        self.parallel = torch.vmap(self.forward_single, in_dims=0, out_dims=0)
    
    def forward_single(self, x):    
        """Encode a single map into moment summaries before realization pooling."""
        x = self.corr_conv(x)
        x = self.spatial_mean(x)
        return x
    
    def forward(self, x, chop=False):
        """Encode ``x`` and return highest-order selected mean/std summaries."""
        x = self.parallel(x)
        mean = torch.mean(x, 1)[:,self.Npt_selection:]
        std = torch.std(x, 1)[:,self.Npt_selection:]
        x = torch.cat((mean, std), 1)
        if self.flag != False:
            if chop == False:
                x = self.bn(x)
        self.flag = True
        return x



class CombinedC3NN(nn.Module):
    """Concatenate a frozen pretrained C3NN output with a trainable C3NN output."""

    def __init__(self, pretrained_c3nn, new_c3nn, total_dim):
        """Build a two-branch C3NN combiner with optional post-concatenation BN."""
        super().__init__()
        pretrained_c3nn.bn = nn.Identity()
        new_c3nn.bn = nn.Identity()
        self.pretrained_c3nn = pretrained_c3nn
        self.new_c3nn = new_c3nn
        self.bn = nn.BatchNorm1d(total_dim, affine=False, eps=3.185289e-35, dtype=torch.float32)

        self.flag = False

    def forward(self, x):
        """Return concatenated summaries from pretrained and new C3NN branches."""
        with torch.no_grad():   # freeze pretrained if desired
            out1 = self.pretrained_c3nn(x)
        out2 = self.new_c3nn(x)
        out = torch.cat([out1, out2], dim=-1)
        if self.flag != False:
            out = self.bn(out)
        self.flag = True
        return out



class CombinedC3NN_Nptbn(nn.Module):
    """Concatenate pretrained and new C3NN outputs without batch normalization."""

    def __init__(self, pretrained_c3nn, new_c3nn):
        """Build the minimal two-branch concatenation module."""
        super().__init__()
        self.pretrained_c3nn = pretrained_c3nn
        self.new_c3nn = new_c3nn

    def forward(self, x):
        """Return concatenated summaries from pretrained and new C3NN branches."""
        with torch.no_grad():   # freeze pretrained if desired
            out1 = self.pretrained_c3nn(x)
        out2 = self.new_c3nn(x)
        out = torch.cat([out1, out2], dim=-1)
        return out


        
class C3NN_nopadding(nn.Module):
    """C3NN summary network using valid convolutions (no spatial padding)."""

    def __init__(self, hparams, absbeta=False):
        """Build the no-padding C3NN embedding variant."""
        super().__init__()
        self.in_channels = hparams.in_channels
        self.num_filts = hparams.num_filts
        self.filter_size = hparams.filter_size
        self.order = hparams.order
        self.absbeta = absbeta

        # This flag is to avoid error reporting for the batch normalization at the beginning when sbi package only inputs one cosmology
        self.flag = False

        # the model architecture with escnn
        self.corr_conv = NonlinearConv_nopadding(self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d()
        # I preserve the batch normalization layer here so that we can scale the compressed summary statistics to the same order of magnitude
        # This may help the numerical stability of inference in the 2nd part
        self.bn = nn.BatchNorm1d(2*hparams.num_filts * hparams.order, affine=False, eps=3.185289e-35, dtype=torch.float32) #3.185289e-35
        self.parallel = torch.vmap(self.forward_single, in_dims=0, out_dims=0)
    
    def forward_single(self, x):    
        """Encode a single map into moment summaries before realization pooling."""
        x = self.corr_conv(x)
        x = self.spatial_mean(x)
        return x
    
    def forward(self, x, chop=False):
        """Encode ``x`` and return concatenated mean/std summaries."""
        x = self.parallel(x)
        mean = torch.mean(x, 1)
        std = torch.std(x, 1)
        x = torch.cat((mean, std), 1)
        if self.flag != False:
            if chop == False:
                x = self.bn(x)
        self.flag = True
        return x


class C3NN_hybrid(nn.Module):
    """Hybrid C3NN that augments C3NN summaries with compressed 2PCF features."""

    def __init__(self, hparams, absbeta=False):
        """Build the hybrid C3NN embedding for map+2PCF-channel inputs."""
        super().__init__()
        self.in_channels = hparams.in_channels
        self.num_filts = hparams.num_filts
        self.filter_size = hparams.filter_size
        self.order = hparams.order
        self.absbeta = absbeta

        # This flag is to avoid error reporting for the batch normalization at the beginning when sbi package only inputs one cosmology
        self.flag = False

        # the model architecture with escnn
        self.corr_conv = NonlinearConv(self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d()
        # I preserve the batch normalization layer here so that we can scale the compressed summary statistics to the same order of magnitude
        # This may help the numerical stability of inference in the 2nd part
        self.bn = nn.BatchNorm1d(hparams.num_filts * hparams.order, affine=False, eps=3.185289e-35, dtype=torch.float32) #3.185289e-35
        self.parallel = torch.vmap(self.forward_single, in_dims=0, out_dims=0)
    
    def forward_single(self, x):    
        """Encode a single map into moment summaries before realization pooling."""
        x = self.corr_conv(x)
        x = self.spatial_mean(x)
        return x
    
    def forward(self, x, chop=False):
        """Encode maps and concatenate with external compressed 2PCF statistics."""
        compressed_2pt = x[:,:,-1].mean(1)[:,0,:8]
        x = self.parallel(x[:,:,:-1])
        mean = torch.mean(x, 1)
        std = torch.std(x, 1)
        #x = torch.cat((mean, std), 1)
        x = mean
        if self.flag != False:
            if chop == False:
                x = self.bn(x)
        self.flag = True
        x = torch.cat((x, compressed_2pt), 1)
        return x
