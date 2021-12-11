import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
import tensorflow as tf
import tensorflow.keras.backend as K


def antifoldloss(y_pred):
    dy = y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :]-1
    dx = y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :]-1
    dz = y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:]-1

    dy = F.relu(dy) * torch.abs(dy*dy)
    dx = F.relu(dx) * torch.abs(dx*dx)
    dz = F.relu(dz) * torch.abs(dz*dz)
    return (torch.mean(dx)+torch.mean(dy)+torch.mean(dz))/3.0


def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x, y)
    dev_xx = torch.mul(x, x)
    dev_yy = torch.mul(y, y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    if (torch.isclose(torch.tensor([-1.0]).to("cuda"), ncc).any()):
        ncc = ncc + torch.tensor([0.01]).to("cuda")

    elif (torch.isclose(torch.tensor([1.0]).to("cuda"), ncc).any()):
        ncc = ncc - torch.tensor([0.01]).to("cuda")

    return ncc, ncc_map


class NormalizedCrossCorrelation(nn.Module):
    """ N-dimensional normalized cross correlation (NCC)
    Args:
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                 eps=1e-8,
                 return_map=False,
                 reduction='mean'):

        super(NormalizedCrossCorrelation, self).__init__()

        self._eps = eps
        self._return_map = return_map
        self._reduction = reduction

    def forward(self, x, y):
        return normalized_cross_correlation(x, y,self._return_map, self._reduction, self._eps)


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        super(Grad, self).__init__()

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class MutualInformation(nn.Module):

    def __init__(self, sigma=0.4, num_bins=256, normalize=True):
        super(MutualInformation, self).__init__()

        self.sigma = 2 * sigma ** 2
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10

        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device='cuda').float(), requires_grad=True)

    def marginalPdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def jointPdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def getMutualInformation(self, input1, input2):
        '''
            input1: B, C, H, W, D
            input2: B, C, H, W, D
            return: scalar
        '''

        # Torch tensors for images between (0, 1)
        input1 = input1 * 255
        input2 = input2 * 255

        B, C, H, W, D = input1.shape
        assert ((input1.shape == input2.shape))

        x1 = input1.view(B, H * W * D, C)
        x2 = input2.view(B, H * W * D, C)

        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        return mutual_information

    def forward(self, input1, input2):
        '''
            input1: B, C, H, W
            input2: B, C, H, W
            return: scalar
        '''
        return self.getMutualInformation(input1, input2)
        

class NMI_torch:

    def __init__(self, bin_centers, vol_size, sigma_ratio=0.5, max_clip=1, local=False, crop_background=False, patch_size=1):
        """
        Mutual information loss for image-image pairs.
        Author: Courtney Guo
        If you use this loss function, please cite the following:
        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis
        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        #print("vxm info: mutual information loss is experimental", file=sys.stderr)
        self.vol_size = vol_size
        self.max_clip = max_clip
        self.patch_size = patch_size
        self.crop_background = crop_background
        self.mi = self.local_mi if local else self.global_mi
        self.vol_bin_centers = torch.tensor(bin_centers)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = torch.tensor(  1 / (  2 * np.square(self.sigma) )   ).to("cuda")
        #self.o = [1, 1, np.prod([10])]
        #self.vbc = torch.reshape(self.vol_bin_centers, self.o).to("cuda")
        self.y_pred_shape1 = torch.tensor([1,2097152,1])
        # print(y_pred_shape1)
        self.nb_voxels = self.y_pred_shape1[1]

    def local_mi(self, y_true, y_pred):
        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, self.num_bins]
        vbc = torch.reshape(self.vol_bin_centers, o)

        # compute padding sizes
        patch_size = self.patch_size
        x, y, z = self.vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0,0]]
        pad_dims.append([x_r//2, x_r - x_r//2])
        pad_dims.append([y_r//2, y_r - y_r//2])
        pad_dims.append([z_r//2, z_r - z_r//2])
        pad_dims.append([0,0])
        padding = torch.tensor(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = torch.exp(- self.preterm * torch.square(torch.nn.functional.pad(y_true, padding, 'constant')  - vbc))
        I_a /= torch.sum(I_a, 0, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(torch.nn.functional.pad(y_pred, padding, 'constant')  - vbc))
        I_b /= torch.sum(I_b, 0, keepdim=True)

        I_a_patch = torch.reshape(I_a, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
        I_a_patch = torch.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = torch.reshape(I_a_patch, [-1, patch_size**3, self.num_bins])

        I_b_patch = torch.reshape(I_b, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
        I_b_patch = torch.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = torch.reshape(I_b_patch, [-1, patch_size**3, self.num_bins])

        # compute probabilities
        I_a_permute = torch.permute(I_a_patch, (0,2,1))
        pab = torch.bmm(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size**3
        pa = torch.mean(I_a_patch, 1, keepdims=True)
        pb = torch.mean(I_b_patch, 1, keepdims=True)

        papb = torch.bmm(torch.permute(pa, (0,2,1)), pb) + K.epsilon()
        return torch.mean(torch.sum(torch.sum(pab * torch.log(pab/papb + 1e-8), 1), 1))

    def global_mi(self, y_true, y_pred):
        if self.crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
            filt = torch.ones([ 1, 1, padding_size, padding_size, padding_size])

            smooth = torch.nn.Conv3d(y_true, filt, padding=[1, 1, 1, 1, 1])
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = torch.masked_select(y_pred, mask)
            y_true = torch.masked_select(y_true, mask)
            y_pred = torch.unsqueeze(torch.unsqueeze(y_pred, 0), 2)
            y_true = torch.unsqueeze(torch.unsqueeze(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
            y_true = torch.reshape(y_true, (-1, torch.prod(torch.tensor([*(y_true.shape)])[1:])))
            y_true = torch.unsqueeze(y_true, 2)
            y_pred = torch.reshape(y_pred, (-1, torch.prod(torch.tensor([*(y_pred.shape)])[1:])))
            y_pred = torch.unsqueeze(y_pred, 2)

        
        #nb_voxels = self.y_pred_shape1[1]
        nb_voxels = torch.tensor(y_pred.shape[1])

        # reshape bin centers to be (1, 1, B)
        vol_bin_centers = torch.tensor(self.vol_bin_centers)
        o = [1, 1, 10]
        vbc = torch.reshape(vol_bin_centers, o).to("cuda")      

        # compute image terms
        I_a = torch.exp(- self.preterm * torch.square(y_true  - vbc))
        I_a = I_a/torch.sum(I_a, -1, keepdims=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred  - vbc))
        I_b = I_b/torch.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = torch.permute(I_a, (0,2,1))
        pab = torch.bmm(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab = pab/nb_voxels
        pa = torch.mean(I_a, 1, keepdims=True)
        pb = torch.mean(I_b, 1, keepdims=True)

        papb = torch.bmm(torch.permute(pa, (0,2,1)), pb) + 1e-7
        return torch.sum(torch.sum(pab * torch.log(pab/papb + 1e-7), 1), 1)

    def loss(self, y_true, y_pred):
        y_pred = torch.clip(y_pred, 0, self.max_clip)
        y_true = torch.clip(y_true, 0, self.max_clip)
        return -self.mi(y_true, y_pred)


class NMI_keras:

    def __init__(self, bin_centers, vol_size, sigma_ratio=0.5, max_clip=1, local=False, crop_background=False, patch_size=1):
        """
        Mutual information loss for image-image pairs.
        Author: Courtney Guo

        If you use this loss function, please cite the following:

        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis

        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        #print("vxm info: mutual information loss is experimental", file=sys.stderr)
        self.vol_size = vol_size
        self.max_clip = max_clip
        self.patch_size = patch_size
        self.crop_background = crop_background
        self.mi = self.local_mi if local else self.global_mi
        self.vol_bin_centers = K.variable(bin_centers)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = K.variable(1 / (2 * np.square(self.sigma)))

    def local_mi(self, y_true, y_pred):
        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, self.num_bins]
        vbc = K.reshape(self.vol_bin_centers, o)

        # compute padding sizes
        patch_size = self.patch_size
        x, y, z = self.vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0,0]]
        pad_dims.append([x_r//2, x_r - x_r//2])
        pad_dims.append([y_r//2, y_r - y_r//2])
        pad_dims.append([z_r//2, z_r - z_r//2])
        pad_dims.append([0,0])
        padding = tf.constant(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = K.exp(- self.preterm * K.square(tf.pad(y_true, padding, 'CONSTANT')  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- self.preterm * K.square(tf.pad(y_pred, padding, 'CONSTANT')  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        I_a_patch = tf.reshape(I_a, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
        I_a_patch = tf.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = tf.reshape(I_a_patch, [-1, patch_size**3, self.num_bins])

        I_b_patch = tf.reshape(I_b, [(x+x_r)//patch_size, patch_size, (y+y_r)//patch_size, patch_size, (z+z_r)//patch_size, patch_size, self.num_bins])
        I_b_patch = tf.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = tf.reshape(I_b_patch, [-1, patch_size**3, self.num_bins])

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a_patch, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b_patch)  # should be the right size now, nb_labels x nb_bins
        pab /= patch_size**3
        pa = tf.reduce_mean(I_a_patch, 1, keepdims=True)
        pb = tf.reduce_mean(I_b_patch, 1, keepdims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        return K.mean(K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1))

    def global_mi(self, y_true, y_pred):
        if self.crop_background:
            # does not support variable batch size
            thresh = 0.0001
            padding_size = 20
            filt = tf.ones([padding_size, padding_size, padding_size, 1, 1])

            smooth = tf.nn.conv3d(y_true, filt, [1, 1, 1, 1, 1], "SAME")
            mask = smooth > thresh
            # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = K.expand_dims(K.expand_dims(y_pred, 0), 2)
            y_true = K.expand_dims(K.expand_dims(y_true, 0), 2)

        else:
            # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
            y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
            y_true = K.expand_dims(y_true, 2)
            y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
            y_pred = K.expand_dims(y_pred, 2)

        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(self.vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(self.vol_bin_centers, o)

        # compute image terms
        I_a = K.exp(- self.preterm * K.square(y_true  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- self.preterm * K.square(y_pred  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keepdims=True)
        pb = tf.reduce_mean(I_b, 1, keepdims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        return K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1)

    def loss(self, y_true, y_pred):
        y_pred = K.clip(y_pred, 0, self.max_clip)
        y_true = K.clip(y_true, 0, self.max_clip)
        return -self.mi(y_true, y_pred)

