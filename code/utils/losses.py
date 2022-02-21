import torch
import torch.nn
from torch.nn import functional as F
import numpy as np

"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NUM = [1113, 6705, 514, 327, 1099, 115, 142]
CLASS_WEIGHT = torch.Tensor([10000/(i) for i in CLASS_NUM]).cuda()

class Loss_Zeros(object):
    """
    map all uncertainty values to 0
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCELoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 0
        return self.base_loss(output, target)

class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output_softmax, target.long())

def get_UncertaintyLoss(method):
    assert method in METHODS
    
    if method == 'U-Zeros':
        return Loss_Zeros()

    if method == 'U-Ones':
        return Loss_Ones()

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2 * CLASS_WEIGHT
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def simclr_loss(output_fast, output_slow, Temperature, normalize=False):
    out = torch.cat((output_fast, output_slow), dim=0)
    sim_mat = torch.mm(out, torch.transpose(out, 0, 1))
    if normalize:
        sim_mat_denom = torch.mm(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t())
        sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
    sim_mat = torch.exp(sim_mat / Temperature)
    if normalize:
        sim_mat_denom = torch.norm(output_fast, dim=1) * torch.norm(output_slow, dim=1)
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / sim_mat_denom / Temperature)
    else:
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / Temperature)
    sim_match = torch.cat((sim_match, sim_match), dim=0)
    norm_sum = torch.exp(torch.ones(out.size(0)) / Temperature)
    norm_sum = norm_sum.cuda()
    loss = -torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum))
    # loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
    return loss

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.k = 1024

    def update_feature(self, features, weight, labels, epoch, step):
        if step == 0:
            self.update_f = features
            self.update_l = labels
            self.update_w = weight
        else:
            self.update_f = torch.cat([self.update_f, features], dim=0)
            self.update_l = torch.cat([self.update_l, labels], dim=0)
            self.update_w = torch.cat([self.update_w, weight], dim=0)
        if (self.update_f.shape[0] > self.k):
            f = self.update_f[-self.k:, :]
            l = self.update_l[-self.k:, :]
            w = self.update_w[-self.k:, :]

            self.update_f = self.update_f[-self.k:, :]
            self.update_l = self.update_l[-self.k:, :]
            self.update_w = self.update_w[-self.k:, :]
        else:
            f = self.update_f
            l = self.update_l
            w = self.update_w

        return f, l, w

    def forward(self, features, epoch, step, unl_weight, labels=None, mask=None):  #ema_weight,
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        label_bs = int(len(unl_weight) / 3)
        lab_weight = torch.unsqueeze(torch.ones(label_bs).float(), dim=1).to(device)  # 有标权重，加一维
        unl_weight = unl_weight.float().to(device)
        #ema_weight = unl_weight.float().to(device)
        weight_half = torch.cat((lab_weight, unl_weight))  # 有标无标权重拼接
        weight_list = weight_half.detach()
        #ema_weight_half = torch.cat((lab_weight, ema_weight))

        feature, ema_feature = torch.split(features, 1, dim=1)
        feature, ema_feature = torch.squeeze(feature, dim=1), torch.squeeze(ema_feature, dim=1)
        labels = torch.unsqueeze(labels, dim=1)
        feature_aug, ema_labels, weight_list = self.update_feature(ema_feature.detach(), weight_list, labels,
                                                                     epoch, step)
        #feature_aug, ema_labels, ema_weight_half = self.update_feature(ema_feature.detach(), ema_weight_half, labels, epoch, step)
        feature_contrast = torch.cat((feature, feature_aug), dim=0)
        #feature_contrast = torch.cat((feature, feature_aug), dim=0)

        labels = torch.cat((labels, ema_labels), dim=0)

        weight = torch.cat((weight_half, weight_list), dim=0)
        #weight_one = torch.ones_like(weight)
        weight_mat = torch.matmul(weight, weight.T)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # if labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = feature_contrast
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), #分母
            self.temperature)
        # for numerical stability  数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)  # 原始特征的标签取的是一组数据的，而计算需要两组的标签，这里需要重复，而在当前特征中标签已经合成双份的，故不需要
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(labels.shape[0]).view(-1, 1).to(device),
            # torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask #分子
        exp_logits_sum = exp_logits.sum(1, keepdim=True)
        for a in range(exp_logits_sum.shape[0]):
            if exp_logits_sum[a] == 0:
                exp_logits_sum[a] = exp_logits_sum[a] + 1.
        log_prob = weight_mat * (logits - torch.log(exp_logits_sum))#

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) +1e-8)  # nan值原因： mask.sum()为0

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()

        return loss


class SupConLoss_w(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.5):
        super(SupConLoss_w, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.k = 1024

    def update_feature(self, features, weight, labels, epoch, step):
        if step == 0:
            self.update_f = features
            self.update_l = labels
            self.update_w = weight
        else:
            self.update_f = torch.cat([self.update_f, features], dim=0)
            self.update_l = torch.cat([self.update_l, labels], dim=0)
            self.update_w = torch.cat([self.update_w, weight], dim=0)
        if (self.update_f.shape[0] > self.k):
            f = self.update_f[-self.k:, :]
            l = self.update_l[-self.k:, :]
            w = self.update_w[-self.k:, :]

            self.update_f = self.update_f[-self.k:, :]
            self.update_l = self.update_l[-self.k:, :]
            self.update_w = self.update_w[-self.k:, :]
        else:
            f = self.update_f
            l = self.update_l
            w = self.update_w

        return f, l, w

    def forward(self, features, epoch, step, unl_weight,  labels=None, mask=None):  #ema_weight,
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        label_bs = int(len(unl_weight)/3)
        lab_weight = torch.unsqueeze(torch.ones(label_bs).float(), dim=1).to(device)  # 有标权重，加一维
        unl_weight = unl_weight.float().to(device)
        #ema_weight = ema_weight.float().to(device)
        weight_half = torch.cat((lab_weight, unl_weight))  # 有标无标权 重拼接
        weight_list = weight_half.detach()

        #ema_weight_half = torch.cat((lab_weight, ema_weight))

        feature, ema_feature = torch.split(features, 1, dim=1)
        feature, ema_feature = torch.squeeze(feature, dim=1), torch.squeeze(ema_feature, dim=1)
        labels = torch.unsqueeze(labels, dim=1)
        feature_aug, ema_labels, weight_list = self.update_feature(ema_feature.detach(), weight_list, labels,
                                                                        epoch, step)
        #feature_aug, ema_labels, ema_weight_half = self.update_feature(ema_feature.detach(), ema_weight_half, labels,
         #                                                              epoch, step)
        feature_contrast = torch.cat((feature, feature_aug), dim=0)

        labels = torch.cat((labels, ema_labels), dim=0)

        weight = torch.cat((weight_half, weight_list), dim=0)
        #weight_one = torch.ones_like(weight)
        weight_mat = torch.matmul(weight, weight.T)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # if labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = feature_contrast
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability  数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)  # 原始特征的标签取的是一组数据的，而计算需要两组的标签，这里需要重复，而在当前特征中标签已经合成双份的，故不需要
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(labels.shape[0]).view(-1, 1).to(device),
            # torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits_sum = exp_logits.sum(1, keepdim=True)
        for a in range(exp_logits_sum.shape[0]):
            if exp_logits_sum[a] == 0:
                exp_logits_sum[a] = exp_logits_sum[a]+1.
        log_prob =weight_mat * (logits - torch.log(exp_logits_sum))#

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-8)  # nan值原因： mask.sum()为0

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()

        return loss


