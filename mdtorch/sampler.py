import torch
import sys

def mix_gauss(pi:torch.Tensor, mu:torch.Tensor, cov:torch.Tensor, num_samples:int = 1)->torch.Tensor:
    """混合ガウス分布からサンプリングする関数。

    Args:
        pi (torch.Tensor): 説明変数の条件付き混合係数。形状は(batch_size, num_cat)。sum(pi, dim=-1) == 1である必要がある。
        mu (torch.Tensor): 説明変数の条件付き期待値ベクトル。形状は(batch_size, num_cat, out_features)。
        cov (torch.Tensor): 説明変数の条件付き分散共分散行列。形状は(batch_size, num_cat, out_features, out_features)。正定値である必要がある。
        num_samples (int, optional): サンプル数。デフォルトは1。
    
    Returns:
        torch.Tensor: サンプル。形状は(batch_size, num_samples, out_features)。
    """
    num_cat = mu.shape[1]
    pi = pi.unsqueeze(1).expand(-1, num_samples, -1)
    cat_dist = torch.distributions.Categorical(pi)
    cat_samples = cat_dist.sample()
    cat_samples_one_hot = torch.nn.functional.one_hot(cat_samples, num_classes=num_cat).float()
    selected_mu = torch.einsum('bsc,bco->bso', cat_samples_one_hot, mu)
    selected_cov = torch.einsum('bsc,bcod->bsod', cat_samples_one_hot, cov)
    normal_dist = torch.distributions.MultivariateNormal(selected_mu, covariance_matrix=selected_cov)
    samples = normal_dist.rsample()
    
    return samples