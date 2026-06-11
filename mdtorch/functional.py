import torch

def spoftplus_jitter(x:torch.Tensor, jitter:float=1e-4)->torch.Tensor:
    """jitter(マージン)付きのsoftplus活性化関数

    Args:
        x (torch.Tensor): 入力テンソル。
        jitter (float, optional): jitter値。デフォルトは1e-4。

    Returns:
        torch.Tensor: 出力テンソル。
    """
    return torch.log1p(torch.exp(x)) + jitter

def get_full_covariance_from_tril(x:torch.Tensor, out_features:int, num_cat:int, diag_func:"function" = spoftplus_jitter)->torch.Tensor:
    """下三角行列から分散共分散行列を出力する。

    Args:
        x (torch.Tensor): 下三角行列を表すテンソル。形状は(batch_size, num_cat * out_features * (out_features + 1) // 2)。
        out_features (int): 出力特徴量の数。
        num_cat (int): カテゴリ数。
        diag_func (function, optional): 対角要素が正となるように適用する関数。デフォルトはspoftplus_jitter。

    Returns:
        torch.Tensor: 分散共分散行列。形状は(batch_size, num_cat, out_features, out_features)。
    """
    batch_size = x.shape[0]
    x = x.view(batch_size, num_cat, out_features * (out_features + 1) // 2)
    full_covariance = torch.zeros(batch_size, num_cat, out_features, out_features, device=x.device)
    tril_indices = torch.tril_indices(out_features, out_features)
    full_covariance[:, :, tril_indices[0], tril_indices[1]] = x
    full_covariance = full_covariance + full_covariance.transpose(-1, -2)
    full_covariance[:, :, torch.arange(out_features), torch.arange(out_features)] /= 2
    full_covariance[:, :, torch.arange(out_features), torch.arange(out_features)] = diag_func(full_covariance[:, :, torch.arange(out_features), torch.arange(out_features)])

    return full_covariance