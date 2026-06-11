import torch
import torch.nn as nn

class MixGaussLoss(nn.Module):
    """混合ガウス分布の負の対数尤度を計算する損失関数。
    """
    def __init__(self, out_type:str = "mean")->None:
        """損失関数の初期化。

        Args:
            out_type (str, optional): 出力のタイプ。'mean'の場合は平均を返し、'sum'の場合は合計を返す。デフォルトは'mean'。
        """
        super().__init__()
        assert out_type in ["mean", "sum"], "out_type must be 'mean' or 'sum'"
        self.out_type = out_type

    def forward(self, pi:torch.Tensor, mu:torch.Tensor, cov:torch.Tensor, y:torch.Tensor, jitter:float = 1e-8)->torch.float:
        """混合ガウス分布の負の対数尤度を計算する。

        Args:
            pi (torch.Tensor): 混合係数。形状は(batch_size, num_cat)。sum(pi, dim=-1) == 1である必要がある。
            mu (torch.Tensor): 期待値ベクトル。形状は(batch_size, num_cat, out_features)。
            cov (torch.Tensor): 分散共分散行列。形状は(batch_size, num_cat, out_features, out_features)。正定値である必要がある。
            y (torch.Tensor): ターゲットデータ。形状は(batch_size, out_features)。
            jitter (float, optional): piに加える数値安定性のための小さな値。デフォルトは1e-8。

        Returns:
            torch.float: 負の対数尤度。
        """
        num_cat = mu.shape[1]
        y = y.unsqueeze(1).expand(-1, num_cat, -1)
        diff = y - mu
        cov_inv = torch.linalg.inv(cov)
        exponent = -0.5 * torch.einsum('bcd,bcde,bce->bc', diff, cov_inv, diff)
        log_det_cov = torch.logdet(cov)
        log_pi = torch.log(pi + jitter)  # 数値安定性のために小さな値を加える
        log_prob = log_pi - 0.5 * log_det_cov + exponent
        nll = -torch.logsumexp(log_prob, dim=1)
        
        if self.out_type == "mean":
            return nll.mean()
        else:
            return nll.sum()