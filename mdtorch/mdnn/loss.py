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

        # Symmetrize numerically and robustly enforce PD for each covariance matrix.
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        batch_size, _, out_dim, _ = cov.shape
        cov_flat = cov.reshape(-1, out_dim, out_dim)
        eye = torch.eye(out_dim, device=cov.device, dtype=cov.dtype).unsqueeze(0).expand(cov_flat.shape[0], -1, -1)
        base_jitter = max(jitter, torch.finfo(cov.dtype).eps)

        chol, info = torch.linalg.cholesky_ex(cov_flat + base_jitter * eye)
        for scale in (10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0):
            failed = info > 0
            if not torch.any(failed):
                break
            cov_failed = cov_flat[failed] + (base_jitter * scale) * eye[failed]
            chol_failed, info_failed = torch.linalg.cholesky_ex(cov_failed)
            chol[failed] = chol_failed
            info[failed] = info_failed

        failed = info > 0
        if torch.any(failed):
            evals, evecs = torch.linalg.eigh(cov_flat[failed])
            evals = evals.clamp_min(base_jitter)
            cov_pd = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
            chol[failed] = torch.linalg.cholesky(cov_pd + base_jitter * eye[failed])

        chol = chol.view(batch_size, num_cat, out_dim, out_dim)

        solved = torch.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
        exponent = -0.5 * torch.einsum('bcd,bcd->bc', diff, solved)
        log_det_cov = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        log_pi = torch.log(pi.clamp_min(base_jitter))
        log_prob = log_pi - 0.5 * log_det_cov + exponent
        nll = -torch.logsumexp(log_prob, dim=1)
        
        if self.out_type == "mean":
            return nll.mean()
        else:
            return nll.sum()