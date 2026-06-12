import torch
import torch.nn as nn
from mdtorch.mdnn.activation import Softplus_jitter

class LinearCat(nn.Linear):
    """混合分布の割合を計算するための線形レイヤー

    (batch_size, in_features)の入力を受け取り、(batch_size, out_features = num_cat)の出力を返す。

    Attributes:
        temp (float): ソフトマックス関数の温度パラメータ。
    """
    def __init__(self, in_features:int, out_features:int, temp:float = 1., bias:bool = True, device:str = None, dtype:torch.dtype = None)->None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.temp = temp
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x:torch.Tensor):
        x = super().forward(x)
        x = self.softmax(x / self.temp)

        return x

class LinearM(nn.Linear):
    """パラメトリック混合分布の各分布のベクトルパラメータ(正規分布の場合期待値ベクトル)を出力する線形レイヤー

    (batch_size, in_features)の入力を受け取り、(batch_size, num_cat, out_features)の出力を返す。

    Attributes:
        num_cat (int): 混合分布のカテゴリ数。
        out_bias (torch.Tensor): 出力に加えるバイアス。形状は(num_cat, out_features)である必要がある。Noneの場合はバイアスを加えない。
    """
    def __init__(self, in_features:int, out_features:int, num_cat:int, bias:bool = True, out_bias:torch.Tensor = None, device:str = None, dtype:torch.dtype = None)->None:
        super().__init__(in_features, out_features*num_cat, bias, device, dtype)
        self.num_cat = num_cat
        self.out_dim = out_features
        self.out_bias = out_bias

    
    def forward(self, x:torch.Tensor):
        x = super().forward(x)
        x = x.view(-1, self.num_cat, self.out_dim)

        if self.out_bias is not None:
            x = x + self.out_bias
        return x


class LinearCov_from_tril(nn.Linear):
    """混合正規分布の分散共分散行列を予測するための線形レイヤー

    (batch_size, in_features)の入力を受け取り、(batch_size, num_cat, out_features, out_features)の出力を返す。
    分散共分散行列を直接求めるのではなく、コレスキー分解の下三角行列を計算する。

    Attributes:
        num_cat (int): 混合分布のカテゴリ数。
        out_dim (int): 出力特徴量の数。
        diag_func (nn.Module): 対角要素が正となるように適用する関数。デフォルトはSoftplus_jitter。
    """
    def __init__(self, in_features:int, out_features:int, num_cat:int, diag_func:nn.Module = Softplus_jitter(), bias:bool = True, device:str = None, dtype:torch.dtype = None)->None:
        super().__init__(in_features, out_features*num_cat*(out_features+1)//2, bias, device, dtype)
        self.out_dim = out_features
        self.num_cat = num_cat
        self.diag_func = diag_func

    def forward(self, x:torch.Tensor):
        batch_size = x.shape[0]
        x = super().forward(x)
        x = x.view(batch_size, self.num_cat, self.out_dim*(self.out_dim+1)//2)
        
        chol_factor = torch.zeros(batch_size, self.num_cat, self.out_dim, self.out_dim, device=x.device, dtype=x.dtype)
        tril_indices = torch.tril_indices(self.out_dim, self.out_dim)
        chol_factor[:, :, tril_indices[0], tril_indices[1]] = x
        diag_idx = torch.arange(self.out_dim, device=x.device)
        chol_factor[:, :, diag_idx, diag_idx] = (
            self.diag_func(chol_factor[:, :, diag_idx, diag_idx])
        )
        full_covariance = chol_factor @ chol_factor.transpose(-1, -2)

        return full_covariance