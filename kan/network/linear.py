import torch

BASE_ACTIVATION = torch.nn.functional.silu


class KANLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        spline_order: int,
    ) -> None:
        super(KANLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.wb = self._init_base_weight()
        self.ws = self._init_spline_weight()

    def _init_base_weight(self) -> torch.nn.Parameter:
        param = torch.nn.Parameter(
            torch.randn(size=(self.out_features, self.in_features))
        )
        torch.nn.init.xavier_uniform_(param)
        return param

    def _init_spline_weight(self) -> torch.nn.Parameter:
        param = torch.nn.Parameter(
            torch.ones(
                size=(
                    self.out_features,
                    self.in_features,
                    self.grid_size + self.spline_order,
                )
            )
        )
        return param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zb = torch.nn.functional.linear(BASE_ACTIVATION(x), self.wb)
        zs = torch.nn.functional.linear(x, self.ws)
        return zb + zs
