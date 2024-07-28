import torch

BASE_ACTIVATION = torch.nn.SiLU


class KANLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(KANLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self._init_weights()

    def _init_weights(self) -> None:
        self.base = torch.nn.Parameter(
            torch.Tensor((self.in_features, self.out_features))
        )
        torch.nn.init.xavier_normal_(self.base)
