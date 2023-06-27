import torch
import torch.nn as nn


class linear_sub(nn.Module):
    def __init__(
        self, dim: int = 2, up_or_low: str = "up", device: torch.device = None
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.eye = torch.eye(dim, dtype=torch.float32)
        self.dim = dim
        self.up_or_low = up_or_low

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        pq_size = pq.size()
        npq = torch.empty_like(pq)

        if len(pq_size) == 2:
            term_size = (pq_size[0], pq_size[1] // 2)

            pterm = torch.empty(term_size, dtype=pq.dtype).to(self.device)
            qterm = torch.empty_like(pterm).to(self.device)

            for i in range(pq_size[0]):
                if self.up_or_low == "up":
                    pterm[i] = torch.mv(self.A + self.A.T, pq[i, self.dim :])
                    qterm[i] = 0

                elif self.up_or_low == "low":
                    pterm[i] = 0
                    qterm[i] = torch.mv(self.A + self.A.T, pq[i, : self.dim])

        else:
            if self.up_or_low == "up":
                pterm = torch.mv(self.A + self.A.T, pq[self.dim :])
                qterm = 0

            elif self.up_or_low == "low":
                pterm = 0
                qterm = torch.mv(self.A + self.A.T, pq[: self.dim])

        npq[..., : self.dim] = pq[..., : self.dim] + pterm
        npq[..., self.dim :] = pq[..., self.dim :] + qterm

        return npq


class Linear(nn.Module):
    def __init__(
        self, dim: int = 2, up_or_low: str = "up", n: int = 3, b: torch.Tensor = None
    ) -> None:
        super().__init__()

        if not up_or_low in ["up", "low"]:
            raise Exception(
                f'Expected up_or_low to be "up" or "low" got "{up_or_low}".'
            )

        ud = up_or_low
        mlist = []
        self.dim = dim

        for _ in range(n):
            mlist.append(linear_sub(dim, ud))

            if ud == "up":
                ud = "low"

            elif ud == "low":
                ud = "up"

        self.layers = nn.ModuleList(mlist)

        if b is None:
            self.b = torch.zeros(2 * dim, dtype=torch.float32).to(device)

        else:
            self.b = b.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pq = torch.empty_like(x)
        nx = torch.empty_like(x)

        pq[..., : self.dim] = x[..., 1::2].clone()
        pq[..., self.dim :] = x[..., 0::2].clone()

        for layer in self.layers:
            pq = layer(pq)

        pq += self.b

        nx[..., : self.dim] = pq[..., 1::2].clone()
        nx[..., self.dim :] = pq[..., 0::2].clone()

        return nx
