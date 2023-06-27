import torch
import torch.nn as nn


def check_device(device: torch.device) -> torch.device:
    if device is None:
        return_device = torch.device("cpu")
    elif type(device) is torch.device:
        return_device = device
    else:
        msg = f"Invalid device, expected argument to be of type {torch.device}, got type {type(device)}."
        raise Exception(msg)

    return return_device


def check_up_or_low(up_or_low: str) -> str:
    if not up_or_low in ["up", "low"]:
        msg = f'Expected up_or_low to be "up" or "low" got "{up_or_low}".'

        raise Exception(msg)

    return up_or_low


class linear_sub(nn.Module):
    def __init__(
        self, dim: int = 2, up_or_low: str = "up", device: torch.device = None
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.eye = torch.eye(dim, dtype=torch.float32)
        self.dim = dim
        self.up_or_low = check_up_or_low(up_or_low)
        self.device = check_device(device)

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        pq_size = pq.size()
        npq = torch.empty_like(pq)

        if len(pq_size) == 2:
            term_size = (pq_size[0], pq_size[1] // 2)

            pterm = torch.zeros(term_size, dtype=pq.dtype).to(self.device)
            qterm = torch.zeros_like(pterm).to(self.device)

            for i in range(pq_size[0]):
                if self.up_or_low == "up":
                    pterm[i] = torch.mv(self.A + self.A.T, pq[i, self.dim :])

                elif self.up_or_low == "low":
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
        self,
        dim: int = 2,
        up_or_low: str = "up",
        n: int = 3,
        b: torch.Tensor = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()

        self.up_or_low = check_up_or_low(up_or_low)
        self.device = check_device(device)
        self.dim = dim

        uplow = str(up_or_low)
        mlist = []

        for _ in range(n):
            mlist.append(linear_sub(dim=dim, up_or_low=uplow, device=device))

            if uplow == "up":
                uplow = "low"

            elif uplow == "low":
                uplow = "up"

        self.layers = nn.ModuleList(mlist)

        if b is None:
            self.b = torch.zeros(2 * dim, dtype=torch.float32).to(self.device)

        else:
            self.b = b.to(self.device)

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
