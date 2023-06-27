from typing import Callable
import torch
import torch.nn as nn

# TODO Add support for changing device

# * Functions


def check_device(device: torch.device) -> torch.device:
    """Checks if device is a valid torch.device or is None. If None it will return a cpu device."""

    if device is None:
        return_device = torch.device("cpu")
    elif type(device) is torch.device:
        return_device = device
    else:
        msg = f"Invalid device, expected argument to be of type {torch.device}, got type {type(device)}."
        raise Exception(msg)

    return return_device


def check_up_or_low(up_or_low: str) -> str:
    """Checks if a valid string was given for up_or_low."""

    if not up_or_low in ["up", "low"]:
        msg = f'Expected up_or_low to be "up" or "low" got {up_or_low}.'

        raise Exception(msg)

    return up_or_low


# * Sub Modules


class activation_sub_up(nn.Module):
    def __init__(
        self, func: Callable, dim: int = 2, device: torch.device = None
    ) -> None:
        """Creates a upper trangular activation sympletic module.

        func is the activation function to be applied. Should apply a nonlinear activation function element by element.
        """

        super().__init__()
        self.a = nn.Parameter(torch.randn(dim))
        self.device = check_device(device)
        self.dim = dim
        self.func = func

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        pq_size = pq.size()
        npq = torch.empty_like(pq)

        if len(pq_size) == 2:
            # This means the batch size is greater than 1 and will loop over the batch to get the term.
            term_size = pq_size[0], pq_size[1] // 2

            term = torch.zeros(term_size, dtype=pq.dtype).to(self.device)

            for i in range(pq_size[0]):
                term[i] = torch.mv(
                    torch.diag(self.a), self.func(pq[i, self.dim :])
                )  # acting on q

        else:
            term = torch.mv(
                torch.diag(self.a), self.func(pq[self.dim :])
            )  # acting on q

        npq[..., : self.dim] = pq[..., : self.dim] + term  # new p
        npq[..., self.dim :] = pq[..., self.dim :]  # new q

        return npq


class activation_sub_low(nn.Module):
    def __init__(
        self, func: Callable, dim: int = 2, device: torch.device = None
    ) -> None:
        """Creates a lower trangular activation sympletic module.

        func is the activation function to be applied. Should apply a nonlinear activation function element by element.
        """

        super().__init__()
        self.a = nn.Parameter(torch.randn(dim))
        self.device = check_device(device)
        self.dim = dim
        self.func = func

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        pq_size = pq.size()
        npq = torch.empty_like(pq)

        if len(pq_size) == 2:
            # This means the batch size is greater than 1 and will loop over the batch to get the term.
            term_size = pq_size[0], pq_size[1] // 2

            term = torch.zeros(term_size, dtype=pq.dtype).to(self.device)

            for i in range(pq_size[0]):
                term[i] = torch.mv(
                    torch.diag(self.a), self.func(pq[i, : self.dim])
                )  # acting on p

        else:
            term = torch.mv(
                torch.diag(self.a), self.func(pq[: self.dim])
            )  # acting on p

        npq[..., : self.dim] = pq[..., : self.dim]  # new p
        npq[..., self.dim :] = pq[..., self.dim :] + term  # new q

        return npq


class linear_sub_low(nn.Module):
    def __init__(self, dim: int = 2, device: torch.device = None) -> None:
        """Creates a lower trangular linear sympletic module."""

        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.device = check_device(device)
        self.dim = dim

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        pq_size = pq.size()
        npq = torch.empty_like(pq)

        if len(pq_size) == 2:
            # This means the batch size is greater than 1 and will loop over the batch to get the term.
            term_size = pq_size[0], pq_size[1] // 2

            term = torch.zeros(term_size, dtype=pq.dtype).to(self.device)

            for i in range(pq_size[0]):
                term[i] = torch.mv(
                    self.A + self.A.T, pq[i, : self.dim]
                )  # Sp = (A +  A^T)p

        else:
            term = torch.mv(self.A + self.A.T, pq[: self.dim])  # Sp = (A +  A^T)p

        npq[..., : self.dim] = pq[..., : self.dim]  # new p
        npq[..., self.dim :] = pq[..., self.dim :] + term  # new q

        return npq


class linear_sub_up(nn.Module):
    def __init__(self, dim: int = 2, device: torch.device = None) -> None:
        """Creates an upper trangular linear sympletic module."""

        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.device = check_device(device)
        self.dim = dim

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        pq_size = pq.size()
        npq = torch.empty_like(pq)

        if len(pq_size) == 2:
            # This means the batch size is greater than 1 and will loop over the batch to get the term.
            term_size = pq_size[0], pq_size[1] // 2

            term = torch.zeros(term_size, dtype=pq.dtype).to(self.device)

            for i in range(pq_size[0]):
                term[i] = torch.mv(
                    self.A + self.A.T, pq[i, self.dim :]
                )  # Sq = (A +  A^T)q

        else:
            term = torch.mv(self.A + self.A.T, pq[self.dim :])  # Sq = (A +  A^T)q

        npq[..., : self.dim] = pq[..., : self.dim] + term  # new p
        npq[..., self.dim :] = pq[..., self.dim :]  # new q

        return npq


# * Full Modules


class Activation(nn.Module):
    def __init__(
        self,
        func: Callable,
        dim: int = 2,
        up_or_low: str = "up",
        device: torch.device = None,
    ) -> None:
        """Creates an activation sympmetic modules."""

        super().__init__()
        self.dim = dim
        self.device = check_device(device)

        if up_or_low == "up":
            self.layer = activation_sub_up(func, dim=dim, device=self.device)

        elif up_or_low == "low":
            self.layer = activation_sub_low(func, dim=dim, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pq = torch.empty_like(x)
        nx = torch.empty_like(x)

        pq[..., : self.dim] = x[..., 1::2].clone()
        pq[..., self.dim :] = x[..., 0::2].clone()

        pq = self.layer(pq)

        nx[..., : self.dim] = pq[..., 1::2].clone()
        nx[..., self.dim :] = pq[..., 0::2].clone()

        return nx


class Linear(nn.Module):
    def __init__(
        self,
        dim: int = 2,
        up_or_low: str = "up",
        n: int = 3,
        b: torch.Tensor = None,
        device: torch.device = None,
    ) -> None:
        """Creates an series of linear sympmetic modules."""

        super().__init__()
        self.up_or_low = check_up_or_low(up_or_low)
        self.device = check_device(device)
        self.dim = dim

        uplow = str(up_or_low)
        mlist = []

        for _ in range(n):
            if uplow == "up":
                mlist.append(linear_sub_up(dim=dim, device=self.device))
                uplow = "low"

            elif uplow == "low":
                mlist.append(linear_sub_low(dim=dim, device=self.device))
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
