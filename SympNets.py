from typing import Callable
import torch
import torch.nn as nn

# * Functions


def x_to_pq(x: torch.Tensor) -> torch.Tensor:
    """Converts X (x, px, y, py) to PQ (x, y, px, py) for sympletic layers.
    Putting in PQ instead of X will return X."""

    dim = x.size()[-1] // 2
    pq = torch.empty_like(x)

    pq[..., :dim] = x[..., 1::2].clone()  # p
    pq[..., dim:] = x[..., 0::2].clone()  # q

    return pq


def check_up_or_low(up_or_low: str) -> str:
    """Checks if a valid string was given for up_or_low."""

    if not up_or_low in ["up", "low"]:
        msg = f'Expected up_or_low to be "up" or "low" got {up_or_low}.'

        raise Exception(msg)

    return up_or_low


def batch_mul_matrix_vector(mat: torch.Tensor, pq: torch.Tensor) -> torch.Tensor:
    """Multiplies a given matrix by each of the batch of tensors given."""
    pq_size = pq.size()

    if len(pq_size) == 2:
        # This is a batch
        mat = torch.stack(tuple(mat for _ in range(pq_size[0])))
        pq = torch.bmm(mat, pq.reshape(*pq_size, 1)).reshape(pq_size)

    else:
        pq = torch.mv(mat, pq)

    return pq


def activate_matrix(
    a: torch.Tensor,
    dim: int,
    index_1: tuple,
    index_2: tuple,
    dtype: type,
    device: torch.device,
    inverse: bool,
) -> torch.Tensor:
    """Creates the matrix to multiply by f(pq) to get the term to add to pq for the activation modules."""
    s1, e1 = index_1
    s2, e2 = index_2
    m = torch.zeros((2 * dim, 2 * dim), dtype=dtype, device=device)

    if inverse:
        sign = -1

    else:
        sign = 1

    m[s1:e1, s2:e2] = sign * torch.diag(a)

    return m


def linear_matrix(
    A: torch.Tensor,
    dim: int,
    index_1: tuple,
    index_2: tuple,
    dtype: type,
    device: torch.device,
    inverse: bool,
) -> torch.Tensor:
    s1, e1 = index_1
    s2, e2 = index_2
    m = torch.eye(2 * dim, dtype=dtype, device=device)

    if inverse:
        sign = -1
    else:
        sign = 1

    m[s1:e1, s2:e2] = sign * (A + A.T)

    return m


# * Sub Modules


class activation_sub_up(nn.Module):
    def __init__(self, func: Callable, dim: int = 2) -> None:
        """Creates a upper trangular activation sympletic module.

        func is the activation function to be applied. Should apply a nonlinear activation function element by element.
        """

        super().__init__()
        self.a = nn.Parameter(torch.randn(dim))
        self.dim = dim
        self.func = func

    def forward(self, pq: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        matmul = activate_matrix(
            self.a,
            self.dim,
            index_1=(0, self.dim),
            index_2=(self.dim, None),
            dtype=pq.dtype,
            device=pq.device,
            inverse=inverse,
        )
        pq += batch_mul_matrix_vector(matmul, self.func(pq))  # Acts on q, gives new p

        return pq


class activation_sub_low(nn.Module):
    def __init__(self, func: Callable, dim: int = 2) -> None:
        """Creates a lower trangular activation sympletic module.

        func is the activation function to be applied. Should apply a nonlinear activation function element by element.
        """

        super().__init__()
        self.a = nn.Parameter(torch.randn(dim))
        self.dim = dim
        self.func = func

    def forward(self, pq: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        matmul = activate_matrix(
            self.a,
            self.dim,
            index_1=(self.dim, None),
            index_2=(0, self.dim),
            dtype=pq.dtype,
            device=pq.device,
            inverse=inverse,
        )
        pq += batch_mul_matrix_vector(matmul, self.func(pq))

        return pq


class linear_sub_low(nn.Module):
    def __init__(self, dim: int = 2) -> None:
        """Creates a lower trangular linear sympletic module."""

        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.dim = dim

    def forward(self, pq: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        matmul = linear_matrix(
            self.A,
            self.dim,
            index_1=(self.dim, None),
            index_2=(0, self.dim),
            dtype=pq.dtype,
            device=pq.device,
            inverse=inverse,
        )
        pq = batch_mul_matrix_vector(matmul, pq)

        return pq


class linear_sub_up(nn.Module):
    def __init__(self, dim: int = 2) -> None:
        """Creates an upper trangular linear sympletic module."""

        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.dim = dim

    def forward(self, pq: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        matmul = linear_matrix(
            self.A,
            self.dim,
            index_1=(0, self.dim),
            index_2=(self.dim, None),
            dtype=pq.dtype,
            device=pq.device,
            inverse=inverse,
        )
        pq = batch_mul_matrix_vector(matmul, pq)

        return pq


# * Full Modules


class Activation(nn.Module):
    def __init__(self, func: Callable, dim: int = 2, up_or_low: str = "up") -> None:
        """Creates an activation sympmetic modules."""

        super().__init__()

        if up_or_low == "up":
            self.layer = activation_sub_up(func, dim=dim)

        elif up_or_low == "low":
            self.layer = activation_sub_low(func, dim=dim)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        pq = x_to_pq(x)

        pq = self.layer(pq, inverse=inverse)

        nx = x_to_pq(pq)

        return nx


class Linear(nn.Module):
    def __init__(
        self, dim: int = 2, up_or_low: str = "up", n: int = 3, b: torch.Tensor = None
    ) -> None:
        """Creates an series of linear sympmetic modules."""

        super().__init__()

        uplow = str(check_up_or_low(up_or_low))
        mlist = []

        for _ in range(n):
            if uplow == "up":
                mlist.append(linear_sub_up(dim=dim))
                uplow = "low"

            elif uplow == "low":
                mlist.append(linear_sub_low(dim=dim))
                uplow = "up"

        self.layers = nn.ModuleList(mlist)

        if b is None:
            self.b = torch.zeros(2 * dim, dtype=torch.float32)

        else:
            self.b = b

    def _apply(self, fn):
        self.b = fn(self.b)

        return super()._apply(fn)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        pq = x_to_pq(x)

        if inverse:
            pq -= self.b

            for layer in reversed(self.layers):
                pq = layer(pq, inverse=True)

        else:
            for layer in self.layers:
                pq = layer(pq)

            pq += self.b

        nx = x_to_pq(pq)

        return nx
