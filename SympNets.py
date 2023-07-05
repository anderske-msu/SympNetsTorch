from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
) -> torch.Tensor:
    """Creates the matrix to multiply by f(pq) to get the term to add to pq for the activation modules."""
    s1, e1 = index_1
    s2, e2 = index_2
    m = torch.zeros((2 * dim, 2 * dim), dtype=dtype, device=device)
    m[s1:e1, s2:e2] = torch.diag(a)

    return m


def linear_matrix(
    A: torch.Tensor,
    dim: int,
    index_1: tuple,
    index_2: tuple,
    dtype: type,
    device: torch.device,
) -> torch.Tensor:
    s1, e1 = index_1
    s2, e2 = index_2

    m = torch.eye(2 * dim, dtype=dtype, device=device)
    m[s1:e1, s2:e2] = A + A.T

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

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        matmul = activate_matrix(
            self.a,
            self.dim,
            index_1=(0, self.dim),
            index_2=(self.dim, None),
            dtype=pq.dtype,
            device=pq.device,
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

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        matmul = activate_matrix(
            self.a,
            self.dim,
            index_1=(self.dim, None),
            index_2=(0, self.dim),
            dtype=pq.dtype,
            device=pq.device,
        )
        pq += batch_mul_matrix_vector(matmul, self.func(pq))

        return pq


class linear_sub_low(nn.Module):
    def __init__(self, dim: int = 2) -> None:
        """Creates a lower trangular linear sympletic module."""

        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.dim = dim

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        matmul = linear_matrix(
            self.A,
            self.dim,
            index_1=(self.dim, None),
            index_2=(0, self.dim),
            dtype=pq.dtype,
            device=pq.device,
        )
        pq = batch_mul_matrix_vector(matmul, pq)

        return pq


class linear_sub_up(nn.Module):
    def __init__(self, dim: int = 2) -> None:
        """Creates an upper trangular linear sympletic module."""

        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.dim = dim

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        matmul = linear_matrix(
            self.A,
            self.dim,
            index_1=(0, self.dim),
            index_2=(self.dim, None),
            dtype=pq.dtype,
            device=pq.device,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pq = x_to_pq(x)

        pq = self.layer(pq)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pq = x_to_pq(x)

        for layer in self.layers:
            pq = layer(pq)

        pq += self.b

        nx = x_to_pq(pq)

        return nx


class test_network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 2
        n1 = 4
        n2 = n1

        self.lu = Linear(dim=dim, up_or_low="up", n=n1, b=torch.ones(2 * dim))
        self.au = Activation(torch.tanh, dim=dim, up_or_low="up")
        self.ll = Linear(dim=dim, up_or_low="low", n=n2, b=torch.ones(2 * dim))
        self.al = Activation(torch.tanh, dim=dim, up_or_low="low")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.autograd.set_detect_anomaly(True)
        pq = x_to_pq(x)
        pq = self.lu(pq)
        pq = self.au(pq)
        pq = self.ll(pq)
        pq = self.al(pq)
        nx = x_to_pq(pq)

        return nx


def test_modules(numdata: int = 1000, batch_size: int = 150):
    print("Starting test...")
    test_size = 0.3
    success = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nn_model = test_network().to(device)

    def train_model(model, epochs, train_loader):
        train_losses = []
        train_counter = []

        # set network to training mode
        model.train()

        # iterate through data batches
        for batch_idx, (data, target) in enumerate(train_loader):
            # reset gradients
            optimizer.zero_grad()

            # evaluate network with data
            output = model(data)

            # compute loss and derivative
            loss = F.mse_loss(output, target)
            loss.backward()

            # step optimizer
            optimizer.step()

            # print out results and save to file
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + ((epochs - 1) * len(train_loader.dataset))
            )

        return train_losses, train_counter

    def test_model(model, test_loader):
        test_losses = []

        model.eval()

        test_loss = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)

                test_loss += F.mse_loss(output, target).item()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        return test_losses

    # Main
    numtest = int(test_size * numdata)
    numtrain = numdata - numtest

    train_x = torch.randn((numtrain, 4)).to(device)
    train_y = torch.randn_like(train_x)

    test_x = torch.randn((numtest, 4)).to(device)
    test_y = torch.randn_like(test_x)

    # Network and Constraint Objects
    optimizer = optim.Adam(nn_model.parameters())

    train_data = torch.utils.data.TensorDataset(train_x, train_y)

    test_data = torch.utils.data.TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )

    train_losses = []
    test_losses = []
    train_count = []

    n_epochs = 5

    for epoch in range(1, n_epochs + 1):
        temp = train_model(nn_model, epoch, train_loader)

        train_losses += temp[0]
        train_count += temp[1]

        test_losses += test_model(nn_model, test_loader)

    success = True

    print("Test complete!")

    return success
