from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

# TODO Add support for changing device

# * Functions


def x_to_pq(x: torch.Tensor) -> torch.Tensor:
    """Converts X (x, px, y, py) to PQ (x, y, px, py) for sympletic layers.
    Putting in PQ instead of X will return X."""

    dim = x.size()[-1] // 2
    pq = torch.empty_like(x)

    pq[..., :dim] = x[..., 1::2].clone()  # p
    pq[..., dim:] = x[..., 0::2].clone()  # q

    return pq


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


def mul_m_pq(mat: torch.Tensor, pq: torch.Tensor) -> torch.Tensor:
    pq_size = pq.size()

    if len(pq_size) == 2:
        # This is a batch
        with torch.no_grad():
            mat = torch.stack(tuple(mat for _ in range(pq_size[0])))
        pq = torch.bmm(mat, pq.reshape(*pq_size, 1)).reshape(pq_size)

    else:
        pq = torch.mv(mat, pq)

    return pq


def activate_pq_term(
    func: Callable,
    a: torch.Tensor,
    pq: torch.Tensor,
    term_index: tuple,
    pq_index: tuple,
) -> torch.Tensor:
    pq_size = pq.size()
    tstart, tend = term_index
    pqstart, pqend = pq_index
    term = torch.zeros_like(pq)

    if len(pq_size) == 2:
        # This means the batch size is greater than 1 and will loop over the batch to get the term.

        for i in range(pq_size[0]):
            term[i, tstart:tend] = torch.mv(torch.diag(a), func(pq[i, pqstart:pqend]))

    else:
        term[tstart:tend] = torch.mv(torch.diag(a), func(pq[pqstart:pqend]))

    return term


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
        pq += activate_pq_term(
            self.func, self.a, pq, term_index=(0, self.dim), pq_index=(self.dim, None)
        )  # Acts on q, gives new p

        return pq


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
        pq += activate_pq_term(
            self.func, self.a, pq, term_index=(self.dim, None), pq_index=(0, self.dim)
        )  # Acts on p, gives new q

        return pq


class linear_sub_low(nn.Module):
    def __init__(self, dim: int = 2, device: torch.device = None) -> None:
        """Creates a lower trangular linear sympletic module."""

        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.device = check_device(device)
        self.dim = dim

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        matmul = linear_matrix(
            self.A,
            self.dim,
            index_1=(self.dim, None),
            index_2=(0, self.dim),
            dtype=pq.dtype,
            device=self.device,
        )
        pq = mul_m_pq(matmul, pq)

        return pq


class linear_sub_up(nn.Module):
    def __init__(self, dim: int = 2, device: torch.device = None) -> None:
        """Creates an upper trangular linear sympletic module."""

        super().__init__()
        self.A = nn.Parameter(torch.randn((dim, dim)))  # S = A + A^T
        self.device = check_device(device)
        self.dim = dim

    def forward(self, pq: torch.Tensor) -> torch.Tensor:
        matmul = linear_matrix(
            self.A,
            self.dim,
            index_1=(0, self.dim),
            index_2=(self.dim, None),
            dtype=pq.dtype,
            device=self.device,
        )
        pq = mul_m_pq(matmul, pq)

        return pq


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
        pq = x_to_pq(x)

        pq = self.layer(pq)

        nx = x_to_pq(pq)

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
            self.b = torch.zeros(2 * dim, dtype=torch.float32, device=self.device)

        else:
            self.b = b.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pq = x_to_pq(x)

        for layer in self.layers:
            pq = layer(pq)

        pq += self.b

        nx = x_to_pq(pq)

        return nx


class test_network(nn.Module):
    def __init__(self, device: torch.device = None) -> None:
        super().__init__()
        self.device = check_device(device)
        dim = 2

        self.layers = nn.ModuleList(
            [
                Linear(
                    dim=dim,
                    up_or_low="up",
                    n=4,
                    device=self.device,
                    b=torch.ones(2 * dim),
                ),
                Activation(torch.tanh, dim=dim, up_or_low="up", device=self.device),
                Linear(
                    dim=dim,
                    up_or_low="low",
                    n=4,
                    b=torch.ones(2 * dim),
                    device=self.device,
                ),
                Activation(torch.tanh, dim=dim, up_or_low="low", device=self.device),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.autograd.set_detect_anomaly(True)
        pq = x_to_pq(x)

        for layer in self.layers:
            pq = layer(pq)

        nx = x_to_pq(pq)

        return nx


def test_modules(
    numdata: int = 1000, batch_size: int = 150, device: torch.device = None
):
    print("Starting test...")
    success = False
    # Split info
    test_size = 0.3
    random_state = 3

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nn_model = test_network(device).to(device)

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

    train_x, test_x, train_y, test_y = train_test_split(
        torch.randn((numdata, 4)),
        torch.randn((numdata, 4)),
        test_size=test_size,
        random_state=random_state,
    )

    train_x = train_x.to(device)
    test_x = test_x.to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)

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

    print("Test Done")

    return success
