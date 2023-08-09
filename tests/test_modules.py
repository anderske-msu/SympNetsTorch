import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import SympNetsTorch.SympNets as snn
from commonfuncs import timeit


class __test_network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 2
        n1 = n2 = 4

        self.lu = snn.Linear(dim=dim, up_or_low="up", n=n1, b=torch.ones(2 * dim))
        self.au = snn.Activation(torch.tanh, dim=dim, up_or_low="up")
        self.ll = snn.Linear(dim=dim, up_or_low="low", n=n2, b=torch.ones(2 * dim))
        self.al = snn.Activation(torch.tanh, dim=dim, up_or_low="low")

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        torch.autograd.set_detect_anomaly(True)
        pq = snn.x_to_pq(x)

        if inverse:
            pq = self.al(pq, inverse=True)
            pq = self.ll(pq, inverse=True)
            pq = self.au(pq, inverse=True)
            pq = self.lu(pq, inverse=True)

        else:
            pq = self.lu(pq)
            pq = self.au(pq)
            pq = self.ll(pq)
            pq = self.al(pq)

        nx = snn.x_to_pq(pq)

        return nx


@timeit
def test_modules(
    numdata: int = 1000, batch_size: int = 150, print_inverse_accuracy: bool = True
):
    print("Starting test...")
    test_size = 0.3
    success = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nn_model = __test_network().to(device)

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
            output = model(output, inverse=True)
            output = model(output)

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
        inverse_accuracy = []

        model.eval()

        test_loss = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                output = model(output, inverse=True)
                inverse_accuracy.append(torch.abs(output - data).mean().item())
                output = model(output)

                test_loss += F.mse_loss(output, target).item()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        if print_inverse_accuracy:
            print(f"Inverse accuracy: {sum(inverse_accuracy)/len(inverse_accuracy)}")

        return test_losses

    # Main
    numtest = int(test_size * numdata)
    numtrain = numdata - numtest

    train_x = torch.randn((numtrain, 4), dtype=torch.float64).to(device)
    train_y = torch.randn_like(train_x)

    test_x = torch.randn((numtest, 4), dtype=torch.float64).to(device)
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


if __name__ == "__main__":
    test_modules(print_inverse_accuracy=True)
