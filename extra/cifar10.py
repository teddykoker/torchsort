from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import torchsort

NUM_CLASSES = 10


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.avg:.4f}"


def main(args):
    torch.manual_seed(0)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_ds = CIFAR10("./data", train=True, transform=transform, download=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    test_ds = CIFAR10("./data", train=False, transform=transform, download=True)
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # from the paper https://arxiv.org/abs/2002.08871:
    #
    # > Following Cuturi et al. (2019), we use a vanilla CNN (4 Conv2D with 2 maxpooling
    # > layers, ReLU activation, 2 fully connected layers with batch norm on each) ), the
    # > ADAM optimizer (Kingma & Ba, 2014) with a constant step size of 10−4, and set k = 1.
    #
    # there are no other details about the architecture in the paper. It reads they are
    # applying batch norm after the fully connected layers, but I think they meant on the
    # Conv2D.
    hidden = args.hidden_size

    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1),
        nn.BatchNorm2d(hidden),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * hidden, 512),
        nn.ReLU(),
        nn.Linear(512, NUM_CLASSES),
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(600):
        train_loss = AverageMeter("train_loss")
        test_acc = AverageMeter("test_acc")

        # train step
        model.train()
        for (img, label) in train_dl:
            img, label = img.to(args.device), label.to(args.device)
            optimizer.zero_grad()

            logit = model(img)

            # computes ranks of logits
            ranks = torchsort.soft_rank(
                logit, args.regularization, args.regularization_strength
            )

            # gather ranks at label
            ranks_label = ranks.gather(-1, label.view(-1, 1))

            # See https://github.com/teddykoker/torchsort/issues/19#issuecomment-831525303
            loss = F.relu(NUM_CLASSES - ranks_label).mean()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=img.shape[0])

        # test step
        model.eval()
        with torch.no_grad():
            for (img, label) in test_dl:
                img, label = img.to(args.device), label.to(args.device)
                logit = model(img)
                test_acc.update(
                    (logit.argmax(-1) == label).float().mean(), img.shape[0]
                )

        print(epoch, test_acc, train_loss)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--regularization", default="kl")
    parser.add_argument("--regularization_strength", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=16)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
