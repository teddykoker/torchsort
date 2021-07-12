from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import torchsort

NUM_CLASSES = 10


def topk_loss(input, target, regularization="kl", regularization_strength=1.0):

    # TODO: not sure if this is what they mean by logistic map
    # "On the other hand, for top-k classification, we find that applying a
    # logistic map to squash \theta to [0, 1] and tuning \epsilon is important "
    input = F.softmax(input, dim=-1)

    # computes ranks of logits
    ranks = torchsort.soft_rank(input, regularization, regularization_strength)

    # gather ranks at label
    ranks_label = ranks.gather(-1, target.view(-1, 1))

    # See https://github.com/teddykoker/torchsort/issues/19#issuecomment-831525303
    return F.relu(NUM_CLASSES - ranks_label).mean()


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

    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    test_transform = T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )

    train_ds = CIFAR10("./data", train=True, transform=train_transform, download=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    test_ds = CIFAR10("./data", train=False, transform=test_transform, download=True)
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

    loss_fn = (
        F.cross_entropy
        if args.loss_fn == "cross_entropy"
        else partial(
            topk_loss,
            regularization=args.regularization,
            regularization_strength=args.regularization_strength,
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    test_accs = []

    for epoch in range(args.epochs):
        train_loss = AverageMeter("train_loss")
        test_acc = AverageMeter("test_acc")

        # train step
        model.train()
        for (img, label) in train_dl:
            img, label = img.to(args.device), label.to(args.device)
            optimizer.zero_grad()

            pred = model(img)

            loss = loss_fn(pred, label)

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
        test_accs.append(test_acc.avg)

    def smooth(xs, factor=0.9):
        out = [xs[0]]
        for x in xs[1:]:
            out.append(out[-1] * factor + x * (1 - factor))
        return out

    test_accs = torch.stack(test_accs).cpu().numpy()
    regularization = (
        f"_{args.regularization}_{args.regularization_strength}"
        if args.loss_fn == "topk"
        else ""
    )
    np.save(f"{args.loss_fn}{regularization}_acc.npy", test_accs)


def plot():
    def smooth(xs, factor=0.9):
        out = [xs[0]]
        for x in xs[1:]:
            out.append(out[-1] * factor + x * (1 - factor))
        return out

    colors = ["tab:blue", "tab:orange"]
    plt.figure(figsize=(5, 3))
    for i, file in enumerate(Path("./").glob("*.npy")):
        print(file)
        test_accs = np.load(file)
        plt.plot(test_accs, alpha=0.1, color=colors[i])
        plt.plot(smooth(test_accs), color=colors[i], label=file.stem)

    plt.ylim(0.78, 0.88)
    plt.xlabel("Epochs")
    plt.ylabel("Test accuracy")
    plt.title("CIFAR-10")
    plt.legend()
    plt.savefig("extra/cifar10_test_accuracy.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--loss_fn", choices=["cross_entropy", "topk"], default="cross_entropy"
    )
    parser.add_argument("--regularization", default="kl")
    parser.add_argument("--regularization_strength", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.plot:
        plot()
    else:
        main(args)
