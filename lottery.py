"""The lottery ticket experiment in one file.

A network is a list of weight tables. A mask is a list of tables of the same
shape that hold 1 (connection kept) or 0 (connection deleted). Everything in
the film is one of four moves on these two lists:

    start_values(seed)          random first values for every connection
    train(start, masks, ...)    learn the digits with the deleted ones held at 0
    prune(weights, masks, p)    delete the smallest p of the connections left
    reset                       train again from the very first values

Paper: Frankle and Carbin, "The Lottery Ticket Hypothesis: Finding Sparse,
Trainable Neural Networks", arXiv 1803.03635.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

torch.set_num_threads(1)

SIZES = [64, 300, 100, 10]          # 8x8 picture -> 300 -> 100 -> 10 digits
STEPS = 1500                        # training steps per run
BATCH = 60                          # pictures per step (as in the paper)
LR = 0.0012                         # Adam step size (as in the paper)
CHECK = 25                          # measure every 25 steps
SHARE = 0.20                        # delete 20 % of what is left per round
SHARE_LAST = 0.10                   # the last table loses 10 % per round


def load(split_seed=0):
    """1,000 pictures to learn from, 297 to watch the training, 500 for the test."""
    d = load_digits()
    x = torch.tensor(d.data / 16.0, dtype=torch.float32)
    y = torch.tensor(d.target, dtype=torch.long)
    order = np.random.RandomState(split_seed).permutation(len(x))
    a, b, c = order[:1000], order[1000:1297], order[1297:]
    return {"train": (x[a], y[a]), "watch": (x[b], y[b]), "test": (x[c], y[c])}


def start_values(seed, sizes=SIZES):
    """Random first values: a bell curve whose width depends on the table size."""
    g = torch.Generator().manual_seed(seed)
    tables = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        width = (2.0 / (n_in + n_out)) ** 0.5
        tables.append(torch.randn(n_out, n_in, generator=g) * width)
    return tables


def full_masks(sizes=SIZES):
    return [torch.ones(n_out, n_in) for n_in, n_out in zip(sizes[:-1], sizes[1:])]


def count(masks):
    return int(sum(m.sum().item() for m in masks))


def forward(tables, biases, x):
    h = x
    for i, (w, b) in enumerate(zip(tables, biases)):
        h = h @ w.t() + b
        if i < len(tables) - 1:
            h = torch.relu(h)
    return h


def score(tables, biases, xy):
    """(share of right answers, loss) on a set of pictures."""
    x, y = xy
    with torch.no_grad():
        out = forward(tables, biases, x)
        return (out.argmax(1) == y).float().mean().item(), F.cross_entropy(out, y).item()


def train(start, masks, data, seed, steps=STEPS, lr=LR, batch=BATCH, check=CHECK, biases=None):
    """Train from `start` with the masked connections held at zero.

    Returns the trained tables and what was measured every `check` steps.
    """
    tables = [(w * m).clone().requires_grad_(True) for w, m in zip(start, masks)]
    if biases is None:
        biases = [torch.zeros(w.shape[0]) for w in start]
    biases = [b.clone().requires_grad_(True) for b in biases]
    opt = torch.optim.Adam(tables + biases, lr=lr)
    x, y = data["train"]
    g = torch.Generator().manual_seed(10_000 + seed)
    log = {"step": [], "watch_loss": [], "watch": [], "test": []}

    def measure(step):
        wa, wl = score(tables, biases, data["watch"])
        ta, _ = score(tables, biases, data["test"])
        log["step"].append(step)
        log["watch_loss"].append(wl)
        log["watch"].append(wa)
        log["test"].append(ta)

    measure(0)
    order = torch.randperm(len(x), generator=g)
    at = 0
    for step in range(1, steps + 1):
        if at + batch > len(x):
            order = torch.randperm(len(x), generator=g)
            at = 0
        idx = order[at:at + batch]
        at += batch
        loss = F.cross_entropy(forward(tables, biases, x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for w, m in zip(tables, masks):
                w.mul_(m)
        if step % check == 0:
            measure(step)
    best = int(np.argmin(log["watch_loss"]))
    return {
        "tables": [w.detach().clone() for w in tables],
        "biases": [b.detach().clone() for b in biases],
        "log": log,
        "test_end": log["test"][-1],
        "stop_step": log["step"][best],          # step with the lowest loss on the watch set
        "test_at_stop": log["test"][best],
    }


def prune(tables, masks, share=SHARE, share_last=SHARE_LAST):
    """Delete the smallest connections that are left, table by table."""
    new = []
    for i, (w, m) in enumerate(zip(tables, masks)):
        p = share_last if i == len(tables) - 1 else share
        alive = m.bool()
        n_alive = int(alive.sum())
        n_cut = int(round(p * n_alive))
        if n_cut == 0:
            new.append(m.clone())
            continue
        sizes = w.abs()[alive]
        limit = torch.kthvalue(sizes, n_cut).values
        keep = alive & (w.abs() > limit)
        new.append(keep.float())
    return new


def random_masks(like, seed):
    """Masks with the same count per table, at random places."""
    g = torch.Generator().manual_seed(20_000 + seed)
    out = []
    for m in like:
        k = int(m.sum())
        flat = torch.zeros(m.numel())
        flat[torch.randperm(m.numel(), generator=g)[:k]] = 1
        out.append(flat.view_as(m))
    return out


def biggest(tables, like):
    """Masks that keep the biggest values of each table: as many as `like` keeps."""
    out = []
    for w, m in zip(tables, like):
        flat = torch.zeros(w.numel())
        flat[w.abs().flatten().topk(int(m.sum())).indices] = 1
        out.append(flat.view_as(w))
    return out


def live_units(masks):
    """For each middle layer: which units still have a way in and a way out."""
    first = (masks[0].sum(1) > 0) & (masks[1].sum(0) > 0)
    second = (masks[1][:, first].sum(1) > 0) & (masks[2].sum(0) > 0)
    return first, second


def dead_units(tables, masks, x, biases=None):
    """How many live units answer zero for every picture in x (first layer, second layer)."""
    b = biases or [torch.zeros(w.shape[0]) for w in tables]
    first, second = live_units(masks)
    with torch.no_grad():
        h1 = torch.relu(x @ (tables[0] * masks[0]).t() + b[0])
        h2 = torch.relu(h1 @ (tables[1] * masks[1]).t() + b[1])
    return int(((h1.sum(0) == 0) & first).sum()), int(((h2.sum(0) == 0) & second).sum())


def find_ticket(seed, data, rounds, share=SHARE, share_last=SHARE_LAST, steps=STEPS):
    """The search of the paper: train, delete the smallest, back to the first values, again.

    Returns the first values, the masks after every round and the trained run of every round.
    """
    first = start_values(seed)
    masks = full_masks()
    all_masks, runs = [], []
    for _ in range(rounds + 1):
        all_masks.append(masks)
        runs.append(train(first, masks, data, seed, steps=steps))
        masks = prune(runs[-1]["tables"], masks, share, share_last)
    return first, all_masks, runs
