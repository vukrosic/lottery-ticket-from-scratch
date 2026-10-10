"""The exercise. The search in the video takes 18 rounds to get from 50,200 connections to 1,037.
All at once (1 round) is far worse. How few rounds are enough?

    python3 exercise.py          3 rounds
    python3 exercise.py 6        6 rounds

Every round deletes the same share of what is left, so that the last round ends at 346, 541 and 150
connections in the three tables: the sizes of the ticket in the video. Guess first, then run it.
For comparison, the same five runs: 1 round 91.9 %, 18 rounds 96.3 %.
"""
import sys
import torch
from lottery import load, start_values, full_masks, train, count

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TARGET = [346, 541, 150]


def keep_biggest(tables, counts):
    out = []
    for w, n in zip(tables, counts):
        flat = torch.zeros(w.numel())
        flat[w.abs().flatten().topk(n).indices] = 1
        out.append(flat.view_as(w))
    return out


def search(seed, rounds, data):
    first = start_values(seed)
    masks = full_masks()
    for k in range(1, rounds + 1):
        trained = train(first, masks, data, seed)["tables"]       # deleted places hold 0, so they are never the biggest
        counts = [round(m.numel() * (t / m.numel()) ** (k / rounds)) for m, t in zip(masks, TARGET)]
        masks = keep_biggest(trained, counts)
    return train(first, masks, data, seed)["test_end"], count(masks)


if __name__ == "__main__":
    data = load()
    print(f"{ROUNDS} round{'s' if ROUNDS > 1 else ''} of training to find the ticket")
    got = []
    for seed in range(5):
        acc, left = search(seed, ROUNDS, data)
        got.append(acc)
        print(f"  run {seed}: {left:,} connections left, {acc * 100:.1f} % right on the test")
    print(f"  mean of five runs: {sum(got) / len(got) * 100:.1f} %")
