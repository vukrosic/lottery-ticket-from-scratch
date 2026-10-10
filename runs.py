"""Every training run of the video, saved in files/. A run that is already there is read, not made again.

python3 runs.py          makes every run that is missing (5 at a time, about 3 minutes for each)

One file per seed. A seed fixes the first values of the big network and the order of the pictures.
In every file:
  rounds 0..22   the search (train, delete a fifth, back to the first values), and for the places of
                 every round: the ticket, the same places with new values, random places
  close looks    at rounds 7, 10, 13, 16, 18, 20 and 22: more time, bigger steps, signs, sizes,
                 all at once, picked before training, a narrow network with as many connections
"""
import sys
from pathlib import Path
import numpy as np
import torch
from lottery import (SIZES, STEPS, LR, load, start_values, full_masks, count, train, prune, random_masks,
                     biggest, live_units, dead_units, score, find_ticket)

FILES = Path(__file__).parent / 'files'
SEEDS = range(10)
ROUNDS = 22
CLOSE = [7, 10, 13, 16, 18, 20, 22]
LONG = 4 * STEPS
BIG_LR = 4 * LR
TOTAL = sum(a * b for a, b in zip(SIZES[:-1], SIZES[1:]))      # 50,200 connections


def name(seed):
    return FILES / f'seed-{seed}.npz'


def new_values(seed):
    """Other random first values for the same network: what "a new start" means in the video."""
    return start_values(1000 + seed)


def narrow(n):
    """A plain network with one middle layer and about n connections (74 for every unit)."""
    return [64, max(1, round(n / 74)), 10]


def make(seed):
    d = load()
    x = d['train'][0]
    first, masks, ticket = find_ticket(seed, d, ROUNDS)
    other = new_values(seed)
    o = {k: [] for k in ['left', 'per_table', 'pixels', 'units', 'plus', 'same_sign', 'cut_only', 'keep_on',
                         'dead_ticket', 'dead_new', 'dead_flip', 'dead_ticket_end', 'dead_new_end']}
    for who in ['ticket', 'new', 'random']:
        for k in ['test', 'watch', 'watch_loss', 'end']:
            o[f'{who}_{k}'] = []

    def put(who, r):
        o[f'{who}_test'].append(r['log']['test']); o[f'{who}_watch'].append(r['log']['watch'])
        o[f'{who}_watch_loss'].append(r['log']['watch_loss']); o[f'{who}_end'].append([r['test_end'], r['stop_step'], r['test_at_stop']])

    # the old way, for comparison: delete and keep on training from the trained values (no way back to the start)
    cur, m = ticket[0], masks[0]
    for r in range(ROUNDS + 1):
        if r:
            m = prune(cur['tables'], m)
            cur = train(cur['tables'], m, d, seed, biases=cur['biases'])
        o['keep_on'].append(cur['test_end'])

    for r, m in enumerate(masks):
        new = train(other, m, d, seed)
        put('ticket', ticket[r]); put('new', new); put('random', train(first, random_masks(m, seed), d, seed))
        o['left'].append(count(m)); o['per_table'].append([int(t.sum()) for t in m])
        o['pixels'].append(m[0].sum(0).numpy()); o['units'].append([int(u.sum()) for u in live_units(m)])
        o['plus'].append([float(((w > 0) & t.bool()).sum() / t.sum()) for w, t in zip(first, m)])
        o['same_sign'].append([float((((a > 0) == (b > 0)) & t.bool()).sum() / t.sum()) for a, b, t in zip(first, ticket[r]['tables'], m)])
        # delete at once from the trained big network, with no training after it
        cut = biggest(ticket[0]['tables'], m)
        o['cut_only'].append(score([w * t for w, t in zip(ticket[0]['tables'], cut)], ticket[0]['biases'], d['test'])[0])
        o['dead_ticket'].append(dead_units(first, m, x)); o['dead_new'].append(dead_units(other, m, x))
        o['dead_flip'].append(dead_units([-w for w in first], m, x))
        o['dead_ticket_end'].append(dead_units(ticket[r]['tables'], m, x, ticket[r]['biases']))
        o['dead_new_end'].append(dead_units(new['tables'], m, x, new['biases']))

    kinds = ['ticket_long', 'new_long', 'ticket_big', 'new_big', 'signs', 'sizes', 'sign_only', 'signs_table1', 'signs_table23',
             'flip', 'shuffle', 'at_once', 'at_once_new', 'before', 'narrow', 'narrow_long']
    for k in kinds:
        o['c_' + k] = []
    o['c_overlap'], o['c_narrow_size'], o['c_ticket_long_test'], o['c_new_long_test'] = [], [], [], []
    g = torch.Generator().manual_seed(30_000 + seed)
    end = lambda r: [r['test_end'], r['stop_step'], r['test_at_stop']]
    for r in CLOSE:
        m = masks[r]
        a = train(first, m, d, seed, steps=LONG); b = train(other, m, d, seed, steps=LONG)
        o['c_ticket_long'].append(end(a)); o['c_new_long'].append(end(b))
        o['c_ticket_long_test'].append(a['log']['test']); o['c_new_long_test'].append(b['log']['test'])
        o['c_ticket_big'].append(end(train(first, m, d, seed, lr=BIG_LR)))
        o['c_new_big'].append(end(train(other, m, d, seed, lr=BIG_LR)))
        # what is in the ticket: its signs or its sizes
        o['c_signs'].append(end(train([p.sign() * q.abs() for p, q in zip(first, other)], m, d, seed)))
        o['c_sizes'].append(end(train([p.abs() * q.sign() for p, q in zip(first, other)], m, d, seed)))
        o['c_sign_only'].append(end(train([p.sign() * p.std() for p in first], m, d, seed)))
        o['c_signs_table1'].append(end(train([first[0].sign() * other[0].abs(), other[1], other[2]], m, d, seed)))
        o['c_signs_table23'].append(end(train([other[0], first[1].sign() * other[1].abs(), first[2].sign() * other[2].abs()], m, d, seed)))
        o['c_flip'].append(end(train([-p for p in first], m, d, seed)))
        mixed = []
        for p, t in zip(first, m):
            q = p.clone(); vals = p[t.bool()]; q[t.bool()] = vals[torch.randperm(len(vals), generator=g)]; mixed.append(q)
        o['c_shuffle'].append(end(train(mixed, m, d, seed)))
        # other ways to pick the places
        once = biggest(ticket[0]['tables'], m)
        o['c_at_once'].append(end(train(first, once, d, seed))); o['c_at_once_new'].append(end(train(other, once, d, seed)))
        early = biggest(first, m)
        o['c_before'].append(end(train(first, early, d, seed)))
        o['c_overlap'].append(float(sum((p * q).sum() for p, q in zip(early, m)) / count(m)))
        # a plain narrow network with as many connections
        sizes = narrow(count(m)); ws, ms = start_values(seed, sizes), full_masks(sizes)
        o['c_narrow_size'].append([sizes[1], count(ms)])
        o['c_narrow'].append(end(train(ws, ms, d, seed))); o['c_narrow_long'].append(end(train(ws, ms, d, seed, steps=LONG)))

    arrays = {k: np.array(v, dtype=np.float32) for k, v in o.items()}
    arrays['masks'] = np.packbits(np.stack([torch.cat([t.flatten() for t in m]).numpy().astype(np.uint8) for m in masks]), axis=1)
    arrays['step'] = np.array(ticket[0]['log']['step'])
    FILES.mkdir(exist_ok=True)
    np.savez_compressed(name(seed), **arrays)


_seen = {}


def run(seed):
    if seed not in _seen:
        if not name(seed).exists():
            make(seed)
        _seen[seed] = dict(np.load(name(seed)))
    return _seen[seed]


def masks_of(seed, r):
    """The masks of round r as three tables of 0 and 1."""
    bits = np.unpackbits(run(seed)['masks'][r])[:TOTAL]
    out, at = [], 0
    for n_in, n_out in zip(SIZES[:-1], SIZES[1:]):
        out.append(torch.tensor(bits[at:at + n_in * n_out].reshape(n_out, n_in), dtype=torch.float32)); at += n_in * n_out
    return out


def mean(key, r=None, col=0):
    """Mean over the ten seeds. For a close look pass the round, for example mean('c_signs', 18)."""
    if key.startswith('c_'):
        v = np.stack([run(s)[key][CLOSE.index(r)] for s in SEEDS])
        return float(v[:, col].mean()) if v.ndim > 1 else float(v.mean())
    v = np.stack([run(s)[key] for s in SEEDS])
    return v.mean(0) if r is None else v.mean(0)[r]


if __name__ == '__main__':
    todo = [s for s in SEEDS if not name(s).exists()]
    if todo:
        import multiprocessing as mp
        with mp.Pool(5) as pool:
            for i, _ in enumerate(pool.imap_unordered(make, todo)):
                print(f'{i + 1} of {len(todo)} made', flush=True)
    print(f'{len(list(SEEDS))} runs in {FILES}')
