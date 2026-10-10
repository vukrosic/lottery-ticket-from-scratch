"""Checks of the claims of the video. python3 tests.py"""
import numpy as np
import torch
from lottery import (SIZES, load, start_values, full_masks, count, train, prune, random_masks, biggest,
                     live_units, dead_units, score, find_ticket)
from runs import run, mean, masks_of, new_values, narrow, SEEDS, ROUNDS, CLOSE, TOTAL

done = []


def check(text, ok):
    done.append(bool(ok))
    print(('ok    ' if ok else 'WRONG ') + text)


ten = lambda key, r, col=0: np.array([run(s)[key][r][col] if not key.startswith('c_') else np.atleast_1d(run(s)[key][CLOSE.index(r)])[col] for s in SEEDS]) * 100
d = load()

# the data and the network
x, y = d['train']
check('1,000 pictures to learn from, 297 to watch, 500 for the test', (len(x), len(d['watch'][0]), len(d['test'][0])) == (1000, 297, 500))
check('every square of a picture is between 0 and 1, never negative', x.min() == 0 and x.max() == 1)
check('the network has 19,200 + 30,000 + 1,000 = 50,200 connections', [m.numel() for m in full_masks()] == [19200, 30000, 1000] and TOTAL == 50200)
a, b = start_values(3), start_values(3)
check('the same seed gives the same first values', all(torch.equal(p, q) for p, q in zip(a, b)))
check('half of the first values are plus, half minus (49 to 51 %)', 0.49 < float((torch.cat([t.flatten() for t in a]) > 0).float().mean()) < 0.51)

# the moves
m = prune(a, full_masks())
check('one round deletes 20 % of tables 1 and 2 and 10 % of table 3', [int(t.sum()) for t in m] == [15360, 24000, 900])
check('the deleted connections are the smallest ones', all(w[t == 0].abs().max() <= w[t == 1].abs().min() for w, t in zip(a, m)))
m2 = prune(a, m)
check('a second round never brings a connection back', all(bool(((q == 1) & (p == 0)).sum() == 0) for p, q in zip(m, m2)))
r = random_masks(m, 0)
check('random places keep as many connections in every table', [int(t.sum()) for t in r] == [int(t.sum()) for t in m])
check('"the biggest" keeps as many as asked, and they are the biggest', [int(t.sum()) for t in biggest(a, m)] == [15360, 24000, 900] and all(torch.equal(p, q) for p, q in zip(biggest(a, m), m)))
small = train(a, m2, d, 0, steps=50)
check('a deleted connection stays at 0 through training', all(bool((w[t == 0] == 0).all()) for w, t in zip(small['tables'], m2)))
again = train(a, m2, d, 0, steps=50)
check('the same seed gives the same training run', all(torch.equal(p, q) for p, q in zip(small['tables'], again['tables'])))
check('training measures at step 0 and then every 25 steps', small['log']['step'] == [0, 25, 50])
one = [torch.zeros(300, 64), torch.zeros(100, 300), torch.ones(10, 100)]
one[0][0, 5] = 1; one[1][0, 0] = 1
lv = live_units(one)
check('a unit is live when it has a way in and a way out', int(lv[0].sum()) == 1 and int(lv[1].sum()) == 1)
w = [torch.full((300, 64), -1.0), torch.ones(100, 300), torch.ones(10, 100)]
check('a live unit with only minus connections answers 0 for every picture', dead_units(w, one, x) == (1, 1))
w[0][0, 5] = 1.0
check('the same unit with a plus connection is not dead', dead_units(w, one, x) == (0, 0))

# the saved runs agree with the code
first, masks, runs_ = find_ticket(0, d, 2)
check('the search made again gives the saved places (run 0, rounds 1 and 2)', all(torch.equal(p, q) for k in (1, 2) for p, q in zip(masks[k], masks_of(0, k))))
check('and the saved test scores', np.allclose([q['test_end'] for q in runs_], run(0)['ticket_end'][:3, 0]))
check('the saved places have the counts of the rounds', all(count(masks_of(0, k)) == int(run(0)['left'][k]) for k in range(ROUNDS + 1)))

# step 1 and 2: the big network and the search
big = ten('ticket_end', 0)
check(f'the big network: {big.mean():.1f} % right on the test, every run above 97', 98.0 < big.mean() < 98.4 and big.min() > 97)
check('deleted at once with no training after it: half 98.0 %, nine tenths 82.5 %, 1,037 left 19.4 %',
      all(abs(mean('cut_only', k) * 100 - v) < 0.1 for k, v in [(3, 98.0), (10, 82.5), (18, 19.4)]))
check('18 rounds leave 1,037 connections: 346, 541 and 150', int(mean('left', 18)) == 1037 and [int(v) for v in mean('per_table', 18)] == [346, 541, 150])
t18, t22 = ten('ticket_end', 18), ten('ticket_end', 22)
check(f'the ticket with 1,037 connections: {t18.mean():.1f} % (lowest run {t18.min():.1f})', abs(t18.mean() - 96.6) < 0.06 and t18.min() > 95)
check(f'the ticket with 462 connections: {t22.mean():.1f} %', int(mean('left', 22)) == 462 and abs(t22.mean() - 93.5) < 0.06)

# step 3: the same places, new values
n18, n22 = ten('new_end', 18), ten('new_end', 22)
check(f'a new start on the same places: {n18.mean():.1f} % and {n22.mean():.1f} %', abs(n18.mean() - 93.9) < 0.06 and abs(n22.mean() - 73.3) < 0.06)
check('the new start beats the ticket in 0 of 10 runs (1,037 and 462 left)', (n18 >= t18).sum() == 0 and (n22 >= t22).sum() == 0)
x18, x22 = ten('random_end', 18), ten('random_end', 22)
check(f'random places with the first values: {x18.mean():.1f} % and {x22.mean():.1f} %', abs(x18.mean() - 73.1) < 0.06 and abs(x22.mean() - 22.9) < 0.06)

# step 4: more time, bigger steps
check('four times the time: ticket 96.7, new start 95.8 (1,037); 94.4 and 77.2 (462)',
      all(abs(ten(k, r_).mean() - v) < 0.06 for k, r_, v in [('c_ticket_long', 18, 96.7), ('c_new_long', 18, 95.8), ('c_ticket_long', 22, 94.4), ('c_new_long', 22, 77.2)]))
check('four times bigger steps: ticket 96.7, new start 95.0 (1,037)', abs(ten('c_ticket_big', 18).mean() - 96.7) < 0.06 and abs(ten('c_new_big', 18).mean() - 95.0) < 0.06)
check('with more time the new start still loses in at least 8 of 10 runs', (ten('c_new_long', 18) < ten('c_ticket_long', 18)).sum() >= 8)

# step 5: signs
vals = {k: ten('c_' + k, 18).mean() for k in ['signs', 'sign_only', 'sizes', 'flip', 'shuffle']}
check(f'first signs with new sizes {vals["signs"]:.1f} %, new signs with first sizes {vals["sizes"]:.1f} %', abs(vals['signs'] - 96.2) < 0.06 and abs(vals['sizes'] - 94.4) < 0.06)
check(f'first signs with one size for all {vals["sign_only"]:.1f} %, every sign turned around {vals["flip"]:.1f} %', abs(vals['sign_only'] - 96.4) < 0.06 and abs(vals['flip'] - 35.0) < 0.06)
p = mean('plus', 18) * 100
check(f'plus signs among the first values the ticket keeps in table 1: {p[0]:.1f} % (all connections: {mean("plus", 0)[0] * 100:.1f} %)', abs(p[0] - 86.0) < 0.06 and abs(mean('plus', 0)[0] * 100 - 50.0) < 0.3)
check('99 of 100 kept connections have the same sign before and after training', mean('same_sign', 18).min() > 0.985)
u, dt, dn, df, dne = (mean(k, 18) for k in ['units', 'dead_ticket', 'dead_new', 'dead_flip', 'dead_new_end'])
check(f'layer 1 at 1,037 left: {u[0]:.0f} live units; dead before step 1: ticket {dt[0]:.0f}, new start {dn[0]:.0f}, turned {df[0]:.0f}',
      round(float(u[0])) == 194 and round(float(dt[0])) == 4 and round(float(dn[0])) == 66 and round(float(df[0])) == 156)
check(f'after training the new start has {dne[0]:.0f} dead units, the ticket {mean("dead_ticket_end", 18)[0]:.0f}', round(float(dne[0])) == 90 and round(float(mean('dead_ticket_end', 18)[0])) == 4)
ones, dead = 0, 0
for s in SEEDS:
    mk = masks_of(s, 18); live = live_units(mk)[0]; k = mk[0].sum(1)
    out = torch.relu(x @ (new_values(s)[0] * mk[0]).t())
    sel = live & (k == 1); ones += int(sel.sum()); dead += int(((out.sum(0) == 0) & sel).sum())
    if s == 0:
        check('run 0: the dead units counted from the saved places are the saved number', dead_units(new_values(0), mk, x) == tuple(int(v) for v in run(0)['dead_new'][18]))
check(f'{ones / 10:.0f} of the 194 units hang on one connection; a new start kills {dead / ones * 100:.0f} % of them', round(ones / 10) == 96 and 0.44 < dead / ones < 0.50)
s0 = np.array([run(s)['ticket_test'][18][0] for s in SEEDS]) * 100
n0 = np.array([run(s)['new_test'][18][0] for s in SEEDS]) * 100
check(f'before the first training step the ticket is right in {s0.mean():.1f} %, the new start in {n0.mean():.1f} %', abs(s0.mean() - 77.5) < 0.06 and n0.mean() < 13)

# step 6: shortcuts
check('all at once 90.6 %, picked before training 79.3 % (1,037 left)', abs(ten('c_at_once', 18).mean() - 90.6) < 0.06 and abs(ten('c_before', 18).mean() - 79.3) < 0.06)
check('the narrow network 64 -> 14 -> 10 has 1,036 connections', narrow(1037) == [64, 14, 10] and count(full_masks(narrow(1037))) == 1036)
check('it is right in 96.0 % after 1,500 steps and 96.9 % after 6,000', abs(ten('c_narrow', 18).mean() - 96.0) < 0.06 and abs(ten('c_narrow_long', 18).mean() - 96.9) < 0.06)
dead = sum(int((torch.relu(x @ start_values(s, narrow(1037))[0].t()).sum(0) == 0).sum()) for s in SEEDS)
check(f'the narrow network starts with {dead} dead units of 140 in ten runs', dead == 2)

print()
print(f'ALL TESTS PASS ({sum(done)} of {len(done)})' if all(done) else f'{len(done) - sum(done)} of {len(done)} TESTS WRONG')
