"""Step 3. The same places with new random values, and random places. The control of the paper (figure 4)."""
import numpy as np
from runs import run, mean, SEEDS, TOTAL, ROUNDS

print('STEP 3  Are the places enough?')
print('  ticket: the places of the search with their first values')
print('  new:    the same places with new random values')
print('  random: as many connections in every table at random places, first values')
print('   connections left      ticket    new     random     new start wins in')
for r in range(ROUNDS + 1):
    left = int(mean('left', r))
    t, n, x = (np.array([run(s)[f'{k}_end'][r][0] for s in SEEDS]) * 100 for k in ['ticket', 'new', 'random'])
    print(f'   {left:>6,}  ({left / TOTAL * 100:5.1f} %)    {t.mean():5.1f}    {n.mean():5.1f}    {x.mean():5.1f}      {int((n > t).sum())} of 10 runs')
print('  Steps until 95 of 100 pictures of the watch set are right (mean of the runs that get there):')
print('   connections left      ticket          new')
for r in [0, 3, 7, 10, 13, 16, 18]:
    out = []
    for k in ['ticket', 'new']:
        steps = [run(s)['step'][np.argmax(run(s)[f'{k}_watch'][r] >= 0.95)] for s in SEEDS if (run(s)[f'{k}_watch'][r] >= 0.95).any()]
        out.append(f'{np.mean(steps):5.0f} ({len(steps):>2} of 10)' if steps else '   no run      ')
    print(f'   {int(mean("left", r)):>6,}             {out[0]}   {out[1]}')
