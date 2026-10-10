"""Step 5. What is in the first values: the signs or the sizes. And what a wrong sign does to a unit."""
import numpy as np
import torch
from runs import run, mean, masks_of, new_values, SEEDS, CLOSE, TOTAL
from lottery import load, live_units

print('STEP 5  Signs and sizes (after Zhou, Lan, Liu, Yosinski 2019)')
print('  All on the places of the ticket. A value has a sign (plus or minus) and a size.')
rows = [('ticket', 'first signs, first sizes'), ('c_signs', 'first signs, new sizes'), ('c_sign_only', 'first signs, one size for all'),
        ('c_shuffle', 'first values, mixed among the places'), ('c_signs_table23', 'first signs in tables 2 and 3 only'),
        ('c_signs_table1', 'first signs in table 1 only'), ('c_sizes', 'new signs, first sizes'), ('new', 'new signs, new sizes'),
        ('c_flip', 'every first sign turned around')]
print(f'   {"connections left":<38}' + ''.join(f'{int(mean("left", r)):>7,}' for r in CLOSE))
for k, text in rows:
    vals = [mean(k, r) * 100 if k.startswith('c_') else mean(k + '_end', r)[0] * 100 for r in CLOSE]
    print(f'   {text:<38}' + ''.join(f'{v:7.1f}' for v in vals))
print()
print('  Share of plus signs among the first values the ticket keeps (table 1, 2, 3):')
for r in [0, 7, 13, 18, 22]:
    p = mean('plus', r) * 100
    print(f'   {int(mean("left", r)):>6,} left:   {p[0]:4.1f} %  {p[1]:4.1f} %  {p[2]:4.1f} %')
s = mean('same_sign', 18) * 100
print(f'  Connections of the ticket with the same sign before and after training (1,037 left): {s[0]:.1f} %, {s[1]:.1f} %, {s[2]:.1f} %')
print()
print('  Dead units: units with a way in and a way out that answer 0 for all 1,000 pictures.')
print('   left      live units     dead before step 1: ticket  new  turned     dead after training: ticket  new')
for r in [7, 13, 18, 22]:
    u = mean('units', r); a, b, c, d, e = (mean(k, r) for k in ['dead_ticket', 'dead_new', 'dead_flip', 'dead_ticket_end', 'dead_new_end'])
    for layer in (0, 1):
        print(f'   {int(mean("left", r)):>6,}   layer {layer + 1}: {u[layer]:5.1f}              {a[layer]:6.1f} {b[layer]:6.1f} {c[layer]:6.1f}                       {d[layer]:6.1f} {e[layer]:6.1f}')
print()
print('  Why a sign can kill a unit: connections into one live unit of layer 1, and the new start.')
print('   left     connections for one unit   units with 1 connection   of those dead with a new start')
x = load()['train'][0]
for r in [13, 18, 22]:
    per, one, dead = [], [], []
    for s in SEEDS:
        m = masks_of(s, r)
        live = live_units(m)[0]
        k = m[0].sum(1)[live]
        out = torch.relu(x @ (new_values(s)[0] * m[0]).t())[:, live]
        per.append(k.mean().item()); one.append(int((k == 1).sum())); dead.append(int(((out.sum(0) == 0) & (k == 1)).sum()))
    print(f'   {int(mean("left", r)):>6,}        {np.mean(per):4.2f}                    {np.mean(one):5.1f}                   {np.sum(dead) / np.sum(one) * 100:4.1f} %')
print()
print('  Right on the test before the first training step (10 % is a guess):')
print('   left       ticket (lowest, highest run)     new start   random places')
for r in [0, 3, 7, 13, 18, 22]:
    t, n, z = (np.array([run(s)[f'{k}_test'][r][0] for s in SEEDS]) * 100 for k in ['ticket', 'new', 'random'])
    print(f'   {int(mean("left", r)):>6,}     {t.mean():5.1f}  ({t.min():.1f}, {t.max():.1f})               {n.mean():5.1f}       {z.mean():5.1f}')
