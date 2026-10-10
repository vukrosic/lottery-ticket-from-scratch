"""Step 6. Cheaper ways to a small network: all at once, picked before training, a narrow plain network."""
import numpy as np
import torch
from runs import run, mean, narrow, SEEDS, CLOSE, TOTAL, ROUNDS
from lottery import load, start_values

print('STEP 6  Shortcuts')
print('  rounds:   the ticket of the search (18 rounds of training for 1,037 connections)')
print('  at once:  train the big network one time, keep the biggest, back to the first values')
print('  before:   keep the biggest first values, with no training at all')
print('  narrow:   a plain network 64 -> h -> 10 with as many connections, 1,500 and 6,000 steps')
print('  keep on:  delete in rounds and keep the trained values (no way back to the start)')
print('   left      rounds  at once  before  random     narrow (h)  1,500   6,000     keep on    "before" shares with the ticket')
for r in CLOSE:
    h = int(run(0)['c_narrow_size'][CLOSE.index(r)][0])
    print(f'   {int(mean("left", r)):>6,}    {mean("ticket_end", r)[0] * 100:5.1f}   {mean("c_at_once", r) * 100:5.1f}   {mean("c_before", r) * 100:5.1f}   {mean("random_end", r)[0] * 100:5.1f}'
          f'       {h:>4}       {mean("c_narrow", r) * 100:5.1f}   {mean("c_narrow_long", r) * 100:5.1f}      {mean("keep_on", r) * 100:5.1f}       {mean("c_overlap", r) * 100:5.1f} %')
print(f'  Training runs to find a ticket with 1,037 connections: 18 of the big network and its smaller rounds, 27,000 steps.')
print(f'  Training runs for the narrow network: 1.')
x = load()['train'][0]
dead = [int((torch.relu(x @ start_values(s, narrow(1037))[0].t()).sum(0) == 0).sum()) for s in SEEDS]
print(f'  Dead units of the narrow network before step 1 (14 units, each with 64 connections), ten runs: {dead}')
