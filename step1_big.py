"""Step 1. The big network learns the digits. Then connections are deleted with no training after it."""
import numpy as np
from runs import run, mean, SEEDS, TOTAL

print('STEP 1  The big network: 64 squares -> 300 units -> 100 units -> 10 digits')
print(f'  {TOTAL:,} connections. 1,000 pictures to learn from, 500 for the test. 1,500 training steps.')
print('   run   right on the test   step with the lowest loss on the watch set')
for s in SEEDS:
    e = run(s)['ticket_end'][0]
    print(f'   {s}        {e[0] * 100:5.1f} %               {int(e[1]):>5}')
print(f'  Mean of the ten runs: {mean("ticket_end", 0)[0] * 100:.1f} %')
print()
print('  Delete the smallest connections of the trained network, all at once, and do not train again:')
print('   connections left      right on the test (mean of ten)')
for r in [0, 1, 3, 5, 7, 10, 13, 18]:
    left = int(mean('left', r))
    print(f'   {left:>6,}  ({left / TOTAL * 100:5.1f} %)     {mean("cut_only", r) * 100:5.1f} %')
