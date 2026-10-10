"""Step 2. The search of the paper: train, delete the smallest fifth, back to the first values, train again."""
import numpy as np
from runs import run, mean, SEEDS, TOTAL, ROUNDS

print('STEP 2  The search in rounds (Frankle and Carbin, section 1)')
print('  Every round: train 1,500 steps, delete 20 % of what is left (10 % in the last table),')
print('  put every connection that is left back to its very first value.')
print('   round   connections left     right on the test: mean (lowest, highest of ten runs)')
for r in range(ROUNDS + 1):
    left = int(mean('left', r))
    v = np.array([run(s)['ticket_end'][r][0] for s in SEEDS]) * 100
    print(f'   {r:>3}     {left:>6,}  ({left / TOTAL * 100:5.1f} %)     {v.mean():5.1f} %   ({v.min():.1f}, {v.max():.1f})')
m = [int(v) for v in mean('per_table', 18)]
print(f'  After 18 rounds the three tables keep {m[0]}, {m[1]} and {m[2]} connections.')
u = mean('units', 18)
print(f'  Units that still have a way in and a way out: {u[0]:.0f} of 300 and {u[1]:.0f} of 100 (means).')
