"""Step 4. Two objections: give the new start more time, or bigger steps."""
from runs import mean, CLOSE, TOTAL, LONG, BIG_LR
from lottery import STEPS, LR

print('STEP 4  More time and bigger steps')
print(f'  time: {LONG:,} steps, not {STEPS:,}.   big steps: step size {BIG_LR}, not {LR}.')
print('                        1,500 steps      6,000 steps      big steps        steps to the lowest watch loss, big steps')
print('   connections left    ticket   new     ticket   new     ticket   new      ticket   new')
for r in CLOSE:
    left = int(mean('left', r))
    a = [mean(k, r)[0] * 100 if not k.startswith('c_') else mean(k, r) * 100 for k in ['ticket_end', 'new_end']]
    b = [mean(k, r) * 100 for k in ['c_ticket_long', 'c_new_long', 'c_ticket_big', 'c_new_big']]
    c = [mean(k, r, 1) for k in ['c_ticket_big', 'c_new_big']]
    print(f'   {left:>6,} ({left / TOTAL * 100:4.1f} %)     {a[0]:5.1f}  {a[1]:5.1f}    {b[0]:5.1f}  {b[1]:5.1f}    {b[2]:5.1f}  {b[3]:5.1f}     {c[0]:5.0f}   {c[1]:5.0f}')
