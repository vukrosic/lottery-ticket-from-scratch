# Lottery ticket from scratch

Code for the video **Delete 49 of 50 Connections. It Still Learns, From One Start Only**.
A network with 50,200 connections reads handwritten digits. A search deletes connections in rounds until 1,037 are left, and that small network still reads 96.6 % of new digits. Give the same 1,037 places new random numbers and it ends lower in all ten runs.
Six small programs build the test of Frankle and Carbin (2018) from an empty file and measure why the new start loses in this network. About 650 lines of Python 3 with numpy, torch and scikit-learn.

```bash
pip install numpy torch scikit-learn
sh run_all.sh         # six steps, the tests and the exercise; expected output is in demo-output.txt
```

Every training run is saved in `files/` (10 files, 1 MB), so the steps only read the runs. All of `run_all.sh` takes about half a minute on a laptop. The exercise trains 15 networks and takes about 15 seconds of it.
To measure everything again, delete `files/` and run `python3 runs.py` (5 runs at a time, about 3 minutes for each). A run with the same seed gives the same numbers again (`tests.py` checks this).

The data is on your disk after the install, nothing to download: the 1,797 handwritten digits of scikit-learn, 8 squares by 8, every square between 0 (white paper) and 1 (full ink). 1,000 pictures are for training, 297 for watching the training, 500 for the test.

## Words

| In the video | In the papers | In the code |
| --- | --- | --- |
| connection | weight | one entry of a weight table |
| first number | initial value | `start_values(seed)` |
| places | mask | a list of tables with 1 (kept) or 0 (deleted) |
| round | one iteration of pruning | train, `prune` a fifth, back to the first numbers |
| ticket | winning ticket | the places with their first numbers |
| new start | random reinitialisation | the same places with new random numbers |
| dead unit | dead ReLU | a unit that answers 0 for all 1,000 training pictures |

## Files

| File | Idea | What the run shows |
| --- | --- | --- |
| `lottery.py` | The network, training, deleting, the search | 196 lines. The network is 64 → 300 → 100 → 10. Training is 1,500 steps of Adam with 60 pictures for each step. |
| `step1_big.py` | The big network | 98.2 % right on the test (mean of ten runs). Delete the smallest connections with no training after it: half deleted 98.0 %, nine in ten deleted 82.5 %, 1,037 left 19.4 %. |
| `step2_search.py` | The search in rounds | Train, delete a fifth, go back to the first numbers. 10,795 left: 97.9 %. 2,960 left: 97.4 %. 1,037 left (round 18): 96.6 %. 462 left (round 22): 93.5 %. |
| `step3_new_start.py` | Are the places enough? | The same places with new random numbers: 93.9 % at 1,037 and 73.3 % at 462. The new start wins 0 of 10 runs at both sizes. The first numbers at random places: 73.1 % and 22.9 %. |
| `step4_objections.py` | More time, bigger steps | 6,000 steps: ticket 96.7, new start 95.8. Step size four times bigger: 96.7 and 95.0. At 462 left: 94.4 against 77.2, and 94.5 against 72.8. |
| `step5_signs.py` | Signs and sizes, dead units | First signs with new sizes 96.2 %. First signs with one size for all 96.4 %. New signs with first sizes 94.4 %. Every sign turned around 35.0 %. Layer 1 at 1,037 left: 194 live units, 96 of them on one connection. Dead before step 1: ticket 4, new start 66, turned 156. Before any training the ticket reads 77.5 %, the new start 11.4 %. |
| `step6_shortcuts.py` | Three cheaper ways | Delete all at once 90.6 %. Keep the biggest first numbers, no training 79.3 %. A plain network 64 → 14 → 10 with 1,036 connections: 96.0 % after 1,500 steps, 96.9 % after 6,000. |
| `runs.py` | The saved runs | Makes every run that is missing in `files/` and reads the ones that exist. |
| `tests.py` | Checks of the claims | 43 tests. |
| `exercise.py` | Change one number | See below. |

## Reading the numbers

- **One small problem.** Every number is the mean of ten runs of one network on 8 × 8 digits. The test has 500 pictures, so one picture is 0.2 of a point. The lowest and highest run are printed in step 2.
- **The effect depends on the setup.** Liu et al. (2018) tested larger networks with well chosen step sizes and found no gain from the first numbers. Here, four times the time makes the gap at 1,037 connections smaller (2.7 points to 0.9), and neither change closes it.
- **Dead units are our own measurement.** The papers do not give this reason. It holds here because every input square is between 0 and 1 and the first layer of the ticket has 346 connections for 194 units: a unit with one connection and a minus sign answers 0 for every picture, and training never corrects it.
- **The ticket is not the best small network here.** A plain narrow network with as many connections scores 96.0 % after one training and 96.9 % with four times the time. The ticket scores 96.6 % and 96.7 %, and costs 18 trainings (27,000 steps) to find.
- **The order of the signs tests.** The gap between "first signs, new sizes" (96.2) and the ticket (96.6) is two pictures of 500. The gap to "every sign turned around" (35.0) is 308 pictures.

## What this is not

- **Not a reproduction of the paper.** Frankle and Carbin train Lenet-300-100 on MNIST (28 × 28 pictures, about 266,000 weights) and convolutional networks on CIFAR-10. This network has 50,200 weights and reads 8 × 8 digits.
- **Not a way to train faster.** The search costs many trainings, and single deleted connections bring no speed on common hardware. The authors name both limits.
- **Not the method for deep networks.** On deep networks the way back to the very first numbers fails. Frankle, Dziugaite, Roy and Carbin (2019) go back to the numbers from a little into training (0.1 % to 7 % of the way). That step is not built here.

Sources: Frankle and Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", ICLR 2019, arXiv 1803.03635. Han, Pool, Tran, Dally, "Learning both Weights and Connections for Efficient Neural Networks", arXiv 1506.02626 (2015). Liu, Sun, Zhou, Huang, Darrell, "Rethinking the Value of Network Pruning", arXiv 1810.05270 (2018). Zhou, Lan, Liu, Yosinski, "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask", arXiv 1905.01067 (2019). Frankle, Dziugaite, Roy, Carbin, "Stabilizing the Lottery Ticket Hypothesis", arXiv 1903.01611 (2019). Ramanujan, Wortsman, Kembhavi, Farhadi, Rastegari, "What's Hidden in a Randomly Weighted Neural Network?", CVPR 2020, arXiv 1911.13299.

## Try it

The search in the video takes 18 rounds to get from 50,200 connections to 1,037: 18 trainings. Deleting all at once (1 round) is six points worse. `exercise.py` runs the search with 3 rounds and ends at the same 346, 541 and 150 connections.

```python
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
```

Guess the score first, then run `python3 exercise.py` (about 15 seconds, five runs). For comparison, the same five runs give 91.9 % with 1 round and 96.3 % with 18. Second exercise: `python3 exercise.py 6` and `python3 exercise.py 9`. Find the smallest number of rounds within half a point of 18.

MIT licence.
