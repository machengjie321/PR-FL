python3 experiments/CelebA/adaptive.py -na -ni -s 0 -e conventional && python3 experiments/CelebA/adaptive.py -a -i -s 0 -e adaptive &&
 python3 experiments/CelebA/iterative.py -s 0 -e iterative && python3 experiments/CelebA/online.py -s 0 -e online &&
 python3 experiments/CelebA/reinitialize.py -m r -s 0 -e reinit && python3 experiments/CelebA/reinitialize.py -m rr -s 0 -e random_reinit
 && python3 experiments/CelebA/snip.py -s 0 -e snip