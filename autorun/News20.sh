python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.9 && python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.8 && python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.7 && python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.6 && python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.5 && python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.4 && python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.3 && python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.2 && python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.1



conda activate /mnt/sda1/mcj/conda_env/d2l/
cd /mnt/sda1/mcj/PruneFL-master/PruneFL-master/
source setenv.sh
python3 experiments/NEWs/results/prodLDA/PruneFL_Prod_LDA.py -a -t 0.01 -r 2e-3
