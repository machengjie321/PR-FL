python3 experiments/FEMNIST/adaptive.py -na -ni -s 0 -e conventional && python3 experiments/FEMNIST/adaptive.py -a -i -s 0 -e adaptive && python3 experiments/FEMNIST/iterative.py -s 0 -e iterative && python3 experiments/FEMNIST/online.py -s 0 -e online && python3 experiments/FEMNIST/reinitialize.py -m r -s 0 -e reinit && python3 experiments/FEMNIST/reinitialize.py -m rr -s 0 -e random_reinit && python3 experiments/FEMNIST/snip.py -s 0 -e snip

conda activate /data/mcj/conda_env/d2l
cd /data/mcj/Prune_fl
source setenv.sh     #setup the path
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


conda activate /mnt/sda1/mcj/conda_env/d2l
cd /mnt/sda1/mcj/PruneFL-master/PruneFL-master/
source setenv.sh
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 2 -m buff_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_wg -t 0.07 -g 0 -pt fast -ps nl -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_w -t 0.07 -g 0 -pt fast -ps nl -pm w  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_fast_l_wg -t 0.07 -g 0 -pt fast -ps l -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_wg -t 0.2 -g 1 -pt fast -ps nl -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_w -t 0.2 -g 1 -pt fast -ps nl -pm w  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_fast_l_wg -t 0.2 -g 1 -pt fast -ps l -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


log_filename="log_$(date +'%Y%m%d%H%M%S').log"
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_g -t 0.07 -g 1 -pt fast -ps nl -pm g  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_normal_nl_g -t 0.07 -g 0 -pt normal -ps nl -pm g > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_normal_nl_wg -t 0.07 -g 0 -pt normal -ps nl -pm wg > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_wg -t 0.07 -g 1 -pt fast -ps nl -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_normal_nl_w -t 0.07 -g 0 -pt normal -ps nl -pm w > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_w -t 0.07 -g 1 -pt fast -ps nl -pm w  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_normal_l_wg -t 0.07 -g 0 -pt normal -ps l -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -a -ni -s 0 -e adaptive_fast_l_wg -t 0.07 -g 1 -pt fast -ps l -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/adaptive.py -na -ni -s 0 -e conventional -t 0.07 -g 1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/snip.py -s 0 -e snip -t 0.07 -g 1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/FEMNIST/online.py -s 0 -e online -g 0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_g -t 0.2 -g 1 -pt fast -ps nl -pm g  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_normal_nl_g -t 0.2 -g 0 -pt normal -ps nl -pm g > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_normal_nl_wg -t 0.2 -g 0 -pt normal -ps nl -pm wg > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_wg -t 0.2 -g 1 -pt fast -ps nl -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_normal_nl_w -t 0.2 -g 0 -pt normal -ps nl -pm w > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_fast_nl_w -t 0.2 -g 1 -pt fast -ps nl -pm w  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_normal_l_wg -t 0.2 -g 0 -pt normal -ps l -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -a -ni -s 0 -e adaptive_fast_l_wg -t 0.2 -g 1 -pt fast -ps l -pm wg  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/adaptive.py -na -ni -s 0 -e conventional -t 0.2 -g 0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/snip.py -s 0 -e snip -g 1 -t 0.2 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CIFAR10/online.py -s 0 -e online -g 1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



nohup python3 experiments/CelebA/adaptive.py -na -ni -s 0 -e conventional -t 0.3 -g 3 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python3 experiments/CelebA/adaptive.py -a -ni -s 0 -e adaptive_prune -t 0.2 -g 3 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CelebA/adaptive.py -a -ni -s 0 -e adaptive_fast -t 0.2 -g 2 -fast > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CelebA/iterative.py -s 0 -e iterative -g 2 -t 0.2 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/CelebA/snip.py -s 0 -e snip -g 3 -t 0.2 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python3 experiments/CelebA/online.py -s 0 -e online -g 1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python3 experiments/ImageNet100/adaptive.py -na -ni -s 0 -e conventional -t 0.8 -g 3 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/ImageNet100/adaptive.py -a -ni -s 0 -e adaptive_prune -t 0.8 -g 0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python3 experiments/ImageNet100/adaptive.py -a -ni -s 0 -e adaptive_fast -t 0.8 -g 0 -fast > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/iterative.py -s 0 -e iterative -g 1 -t 0.8 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/snip.py -s 0 -e snip -g 1 -t 0.8 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/online.py -s 0 -e online -g 2 -t 0.8 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn

python experiments/FEMNIST/PIF.py

python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re

nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.05 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 3 -m buff_mask_fed_avg -wd 0 -md 0.05 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.05 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.05 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 2 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.09 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.09 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 3 -m buff_mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 3 -m buff_mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.5 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.5 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 3 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m heterofl -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch syn -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch syn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m heterofl -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch syn -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch syn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m heterofl -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch syn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch syn -stal poly -re -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m mask_fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fed_avg -wd 0 -md 0.08 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m heterofl -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch syn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch syn -stal poly -re -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m mask_fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m fed_avg -wd 0 -md 0.10 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m heterofl -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m buff_mask_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m mask_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m fedfix -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m heterofl -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch syn -niid -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m buff_mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f u -d 0 -g 3 -m buff_mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m buff_mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 2 -m buff_mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m mask_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 2 -m buff_fed_avg -wd 0 -md 0.2 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re -niid -Res > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &




nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 20 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 20 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 20 -f n -d 0 -g 3 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 20 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 20 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 2 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 0 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.1 -i 10 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

 nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
 nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 3 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
 nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 50 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 3 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
  nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 1 -m fedasyn -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.1 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



python experiments/MNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re

nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 1 -m sub_fed_avg_test -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 2 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 0 -m sub_fed_avg_test -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 0 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly  -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 2 -m sub_fed_avg_test -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 0 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 1 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 1 -m sub_fed_avg_test -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 0 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly  -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 0 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg_test -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg_test -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg_test -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg_test -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 2 -m sub_fed_avg_test -md 0.4 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m sub_fed_avg -md 0.4 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 0 -m sub_fed_avg_test -md 0.4 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m sub_fed_avg -md 0.4 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m sub_fed_avg_test -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m sub_fed_avg_test -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m sub_fed_avg -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m sub_fed_avg_test -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 1 -m sub_fed_avg -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg_test -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m sub_fed_avg -md 1 -ac g -clr 0 -wdn 10 -wd 0 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.5 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f un_fair -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.5 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 3 -m sub_fed_avg -md 1 -ac g -clr 0 -wdn 10 -wd 0 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.5 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f un_fair -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.5 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 3 -m sub_fed_avg -md 1 -ac g -clr 0 -wdn 10 -wd 0 -ft n -ch asyn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 3 -m sub_fed_avg_test -md 1 -ac g -clr 0 -wdn 10 -wd 0 -ft n -ch asyn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 3 -m sub_fed_avg -md 1 -ac g -clr 0 -wdn 10 -wd 0 -ft n -ch asyn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m sub_fed_avg_test -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg_test -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &






nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -stal poly > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CelebA/Prune_increase_FL_CMD.py -ic 0.2 -i 10 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30  -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch syn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch syn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -ic 0.2 -i 100 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.3 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &




nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 2 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 3 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 1 -m sub_fed_avg -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch syn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 0 -m sub_fed_avg_test -wd 0 -md 1.0 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 2 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -ic 0.2 -i 30 -f n -d 0 -g 3 -m sub_fed_avg_test -wd 0 -md 0.1 -ac g -clr 0 -wdn 10 -ft n -ch asyn > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



