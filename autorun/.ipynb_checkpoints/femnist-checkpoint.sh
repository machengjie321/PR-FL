python3 experiments/FEMNIST/adaptive.py -na -ni -s 0 -e conventional && python3 experiments/FEMNIST/adaptive.py -a -i -s 0 -e adaptive && python3 experiments/FEMNIST/iterative.py -s 0 -e iterative && python3 experiments/FEMNIST/online.py -s 0 -e online && python3 experiments/FEMNIST/reinitialize.py -m r -s 0 -e reinit && python3 experiments/FEMNIST/reinitialize.py -m rr -s 0 -e random_reinit && python3 experiments/FEMNIST/snip.py -s 0 -e snip

conda activate /data/mcj/conda_env/d2l
cd /data/mcj/Prune_fl
source setenv.sh     #setup the path
nohup python experiments/FEMNIST/Hetero_Fair_FL.py > /dev/tty &

conda activate /mnt/sda1/mcj/conda_env/d2l
cd /mnt/sda1/mcj/PruneFL-master/PruneFL-master/
source setenv.sh



nohup jupyter-lab --ip=0.0.0.0 --port=8896 --allow-root > jupyterlab.log 2>&1 &


nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd -1  > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m R2SP -wd -1  > no_increase_1_unfair_R2SP_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m fed_avg -wd -1  > no_increase_1_unfair_fed_avg_-1.log 2>&1 &

nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_unfair_sub_fed_avg_0.0001.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 30 -f unfair -d 1 -g 0 -m sub_fed_avg -wd 0.0001  > no_increase_30_unfair_sub_fed_avg_0.0001.log 2>&1 &

nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > increase_1_unfair_sub_fed_avg_0.0001.log 2>&1 &

nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f u -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_u_1_sub_fed_avg_0.0001.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f n -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_n_1_sub_fed_avg_0.0001.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f u -d -2 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_u_-2_sub_fed_avg_0.0001.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f n -d -2 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_n_-2_sub_fed_avg_0.0001.log 2>&1 &



(/data/mcj/conda_env/d2l) [mcj@omnisky 日 7月 30 14:23 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m sub_fed_avg -wd -1  > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
[11] 51212
(/data/mcj/conda_env/d2l) [mcj@omnisky 日 7月 30 14:24 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m R2SP -wd -1  > no_increase_1_unfair_R2SP_-1.log 2>&1 &
[12] 51213
(/data/mcj/conda_env/d2l) [mcj@omnisky 日 7月 30 14:24 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m fed_avg -wd -1  > no_increase_1_unfair_fed_avg_-1.log 2>&1 &
[13] 51431

(/data/mcj/conda_env/d2l) [mcj@omnisky 日 7月 30 14:09 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m sub_fed_avg -wd 0.0001  > no_increase_1_unfair_sub_fed_avg_0.0001.log 2>&1 &
[4] 48766
(/data/mcj/conda_env/d2l) [mcj@omnisky 日 7月 30 14:09 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 30 -f unfair -d 1 -g 0 -m sub_fed_avg -wd 0.0001  > no_increase_30_unfair_sub_fed_avg_0.0001.log 2>&1 &
[5] 48767
001.log 2>&1 &da_env/d2l) [mcj@omnisky 日 7月 30 14:09 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > increase_1_unfair_sub_fed_avg_0.0001.log
[6] 48768
(/data/mcj/conda_env/d2l) [mcj@omnisky 日 7月 30 14:09 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f u -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_u_1_sub_fed_avg_0.0001.log 2>&1 &
[7] 48769
1.log 2>&1 &onda_env/d2l) [mcj@omnisky 日 7月 30 14:09 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f n -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_n_1_sub_fed_avg_0.0001
[8] 48770
001.log 2>&1 &da_env/d2l) [mcj@omnisky 日 7月 30 14:09 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f u -d -2 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_u_-2_sub_fed_avg_0.00
[9] 48771
(/data/mcj/conda_env/d2l) [mcj@omnisky 日 7月 30 14:09 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f n -d -2 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_n_-2_sub_fed_avg_0.0001.log 2>&1 &
[10] 48772


001.log 2>&1 &da_env/d2l) [mcj@omnisky 日 7月 30 14:09 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > increase_1_unfair_sub_fed_avg_0.00
[6] 48768


(1,0.5,0.2)
(/data/mcj/conda_env/d2l) [mcj@omnisky 一 7月 31 11:27 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_unfair_sub_fed_avg_0.0001.log 2>&1 &
[4] 24528
(/data/mcj/conda_env/d2l) [mcj@omnisky 一 7月 31 11:28 /data/mcj/Prune_fl]$ python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f u -d 1 -g 1 -m sub_fed_avg -wd 0.0001  > no_increase_1_u_1_sub_fed_avg_0.0001.log 2>&1 &
[5] 24619
(/data/mcj/conda_env/d2l) [mcj@omnisky 一 7月 31 11:26 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m sub_fed_avg -wd -1  > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
[1] 24262
(/data/mcj/conda_env/d2l) [mcj@omnisky 一 7月 31 11:26 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m R2SP -wd -1  > no_increase_1_unfair_R2SP_-1.log 2>&1 &
[2] 24263
&1 &ta/mcj/conda_env/d2l) [mcj@omnisky 一 7月 31 11:26 /data/mcj/Prune_fl]$ nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m fed_avg -wd -1  > no_increase_1_unfair_fed_avg_-1.log 2>&
[3] 24264


nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m R2SP -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &

nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd -1 -md 0.1 -ac w > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m sub_fed_avg -wd 0.00001 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd -1 -md 0.2 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd -1 -md 0.4 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd -1 -md 0.6 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m sub_fed_avg -wd -1 -md 0.8 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 0 -m sub_fed_avg -wd -1 -md 1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f u -d 1 -g 0 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f u -d -2 -g 0 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f n -d 1 -g 0 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f n -d -2 -g 0 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &

nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f unfair -d 1 -g 1 -m sub_fed_avg -wd -1 -md 0.1 -ac w > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &

nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f u -d 0.5 -g 1 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f nu -d 0.5 -g 0 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f nu -d 1 -g 1 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &

nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f n -d 0.5 -g 0 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
nohup  python experiments/FEMNIST/Hetero_fair_FL_cmd.py -ic no_increase -i 1 -f n -d -4 -g 0 -m sub_fed_avg -wd -1 -md 0.1 -ac g > no_increase_1_unfair_sub_fed_avg_-1.log 2>&1 &
