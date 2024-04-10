# Description
This is the official code repository for the following paper accepted at TNNLS 2022:

# Setup
#python setup.py install #安装 生成库文件 且能生成环境变量 以便直接调库

```python3
sudo -E python3 setup.py install
```

# Run prototype 
For each new terminal, please run
```shell
source setenv.sh     #setup the path
```
```
#!/bin/bash

full_path=$(realpath "${BASH_SOURCE[0]}")
dir_path=$(dirname "$full_path")
export PYTHONPATH=$dir_path

```


in the `PR-FL` root folder for the correct environment.



```
to generate figures in `results/{experiment_name}/figs` folder for each experiment. Non-existing results will be skipped.

The code has been tested on Ubuntu 20.04, and example results are given in the `example_results` folder.
