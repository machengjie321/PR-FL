import argparse
import os
import torch
import configs.femnist as config
import configs.network as network_config
from bases.nn.models.leaf import Conv2
from bases.vision.load import get_data_loader
from bases.optim.optimizer import SGD
from bases.fl.modules import Server
from bases.optim.optimizer_wrapper import OptimizerWrapper
from control.algorithm import ControlModule
from control.utils import ControlScheduler
from utils.functional import disp_num_params

from timeit import default_timer as timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed',
                        help="The seed to use for both server and clients.",
                        action='store',
                        dest='seed',
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument('-r', '--resume',
                        help="Resume previous prototype",
                        action='store_true',
                        dest='resume',
                        default=False,
                        required=False)

    mutex = parser.add_mutually_exclusive_group(required=True)#创建一个互斥组。 argparse 将会确保互斥组中只有一个参数在命令行中可用:
    mutex.add_argument('-a', '--adaptive',
                       help="Use adaptive pruning",
                       action='store_true',
                       dest='use_adaptive')
    mutex.add_argument('-na', '--no-adaptive',
                       help="Do not use adaptive pruning",
                       action='store_false',
                       dest='use_adaptive')

    return parser.parse_args()


class FEMNISTServer(Server):
    def get_init_extra_params(self):
        self.check_client_to_sparse()
        return [([i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)], self.client_is_sparse) for j in range(10)]


if __name__ == "__main__":
    #args = parse_args()
    #seed, resume, use_adaptive = args.seed, args.resume, args.use_adaptive
    seed, resume, use_adaptive = 0, False, True
    torch.manual_seed(seed)
    model = Conv2()
    test_loader = get_data_loader(config.EXP_NAME, data_type="test", num_workers=8, pin_memory=True)
    if use_adaptive:
        control_module = ControlModule(model, config=config)
        control_scheduler = ControlScheduler(init_max_dec_diff=config.MAX_DEC_DIFF,
                                             dec_half_life=config.ADJ_HALF_LIFE)
    else:
        control_module = None
        control_scheduler = None

    # Initial pruning
    ip_start = timer()
    print("Starting initial pruning")

    save_path = os.path.join("results", "exp_{}".format(config.EXP_NAME), "server")
    from utils.save_load import mkdir_save, load

    if True:
        print("save the inital model in "+os.path.join(save_path, "init_model.pt"))
        model = load(os.path.join(save_path, "init_model.pt"))


    server = FEMNISTServer(config, network_config, model, test_loader, seed, SGD, {"lr": config.INIT_LR}, use_adaptive,
                           use_evaluate=True, control=control_module, control_scheduler=control_scheduler,
                           resume=resume, init_time_offset=-(430.81933270400623))

    while True:
        terminate = server.main()#base.fl.modules.server.main
        if terminate:
            break


