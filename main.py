import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import json
import random



from flcore.servers import get_server
from flcore.pretty.logger import log


def parse_results(args):
    exp_name = os.path.join(args.dataset, args.image_backbone, args.fed_algo, args.prompt_algo)
    results_dir = os.path.join('results', exp_name, f'shot_{args.num_shot}')
    json_files = [os.path.join(results_dir, file) for file in os.listdir(results_dir) if 'exps' in file]

    with open(json_files[0], 'r') as json_file:
            json_data = json.load(json_file)
            accs = {k: [] for k in json_data.keys()}

    results = copy.deepcopy(accs)
    for file in json_files:
        with open(file) as json_file:
            json_data = json.load(json_file)
            for k,v in json_data.items():
                accs[k].append(v)
    for k,v in accs.items():
        results[k].append(np.mean(v)) # mean
        results[k].append(np.std(v)) # var
    with open(os.path.join(results_dir, 'summary.json'), 'w+') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()



def special_args(args):
    if args.prompt_algo == 'CoCoOp':
        args.batch_size = 1
        args.eval_scaler = 1
    if args.prompt_algo == 'OTP':
        args.fed_algo = 'FedOTP'
        args.num_prompt = 2
    if args.prompt_algo == 'BPL':
        args.batch_size = 1
        args.eval_scaler = 1
    if args.prompt_algo == 'ProDA':
        args.prompt_batch_size = 1
    if args.prompt_algo == 'CLIP':
        args.ctx_init = 'a photo of a'
    if args.central == 'true':
        args.fed_algo = 'Central'
    if args.prompt_algo == 'MaPLe':
        args.fed_algo = 'MaPLe'
        assert 'ViT' in args.image_backbone, 'MaPLe requires Transformer vision encoder'
        assert args.num_prompt == 1, 'MaPLe requires num_prompt = 1'
    if args.prompt_algo == 'PGP':
        args.fed_algo = 'FedPGP'
    if args.prompt_algo == 'Folio':
        args.fed_algo = 'Folio'
    if args.prompt_algo == 'DPFPL':
        args.precision = 'fp32'
        args.fed_algo = 'FedDPFPL'
    return args

def main(args):
    if args.deterministic == 'true':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
    if args.verbose:
        log.level = 'verbose'
    elif args.verbose2:
        log.level = 'debug'
    for i in range(args.times):
        server = get_server(args, i)
        start = time.time()
        server.run()
        end = time.time()
        log.info(f'Experiment run: {i}, Total time ellapsed: {int(end-start)}s')
    parse_results(args)



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-task', "--task", type=str, default='class',       # 指定任务类型
                                        choices=['class', 'seg', ], )
    parser.add_argument('-bench', "--benchmark", type=str, default="global",        # 不同评估维度
                        choices=['dual', 'global', 'personal',
                                 'base2novel', 'xdomain', 'multidomain', 'xdataset'])
    """
    global: 标准联邦学习，在全局测试集上评估（通用性能）。
    personal: 个性化联邦学习，在客户端本地数据上评估（个性化性能）。
    base2novel: Base-to-Novel 泛化。在“基础”类别上训练，在未见过的“新”类别上测试（零样本能力）。
    xdomain / multidomain: 跨域泛化 (Domain Generalization)。例如在素描图上训练，在真实照片上测试。
    xdataset: 跨数据集评估。
    """
    parser.add_argument('-falg', "--fed_algo", type=str, default='FedAvg',      # 服务端聚合算法
                                        choices=['FedAvg', 'FedOTP', ], )
    parser.add_argument('-palg', "--prompt_algo", type=str, default="CoOp",     # 本地提示学习算法
                                        choices=['CLIP', 'CoOp', 'CoCoOp', 'PLOT',
                                                 'ALIGN', 'ProDA', 'ProGrad', 'PromptSRC',
                                                 'KgCoOp','OTP', 'PGP',
                                                 'TPG', 'DPFPL', 'MaPLe', 'Folio',
                                                 'DenseCoOp'])
    # 是否进行集中式训练 (Centralized)。如果设为 true，则不进行联邦划分，视为将所有数据集中在一起训练（通常作为性能上限对比）。
    parser.add_argument('-ctr', "--central", type=str, default='false')
    parser.add_argument('-did', "--device_id", type=int, default=0)
    parser.add_argument('-t', "--times", type=int, default=1)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='summary')
    parser.add_argument('-slurm', "--slurm", type=str, default='false', choices=['true', 'false'])
    parser.add_argument('-detm', "--deterministic", type=str, default='true', choices=['true', 'false'])
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--verbose2", action='store_true')
    # model
    parser.add_argument('-ibb', "--image_backbone", type=str, default='RN50',       # CLIP 模型的视觉编码器骨干。
                                                    choices=['RN50', 'RN101', 'ViT-B/16'])  
    parser.add_argument('-m', "--model", type=str, default="resnet50")  # 指定模型名称.在提示学习场景下主要由 -palg 决定，这个参数可能用于纯 CNN 基准线
    # dataset
    parser.add_argument('-data', "--dataset", type=str, default="caltech101")   # 训练用的源数据集名称
    parser.add_argument('-tdata', "--target_dataset", type=str, default="caltech101")   # 目标数据集名称（用于跨域/跨数据集测试）。
    parser.add_argument('-root', "--data_root", type=str, default="~/data/prompt")
    parser.add_argument('-dnt', "--num_shot", type=int, default=1)  # Few-shot 设置。每个类别包含的样本数量
    parser.add_argument('-dns', "--num_shards", type=int, default=10)
    parser.add_argument('-dsm', "--split_mode", type=str, default='dirichlet',   # 联邦数据划分模式（Non-IID 设置的核心）。
                        choices=['dirichlet', 'iid', 'task','predefined'])
    parser.add_argument('-dsa', "--split_alpha", type=float, default=1)     # Dirichlet 分布的 Alpha 参数。
    parser.add_argument('-dsb', "--split_beta", type=float, default=1)
    parser.add_argument('-dtf', "--data_transform", type=str, default="default", choices=['default', 'randaug'])    # 数据增强策略
    parser.add_argument('-dlt', "--drop_last", type=str, default='false', choices=['true', 'false'])
    parser.add_argument('-dpl', "--parallel", type=str, default='true', choices=['true', 'false'])
    parser.add_argument('-dnw', "--num_workers", type=int, default=8)
    # general federated learning settings
    parser.add_argument('-nc', "--num_clients", type=int, default=4)    # 客户端总数
    parser.add_argument('-cevl', "--client_eval", type=str, default='false', choices=['true', 'false']) # 是否在客户端本地数据上进行评估（用于计算个性化精度）
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)    # 全局通信轮数（服务器聚合的总次数）。
    parser.add_argument('-le', "--local_epochs", type=int, default=1,   # 本地训练轮数（每轮通信前，客户端在本地更新多少个 Epoch）。
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-tf', "--train_fraction", type=float, default=1.)  # 每轮参与训练的客户端比例（0-1之间）。
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0)
    # optimization
    parser.add_argument('-opn', "--optim_name", type=str, default='sgd')    # 优化器名称
    parser.add_argument('-orho', "--optim_rho", type=float, default=0.01)
    parser.add_argument('-lrs', "--lr_scheduler", type=str, default='', choices=['', 'cos'])
    parser.add_argument('-lbs', "--batch_size", type=int, default=8)    # 本地训练的 Batch Size（显存不足时调小此参数）。
    parser.add_argument('-lvbs', "--eval_scaler", type=int, default=1)
    parser.add_argument('-evrds', "--eval_rounds", type=int, default=1)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-omt', "--optim_momentum", type=float, default=0.9,)
    parser.add_argument('-owd', "--optim_weight_decay", type=float, default=0.0001,)
    parser.add_argument('-lrd', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-lrdg', "--learning_rate_decay_gamma", type=float, default=0.99)
    # 训练精度。fp32 (全精度), fp16 (半精度), amp (自动混合精度，节省显存)。
    parser.add_argument('-prec', "--precision", type=str, default="fp32", choices=['fp32', 'fp16', 'amp'])
    parser.add_argument('-iws', "--init_weights", type=str, default=None,)
    parser.add_argument('-lst', "--loss_type", type=str, default='ce', choices=['ce', 'bce'])
    parser.add_argument('-gcn', "--grad_clipping_norm", type=float, default=0.,)
    parser.add_argument('-seed', "--seed", type=int, default=0)
    # prompt learning general settings
    parser.add_argument('-npt', "--num_prompt", type=int, default=1)    # 提示向量的数量（例如 16 个 token）。
    parser.add_argument('-nptv', "--num_prompt_vision", type=int, default=1)
    parser.add_argument('-nctx', "--num_context", type=int, default=4)  # 上下文长度。
    parser.add_argument('-ctp', "--class_token_position", type=str, default='end')  # 类别 Token [CLASS] 放在提示语句的哪个位置（end, middle, front）。
    # 是否使用类别特定的上下文（CoCoOp 的特性，每个类别有独立的 Context）。
    parser.add_argument('-csc', "--class_specific_context", type=str, default='false', choices=['true', 'false'])
    parser.add_argument('-cti', "--ctx_init", type=str, default='') # 提示初始化文本。例如 "a photo of a"，用于手动初始化提示向量，而不是随机初始化。
    # prompt learning algorithms' settings
    parser.add_argument('-pbsz', "--prompt_batch_size", type=int, default=0) # ProDA
    parser.add_argument('-pgpbt', "--pgp_bottleneck", type=int, default=8) # FedPGP 的瓶颈层大小
    parser.add_argument('-frac', "--folio_frac", type=float, default=0.2) # PromptFolio
    parser.add_argument('-dprank', "--dpfpl_rank", type=int, default=8) # DP-FPL
    parser.add_argument('-noise', "--noise", type=float, default=0.4) # DP-FPL
    parser.add_argument('-nthr', "--norm_thresh", type=float, default=10) # DP-FPL
    parser.add_argument('-fact', "--factorization", type=str, default='dpfpl') # DP-FPL
    parser.add_argument('-pdepth', "--prompt_depth", type=int, default=1) # MaPLe
    # segmentation task settings
    parser.add_argument('-stls', "--seg_text_loss_scale", type=float, default=1) #
    # FL algorithms' settings

    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)
    if args.slurm == 'true':
        print(f'Job {args.prompt_algo} allocated GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        args.device_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    args = special_args(args)
    main(args)