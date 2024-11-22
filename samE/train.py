import os
import time
import tqdm
import random
import yaml
import argparse

from collections import defaultdict
from contextlib import redirect_stdout
# 引入 PyTorch 及其分布式计算工具
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import samE.config as exp_cfg_mod
import samE.models.samE_agent as samE_agent 
import samE.utils.ddp_utils as ddp_utils # 分布式工具
import samE.mvt.config as mvt_cfg_mod # 多视图任务配置

from samE.mvt.mvtc_sam import MVTC_Sam
from samE.models.samE_agent import print_eval_log, print_loss_log,save_hm_log
from samE.utils.get_dataset import get_dataset_sequence
from samE.utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    load_agent,
    RLBENCH_TASKS,
)

from samE.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
)


def get_model_size(model):
    """
    Calculate the size of a PyTorch model in bytes.
    """
    param_size = 0
    trainable_param_size = 0
    param_num = 0
    trainable_para_num = 0
    for param in model.parameters():
        param_num += param.nelement() 
        param_size += param.nelement() * param.element_size()
        trainable_para_num += param.nelement() if param.requires_grad else 0
        trainable_param_size += param.nelement() * param.element_size() if param.requires_grad else 0
        
    
    print(f'{model.__class__.__name__}\'s parameter size: {param_size/1024/1024}MB')
    print(f'{model.__class__.__name__}\'s trainable parameter size: {trainable_param_size/1024/1024}MB')
    
    print(f'{model.__class__.__name__}\'s parameter num: {param_num/1000/1000}M')
    print(f'{model.__class__.__name__}\'s trainable parameter num: {trainable_para_num/1000/1000}M')

# new train takes the dataset as input
def train(agent, dataset, training_iterations, rank=0):
    """
    执行训练循环
    :param agent: 模型智能体
    :param dataset: 训练数据集
    :param training_iterations: 训练迭代次数
    :param rank: 当前进程的分布式训练编号
    """
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):
        # 获取一个批
        raw_batch = next(data_iter)
        # 将数据批次转换到模型所需的设备上
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        # 任务数据以及语言目标
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        # batch["llama_lang_goal"] = 
        
        update_args = {
            "step": iteration,
        }
        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": False,
            }
        )
        # 更新模型
        return_out = agent.update(**update_args)
        # 每 100 步打印一次日志
        if iteration%100 == 0 and rank == 0:
            print(return_out['ah_log'])
            
    # 如果是主进程，打印loss日志   
    if rank == 0:
        log = print_loss_log(agent)
    return log


def save_agent(agent, path, epoch):
    """
    保存模型的状态字典
    :param agent: 模型智能体
    :param path: 模型保存路径
    :param epoch: 当前训练轮次
    """
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched
    
    # 如果模型被封装在 DDP 中，则保存其内部的实际模型
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )


def get_tasks(exp_cfg):
    """
    根据配置文件获取任务列表
    """
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks


def get_logdir(cmd_args, exp_cfg):
    """
    获取日志目录路径
    """
    log_dir = os.path.join(cmd_args.log_dir, exp_cfg.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    """
    将配置信息保存为日志文件
    """
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


def experiment(rank, cmd_args, devices, port):
    """experiment.
    :param rank: 当前进程的分布式训练编号
    :param cmd_args: 命令行参数
    :param devices: 设备列表或单一设备。如果是列表，使用分布式训练（DDP）
    :param port: 用于分布式训练通信的端口
    """
    device = devices[rank]
    device = f"cuda:{device}"
    # 判断是否是分布训练
    ddp = len(devices) > 1
    # 初始化分布式训练
    ddp_utils.setup(rank, world_size=len(devices), port=port)
    # 获取实验配置默认值
    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    
    # 如果提供了实验配置文件路径，则加载该文件
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    # 如果提供了额外的实验配置选项，则合并这些选项
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))
        
    # 如果是分布式训练，打印当前 rank 的运行信息
    if ddp:
        print(f"Running DDP on rank {rank}.")

    # 保存原始学习率和实验 ID
    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    # 根据设备数量动态调整学习率
    exp_cfg.peract.lr *= len(devices) * exp_cfg.bs   #  calculate the real lr in practice
    # 更新实验 ID 以反映额外选项
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.exp_cfg_opts)}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.mvt_cfg_opts)}"
        
    # 如果是主进程，打印实验配置
    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    # Things to change
    BATCH_SIZE_TRAIN = exp_cfg.bs
    NUM_TRAIN = 100  
    ACTION_HORIZON = exp_cfg.ah
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(10000 // (exp_cfg.bs * len(devices) / 16))
    EPOCHS = exp_cfg.epochs
    TRAIN_REPLAY_STORAGE_DIR = "replay/replay_sequence"  ## TODO
    # TEST_REPLAY_STORAGE_DIR = "replay/replay_val"
    log_dir = get_logdir(cmd_args, exp_cfg)
    tasks = get_tasks(exp_cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))

    if_chunk=True
    t_start = time.time()
    # 数据集划分
    get_dataset_func = lambda: get_dataset_sequence(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        DATA_FOLDER,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
        if_chunk=if_chunk,
        horizon = ACTION_HORIZON
    )
    # 获得划分后的数据集
    train_dataset, _ = get_dataset_func()
    t_end = time.time()
    print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    if exp_cfg.agent == "our":
        mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
        if cmd_args.mvt_cfg_path != "":
            mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
        if cmd_args.mvt_cfg_opts != "":
            mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

        mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
        mvt_cfg.freeze()

        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        # 构建samE模型
        samE = MVTC_Sam(                                                                             
            action_horizon=ACTION_HORIZON,
            renderer_device=device,
            **mvt_cfg,
        ).to(device)
        # 
        get_model_size(samE)
        # 重构为分布式模型
        if ddp:
            samE = DDP(samE, device_ids=[device], find_unused_parameters=True)

        # 包装成智能体
        agent = samE_agent.samEAgent(
            network=samE,
            action_horizon = ACTION_HORIZON,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/image_save",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        agent.build(training=True, device=device)
        
        
    else:
        assert False, "Incorrect agent"

    start_epoch = 0
    end_epoch = EPOCHS
    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume
        print(f"Recovering model and checkpoint from {exp_cfg.resume}")
        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1
    elif os.path.exists(f'{log_dir}/model_last.pth'):
        
        agent_path = f'{log_dir}/model_last.pth'
        print(f"resume from checkpoint")
        
        epoch = load_agent(agent_path, agent, only_epoch=False)
        print(f"Recovering model and checkpoint from {agent_path}, model epoch: {epoch}")
        start_epoch = epoch + 1
        
    dist.barrier()
    # 如果是主进程，记录实验配置
    if rank == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost() # 解冻配置进行修改
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir) # 保存配置到日志
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze() # 再次冻结配置
        tb = TensorboardManager(log_dir) # 初始化 TensorBoard 管理器

    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break
        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")
        out = train(agent, train_dataset, TRAINING_ITERATIONS, rank)

        if rank == 0:
            tb.update("train", i, out)

        if rank == 0:
            # TODO: add logic to only save some models
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_last.pth", i)
        i += 1

    if rank == 0:
        tb.close()
        print("[Finish]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--with-eval", action="store_true", default=False)

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]
    port = (random.randint(0, 3000) % 3000) + 27000
    
    # experiment(0, cmd_args, devices, port)
    mp.spawn(experiment, args=(cmd_args, devices, port), nprocs=len(devices), join=True)