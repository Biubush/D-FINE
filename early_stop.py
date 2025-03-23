"""
早停训练脚本 - 监控训练指标并在指标不再提升时提前终止训练
用于D-FINE模型的训练过程中添加早停策略

使用方法:
    python early_stop.py -c CONFIG [--use-amp] [--seed=SEED] [--patience=PATIENCE] [--min-delta=MIN_DELTA] [--min-epochs=MIN_EPOCHS]

参数:
    -c CONFIG: 配置文件路径，与原训练脚本相同
    --use-amp: 是否使用混合精度训练
    --seed: 随机种子
    --patience: 容忍的训练周期数，在这些周期内如果指标没有改善则停止训练，默认为5
    --min-delta: 指标改善的最小阈值，默认为0.001
    --min-epochs: 最小训练周期数，在此之前不启用早停，默认为20
    --monitor: 监控的指标，默认为'AP50:95'
"""

import os
import sys
import argparse
import torch
import json
import numpy as np

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils, stats
from src.solver import TASKS
from src.solver.det_engine import evaluate, train_one_epoch

# 检查是否安装了wandb
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
    print("警告: 未安装wandb，将禁用wandb相关功能")

class EarlyStopping:
    """早停机制
    
    在指定的patience个epoch内，如果监控指标没有提高min_delta，则停止训练
    
    Args:
        patience (int): 容忍的训练周期数
        min_delta (float): 指标改善的最小阈值
        min_epochs (int): 最小训练周期数，在此之前不启用早停
        monitor (str): 监控的指标名称
        mode (str): 'max'表示监控指标越大越好，'min'表示越小越好
    """
    def __init__(self, patience=5, min_delta=0.001, min_epochs=20, monitor='AP50:95', mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = -1
        
        # 验证mode的值是否合法
        if mode not in ['min', 'max']:
            raise ValueError(f"mode {mode} is unknown!")
        
        # 根据mode设置比较函数
        self.monitor_op = np.greater if mode == 'max' else np.less
        self.min_delta = min_delta if mode == 'max' else -min_delta
    
    def __call__(self, epoch, metric_value):
        """检查是否应该停止训练
        
        Args:
            epoch (int): 当前训练周期
            metric_value (float): 当前周期的指标值
            
        Returns:
            bool: 如果应该停止训练则返回True，否则返回False
        """
        # 当前周期小于最小周期时，不启用早停
        if epoch < self.min_epochs:
            return False
            
        if self.best_score is None:
            # 首次调用时初始化最佳分数
            self.best_score = metric_value
            self.best_epoch = epoch
            return False
        
        # 指标是否有改善
        if self.monitor_op(metric_value - self.min_delta, self.best_score):
            self.best_score = metric_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping: 指标{self.monitor}未提升 - 计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"EarlyStopping: 触发早停! 最佳指标{self.monitor}={self.best_score:.4f}，出现在第{self.best_epoch}个epoch")
                return True
        return False

class EarlyStoppingDetSolver(TASKS["detection"]):
    """带有早停功能的检测Solver
    
    继承自原始的检测Solver，添加了早停策略
    """
    def __init__(self, cfg, early_stopping):
        super().__init__(cfg)
        self.early_stopping = early_stopping
        
    def fit(self):
        """训练模型，添加了早停功能"""
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
        # 找到监控指标在metric_names中的索引
        if self.early_stopping.monitor in metric_names:
            monitor_idx = metric_names.index(self.early_stopping.monitor)
        else:
            print(f"警告: 监控指标 {self.early_stopping.monitor} 不在可用指标列表中，将使用AP50:95")
            self.early_stopping.monitor = "AP50:95"
            monitor_idx = 0

        if self.use_wandb and has_wandb:
            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        
        # 主要训练循环
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            # =================== 早停检查 ===================
            # 获取监控指标的值
            if "coco_eval_bbox" in test_stats and len(test_stats["coco_eval_bbox"]) > monitor_idx:
                monitor_value = test_stats["coco_eval_bbox"][monitor_idx]
                
                # 检查是否应该停止训练
                if self.early_stopping(epoch, monitor_value):
                    print(f"早停触发! 在第{epoch}个epoch停止训练。")
                    print(f"最佳{self.early_stopping.monitor}={self.early_stopping.best_score:.4f}，出现在第{self.early_stopping.best_epoch}个epoch")
                    
                    # 保存最终状态
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "early_stopped.pth")
                        print(f"最终模型已保存到 {self.output_dir}/early_stopped.pth")
                    
                    # 记录早停信息到日志
                    log_stats = {
                        "early_stopped": True,
                        "best_epoch": self.early_stopping.best_epoch,
                        f"best_{self.early_stopping.monitor}": float(self.early_stopping.best_score),
                        "stop_epoch": epoch
                    }
                    if self.output_dir and dist_utils.is_main_process():
                        with (self.output_dir / "early_stop_log.txt").open("w") as f:
                            f.write(json.dumps(log_stats) + "\n")
                    
                    break  # 终止训练循环
            # ==================================================

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        "epoch": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb and has_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


def main(args):
    """主函数"""
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k
            not in [
                "update",
                "patience",
                "min_delta",
                "min_epochs",
                "monitor"
            ]
            and v is not None
        }
    )

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if safe_get_rank() == 0:
        print("cfg: ")
        pprint(cfg.__dict__)
    
    # 创建早停对象
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        min_epochs=args.min_epochs,
        monitor=args.monitor
    )
    
    # 创建自定义solver
    solver = EarlyStoppingDetSolver(cfg, early_stopping)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    # 添加必要的导入项
    import time
    import datetime
    from pprint import pprint
    
    def safe_get_rank():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="带早停策略的D-FINE训练脚本")
    
    # 原始train.py参数
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--use-amp", action="store_true", help="auto mixed precision training")
    parser.add_argument("--output-dir", type=str, help="output directoy")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summry")
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")
    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", type=int, help="local rank id")
    
    # 早停相关参数
    parser.add_argument("--patience", type=int, default=5, 
                        help="在指标未提升的情况下等待的训练周期数，默认为5")
    parser.add_argument("--min-delta", type=float, default=0.001, 
                        help="指标改善的最小阈值，默认为0.001")
    parser.add_argument("--min-epochs", type=int, default=20, 
                        help="最小训练周期数，在此之前不启用早停，默认为20")
    parser.add_argument("--monitor", type=str, default="AP50:95", 
                        help="监控的指标，默认为'AP50:95'，可选值：AP50:95, AP50, AP75, APsmall, APmedium, APlarge")
    
    args = parser.parse_args()
    
    # 打印早停配置信息
    print(f"\n早停配置:")
    print(f"监控指标: {args.monitor}")
    print(f"容忍周期: {args.patience}")
    print(f"最小改善阈值: {args.min_delta}")
    print(f"最小训练周期: {args.min_epochs}")
    print(f"=" * 50)
    
    # 运行主函数
    main(args) 