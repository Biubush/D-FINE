"""
验证脚本 - 用于评估训练后的模型性能

使用方法:
    python val.py -c CONFIG --resume CHECKPOINT [--device DEVICE]

参数:
    -c CONFIG: 配置文件路径
    --resume: 模型检查点路径
    --device: 运行设备，默认为'cuda'
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from pprint import pprint

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from src.solver.det_engine import evaluate

def main(args):
    """主函数"""
    # 设置分布式训练
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    
    # 解析配置
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k not in ["update"]
            and v is not None
        }
    )
    
    cfg = YAMLConfig(args.config, **update_dict)
    
    if safe_get_rank() == 0:
        print("配置信息:")
        pprint(cfg.__dict__)
    
    # 创建solver
    solver = TASKS["detection"](cfg)
    
    # 加载检查点
    if args.resume:
        solver.load_resume_state(args.resume)
        print(f"已加载检查点: {args.resume}")
    
    # 执行验证
    solver.val()
    
    # 清理分布式环境
    dist_utils.cleanup()

if __name__ == "__main__":
    def safe_get_rank():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="D-FINE模型验证脚本")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备，默认为'cuda'")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output-dir", type=str,
                        help="输出目录")
    parser.add_argument("-u", "--update", nargs="+",
                        help="更新yaml配置")
    parser.add_argument("--print-method", type=str, default="builtin",
                        help="打印方法")
    parser.add_argument("--print-rank", type=int, default=0,
                        help="打印进程ID")
    parser.add_argument("--local-rank", type=int,
                        help="本地进程ID")
    
    args = parser.parse_args()
    
    # 运行主函数
    main(args) 