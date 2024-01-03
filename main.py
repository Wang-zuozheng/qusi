import os
import random
import argparse
import numpy as np
import torch

from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer


def main(args):
    if 'coco' in args.datasets:
        cfg_data_file = os.path.join("./configs/data", args.data + "coco_lt.yaml")
        cfg_model_file = os.path.join("./configs/model", args.model + "clip_rn50_peft.yaml")
    else:
        cfg_data_file = os.path.join("./configs/data", args.data + "voc_lt.yaml")
        cfg_model_file = os.path.join("./configs/model", args.model + "clip_vit_b16_peft_voc.yaml")
    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    cfg.is_gcn = True if args.is_gcn else False
    cfg.is_prompt_tuning = True if args.is_prompt_tuning else False
    cfg.T = args.T
    cfg.wr = args.wr
    cfg.sparse_topk = args.sparse_topk
    cfg.reweight_p = args.reweight_p
    
    cfg.loss_type = args.loss_type
    cfg.vpt_shallow = True if args.is_vptsh else False
    cfg.vpt_deep = True if args.is_vptd else False
    cfg.adaptformer = True if args.is_af else False
    cfg.full_tuning = True if args.is_ft else False
    
    cfg.lr = args.lr # 1e-4 #0.01
    cfg.weight_decay = args.weight_decay #1e-4 #5e-4
    cfg.momentum = args.momentum
    cfg.gcn_lr = args.gcn_lr # 1e-4 #0.01
    cfg.gcn_weight_decay = args.gcn_weight_decay #1e-4 #5e-4
    cfg.gcn_momentum = args.gcn_momentum
    cfg.kl_lambda = args.kl_lambda
    
    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    setup_logger(cfg.output_dir)
    
    print("** Config **")
    print(cfg)
    print("************")
    
    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg)
    
    if cfg.zero_shot:
        trainer.test()
        return

    if cfg.test_train == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_train_True")]
            print("Model directory: {}".format(cfg.model_dir))

        trainer.load_model(cfg.model_dir)
        trainer.test("train")
        return

    if cfg.test_only == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_only_True")]
            print("Model directory: {}".format(cfg.model_dir))
        
        trainer.load_model(cfg.model_dir)
        trainer.test()
        return

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="", help="model config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    parser.add_argument("--datasets", "-datasets", type=str, default="coco", help="")
    parser.add_argument("--reweight_p", "-rp", type=float, default=0.2, help="")
    parser.add_argument("--sparse_topk", "-st", type=int, default=60, help="")
    parser.add_argument("--T", "-T", type=float, default=0.6, help="")
    parser.add_argument("--wr", "-wr", type=float, default=0.5, help="")
    parser.add_argument("--loss_type", "-loss", type=str, default="Focal", help="")
    parser.add_argument("--is_gcn", "-is_gcn", type=int, default=0, help="")
    parser.add_argument("--is_prompt_tuning", "-is_prompt_tuning", type=int, default=1, help="")
    parser.add_argument("--is_vptsh", "-is_vptsh", type=int, default=0, help="")
    parser.add_argument("--is_vptd", "-is_vptd", type=int, default=0, help="")
    parser.add_argument("--is_af", "-is_af", type=int, default=0, help="")
    parser.add_argument("--is_ft", "-is_ft", type=int, default=1, help="")
    
    parser.add_argument("--lr", "-lr", type=float, default=1e-4, help="")
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4, help="")
    parser.add_argument("--momentum", "-mom", type=float, default=0.9, help="")
    parser.add_argument("--gcn_lr", "-gcn_lr", type=float, default=1e-3, help="")
    parser.add_argument("--gcn_weight_decay", "-gcn_wd", type=float, default=1e-4, help="")
    parser.add_argument("--gcn_momentum", "-gcn_mom", type=float, default=0.9, help="")
    parser.add_argument("--kl_lambda", "-kl_lambda", type=float, default=0.2, help="")
  
    args = parser.parse_args()
    main(args)
