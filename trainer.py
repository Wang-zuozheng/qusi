import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224

import datasets
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision import datasets as datasets
import copy
from randaugment import RandAugment
import mmcv

def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp32" or cfg.prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(cfg):
    backbone_name = cfg.backbone
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model

class CocoDetection(datasets.coco.CocoDetection):

    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def labels(self):
        return [v["name"] for v in self.coco.cats.values()]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:  # type: ignore
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']  # type: ignore
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = target.max(dim=0)[0]
        return img, target

class WeakStrongDataset(torch.utils.data.Dataset):  # type: ignore

    def __init__(self,
                 root,
                 annFile,
                 transform,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        self.name = names
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform
        self.strong_transform: transforms.Compose = copy.deepcopy(
            transform)  # type: ignore
        self.strong_transform.transforms.insert(0, RandAugment(3, 5))  # type: ignore

    def __getitem__(self, index):
        name = self.name[index]
        
        # path = name.strip('\n').split(',')[0]
        # num = name.strip('\n').split(',')[1]
        # num = num.strip(' ').split(' ')
        
        temp = name.strip('\n').split(' ')
        path = temp[0]
        if 'COCO' in self.root:
            num = temp[1: len(temp)-1]
        elif 'voc' in self.root:
            num = temp[1: ]
        else:
            ValueError
        
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        img_w = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)  # type: ignore # noqa
        assert (self.target_transform is None)
        return [index, img_w,
                self.transform(img),
                self.strong_transform(img)], label

    def __len__(self):
        return len(self.name)


class Trainer:
    def __init__(self, cfg):
        # 设置设备device
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.build_relation()
        self.relation_process(cfg)
        self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
        self._writer = None

    def build_data_loader(self):
        cfg = self.cfg
        resolution = cfg.resolution
        expand = cfg.expand
        
        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_plain = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])  
        
        # test端做不做集成
        if cfg.test_ensemble:
            transform_test = transforms.Compose([
                transforms.Resize(resolution + expand),
                transforms.FiveCrop(resolution),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Normalize(mean, std),
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(resolution * 8 // 7),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                transforms.Normalize(mean, std),
            ])
        
        # COCO Data loading
        instances_path_val = os.path.join(cfg.root,
                                      cfg.test_path)
      
        instances_path_train = cfg.train_path
    
        data_path_val = cfg.data_path_val
        data_path_train = cfg.data_path_train
        test_dataset = CocoDetection(data_path_val, instances_path_val,
                                transform_plain)
        train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      transform_train,
                                      class_num=80)
        
        self.num_classes = train_dataset.class_num

        self.co = torch.zeros(self.num_classes, self.num_classes)
        
        # Pytorch Data loader
        train_loader = torch.utils.data.DataLoader(  
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True)

        test_loader = torch.utils.data.DataLoader(  # type: ignore
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=False)
        
        self.train_loader = train_loader
        self.test_loader = test_loader

        class_freq = np.asarray(mmcv.load('/home/wzz/LMPT/data/voc/class_freq.pkl')['class_freq'])
        self.many_idxs = list(np.where(class_freq>=100)[0])
        self.med_idxs = list(np.where((class_freq<100) * (class_freq >= 20))[0])
        self.few_idxs = list(np.where(class_freq<20)[0])

        self.classnames = self.test_loader.dataset.labels()
        self.num_classes = len(self.classnames)
        
        
        assert cfg.batch_size % cfg.micro_batch_size == 0
        self.accum_step = cfg.batch_size // cfg.micro_batch_size


    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = ZeroShotCLIP(clip_model)
            self.model.to(self.device)
            self.tuner = None
            self.head = None

            prompts = self.get_tokenized_prompts(classnames)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg)
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes, self.classnames)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head
            self.gcn = self.model.gcn

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            self.head = self.model.head
        
        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg
        prompt_p = []
        if cfg.full_tuning == True:
            for name, param in self.model.named_parameters():
                param.requires_grad_(True)
                if "text_encoder" in name:
                    param.requires_grad_(False)
        else:
            print("Turning off gradients in the model")
            for name, param in self.model.named_parameters():
                param.requires_grad_(False)
            print("Turning on gradients in the prompt")
            for name, param in self.model.named_parameters():
                if "prompt_learner" in name:
                    param.requires_grad_(True)
                    prompt_p.append(param)
            print("Turning on gradients in the tuner")
            for name, param in self.tuner.named_parameters():
                param.requires_grad_(True)
            print("Turning on gradients in the head")
            for name, param in self.head.named_parameters():
                param.requires_grad_(True)
            print("Turning on gradients in the gcn")
            for name, param in self.gcn.named_parameters():
                param.requires_grad_(True)
        

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        gcn_params = sum(p.numel() for p in self.gcn.parameters())
        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        print(f"Head params: {head_params}")
        print(f"Gcn params: {gcn_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay, "momentum": cfg.momentum},
                                      {"params": self.head.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay, "momentum": cfg.momentum},
                                      {"params": self.gcn.parameters(), "lr": cfg.gcn_lr, "weight_decay": cfg.gcn_weight_decay, "momentum": cfg.gcn_momentum},
                                      {"params": prompt_p, "lr": cfg.lr, "weight_decay": cfg.weight_decay, "momentum": cfg.momentum}],
                                      )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg
        # cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        if cfg.loss_type == "CE":
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
            self.criterion = LDAMLoss(cls_num_list=self.cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
            self.criterion = ClassBalancedLoss(cls_num_list=self.cls_num_list)
        elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
            self.criterion = GeneralizedReweightLoss(cls_num_list=self.cls_num_list)
        elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
            self.criterion == BalancedSoftmaxLoss(cls_num_list=self.cls_num_list)
        elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
            self.criterion = LogitAdjustedLoss(cls_num_list=self.cls_num_list)
        elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
            self.criterion = LADELoss(cls_num_list=self.cls_num_list)
        elif cfg.loss_type == "DBLoss": # https://arxiv.org/abs/1708.02002
            if 'COCO' in cfg.root:
                freq_file = 'data/coco/class_freq.pkl'
                self.criterion = DBLoss(
                        freq_file=freq_file,
                        weight=None, gamma1=4.0, gamma2=0.0,
                        logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                    )
            else:
                freq_file='data/voc/class_freq.pkl'
                self.criterion = ResampleLoss(
                    use_sigmoid=True,
                    reweight_func='rebalance',
                    focal=dict(focal=True, balance_param=2.0, gamma=2),
                    logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                    loss_weight=1.0, freq_file=freq_file
                )
        
    def build_relation(self):
        prompts = self.get_tokenized_prompts(self.classnames, template = "a photo of a {}.")
        class_features = self.model.encode_text(prompts)
        class_features = F.normalize(class_features, dim=-1)
        self.relation = class_features @ class_features.T
    
    def relation_process(self, cfg):
        _ ,max_idx = torch.topk(self.relation, cfg.sparse_topk)
        mask = torch.ones_like(self.relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        self.relation[mask] = 0
        sparse_mask = mask
        dialog = torch.eye(self.num_classes).type(torch.bool)
        self.relation[dialog] = 0
        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1) * cfg.reweight_p
        self.relation[dialog] = 1-cfg.reweight_p

        self.gcn_relation = self.relation.clone()
        assert(self.gcn_relation.requires_grad == False)
        self.relation = torch.exp(self.relation/cfg.T) / torch.sum(torch.exp(self.relation/cfg.T), dim=1).reshape(-1,1)
        self.relation[sparse_mask] = 0
        self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1)
        
        
    def get_tokenized_prompts(self, classnames, template = "a photo of a {}."):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Initialize head with text features")
        prompts = self.get_tokenized_prompts(classnames)
        text_features = self.model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def init_head_class_mean(self):
        print("Initialize head with class means")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        for batch in tqdm(self.train_init_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        # acc_meter = AverageMeter(ema=True)
        # cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.tuner.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                
                
                image_ = batch[0]
                label = batch[1]
                image = image_[1].to(self.device)
                
                image_w = image_[2].to(self.device)
                image_s = image_[3].to(self.device)
                label = label.float().to(self.device)
                
                if cfg.prec == "amp":
                    with autocast():
                        output = self.model(image, return_feature=True, gcn_relation=self.relation)#
                        loss = self.criterion(output, label)
                        loss_micro = loss / self.accum_step
                        self.scaler.scale(loss_micro).backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    output = self.model(image)
                    loss = self.criterion(output, label)
                    loss_micro = loss / self.accum_step
                    loss_micro.backward()
                    if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
                        self.optim.step()
                        self.optim.zero_grad()
                
                # with torch.no_grad():
                #     pred = torch.sigmoid(output)
                #     acc = average_precision_score(label.cpu().detach(), torch.sigmoid(output).cpu().detach())
                     
                # with torch.no_grad():
                #     pred = output.argmax(dim=1)
                #     correct = pred.eq(label).float()
                #     acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                # acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                # for _c, _y in zip(correct, label):
                #     cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
                # cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

                # mean_acc = np.mean(np.array(cls_accs))
                # many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
                # med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
                # few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    # info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    # info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                # self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                # self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                # self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
                # self._writer.add_scalar("train/many_acc", many_acc, n_iter)
                # self._writer.add_scalar("train/med_acc", med_acc, n_iter)
                # self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
                end = time.time()
            ########################################### batch end ############################################################ 
            self.sched.step()
            torch.cuda.empty_cache()
            
            result = self.test()
            print("mAP:{}".format(result))

        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        self.save_model(cfg.output_dir)

        result = self.test()
        print("mAP:{}".format(result))

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            # _bsz, _ncrops, _c, _h, _w = image.size()
            # image = image.view(_bsz * _ncrops, _c, _h, _w)

            output = self.model(image, return_feature=True, gcn_relation=self.relation)
            # output = output.view(_bsz, _ncrops, -1).mean(dim=1)

            label, pred = self.evaluator.process(output, label)

        # results = self.evaluator.evaluate()
        results = self.mAP(label, pred)
        # for k, v in results.items():
        #     tag = f"test/{k}"
        #     if self._writer is not None:
        #         self._writer.add_scalar(tag, v)
        
        # return list(results.values())[0]
        self.mAP_hmt(label, pred)
        return results
    
    def average_precision(self, output, target):
        epsilon = 1e-8
        indices = np.array(output).argsort()[::-1]
        # sort examples
        # indices = output.argsort()[::-1]
        # Computes prec@i
        total_count_ = np.cumsum(np.ones((len(output), 1)))

        target_ = np.array(target)[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0  # type: ignore
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)
        precision_at_i = precision_at_i_ / (total + epsilon)

        return precision_at_i
    
    def mAP(self, targs, preds):
        if np.size(preds) == 0:
            return 0
        ap = np.zeros(self.num_classes)
        # compute average precision for each class
        for k in range(self.num_classes):
            # sort scores
            scores = preds[:][k]
            targets = targs[:][k]
            # compute average precision
            ap[k] = self.average_precision(scores, targets)
            ap[k] = average_precision_score(targets, scores)
        return 100 * ap.mean()

    def mAP_hmt(self, targs, preds):
        import mmcv
        class_freq = np.asarray(mmcv.load('/home/wzz/LMPT/data/voc/class_freq.pkl')['class_freq'])
        hids = list(np.where(class_freq>=100)[0])
        mids = list(np.where((class_freq<100) * (class_freq >= 20))[0])
        tids = list(np.where(class_freq<20)[0])
        
        if np.size(preds) == 0:
            return 0
        ap = np.zeros(self.num_classes)
        # compute average precision for each class
        for k in range(self.num_classes):
            # sort scores
            scores = preds[:][k]
            targets = targs[:][k]
            # compute average precision
            ap[k] = self.average_precision(scores, targets)
        print(f'mAP head: {ap[hids].mean()}, mAP medium: {ap[mids].mean()}, mAP tail: {ap[tids].mean()}')
        
    def save_model(self, directory):
        tuner_dict = self.tuner.state_dict()
        head_dict = self.head.state_dict()
        checkpoint = {
            "tuner": tuner_dict,
            "head": head_dict
        }

        # remove 'module.' in state_dict's keys
        for key in ["tuner", "head"]:
            state_dict = checkpoint[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict

        # save model
        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        tuner_dict = checkpoint["tuner"]
        head_dict = checkpoint["head"]

        print("Loading weights to from {}".format(load_path))
        self.tuner.load_state_dict(tuner_dict)
        self.head.load_state_dict(head_dict)

# class Trainer:
#     def __init__(self, cfg):

#         if not torch.cuda.is_available():
#             self.device = torch.device("cpu")
#         elif cfg.gpu is None:
#             self.device = torch.device("cuda")
#         else:
#             torch.cuda.set_device(cfg.gpu)
#             self.device = torch.device("cuda:{}".format(cfg.gpu))

#         self.cfg = cfg
#         self.build_data_loader()
#         self.build_model()
#         self.evaluator = Evaluator(cfg, self.many_idxs, self.med_idxs, self.few_idxs)
#         self._writer = None

#     def build_data_loader(self):
#         cfg = self.cfg
#         root = cfg.root
#         resolution = cfg.resolution
#         expand = cfg.expand

#         if cfg.backbone.startswith("CLIP"):
#             mean = [0.48145466, 0.4578275, 0.40821073]
#             std = [0.26862954, 0.26130258, 0.27577711]
#         else:
#             mean = [0.5, 0.5, 0.5]
#             std = [0.5, 0.5, 0.5]
#         print("mean:", mean)
#         print("std:", std)

#         transform_train = transforms.Compose([
#             transforms.RandomResizedCrop(resolution),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ])

#         transform_plain = transforms.Compose([
#             transforms.Resize(resolution),
#             transforms.CenterCrop(resolution),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ])

#         if cfg.test_ensemble:
#             transform_test = transforms.Compose([
#                 transforms.Resize(resolution + expand),
#                 transforms.FiveCrop(resolution),
#                 transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#                 transforms.Normalize(mean, std),
#             ])
#         else:
#             transform_test = transforms.Compose([
#                 transforms.Resize(resolution * 8 // 7),
#                 transforms.CenterCrop(resolution),
#                 transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
#                 transforms.Normalize(mean, std),
#             ])

#         train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
#         train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
#         train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
#         test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

#         self.num_classes = train_dataset.num_classes
#         self.cls_num_list = train_dataset.cls_num_list
#         self.classnames = train_dataset.classnames

#         if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
#             split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
#         else:
#             split_cls_num_list = self.cls_num_list
#         self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
#         self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
#         self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

#         if cfg.init_head == "1_shot":
#             init_sampler = DownSampler(train_init_dataset, n_max=1)
#         elif cfg.init_head == "10_shot":
#             init_sampler = DownSampler(train_init_dataset, n_max=10)
#         elif cfg.init_head == "100_shot":
#             init_sampler = DownSampler(train_init_dataset, n_max=100)
#         else:
#             init_sampler = None

#         self.train_loader = DataLoader(train_dataset,
#             batch_size=cfg.micro_batch_size, shuffle=True,
#             num_workers=cfg.num_workers, pin_memory=True)

#         self.train_init_loader = DataLoader(train_init_dataset,
#             batch_size=64, sampler=init_sampler, shuffle=False,
#             num_workers=cfg.num_workers, pin_memory=True)

#         self.train_test_loader = DataLoader(train_test_dataset,
#             batch_size=64, shuffle=False,
#             num_workers=cfg.num_workers, pin_memory=True)

#         self.test_loader = DataLoader(test_dataset,
#             batch_size=64, shuffle=False,
#             num_workers=cfg.num_workers, pin_memory=True)
        
#         assert cfg.batch_size % cfg.micro_batch_size == 0
#         self.accum_step = cfg.batch_size // cfg.micro_batch_size

#         print("Total training points:", sum(self.cls_num_list))
#         # print(self.cls_num_list)

#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.classnames
#         num_classes = len(classnames)

#         print("Building model")
#         if cfg.zero_shot:
#             assert cfg.backbone.startswith("CLIP")
#             print(f"Loading CLIP (backbone: {cfg.backbone})")
#             clip_model = load_clip_to_cpu(cfg)
#             self.model = ZeroShotCLIP(clip_model)
#             self.model.to(self.device)
#             self.tuner = None
#             self.head = None

#             prompts = self.get_tokenized_prompts(classnames)
#             self.model.init_text_features(prompts)

#         elif cfg.backbone.startswith("CLIP"):
#             print(f"Loading CLIP (backbone: {cfg.backbone})")
#             clip_model = load_clip_to_cpu(cfg)
#             self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
#             self.model.to(self.device)
#             self.tuner = self.model.tuner
#             self.head = self.model.head

#         elif cfg.backbone.startswith("IN21K-ViT"):
#             print(f"Loading ViT (backbone: {cfg.backbone})")
#             vit_model = load_vit_to_cpu(cfg)
#             self.model = PeftModelFromViT(cfg, vit_model, num_classes)
#             self.model.to(self.device)
#             self.tuner = self.model.tuner
#             self.head = self.model.head

#         if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
#             self.build_optimizer()
#             self.build_criterion()

#             if cfg.init_head == "text_feat":
#                 self.init_head_text_feat()
#             elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
#                 self.init_head_class_mean()
#             elif cfg.init_head == "linear_probe":
#                 self.init_head_linear_probe()
#             else:
#                 print("No initialization with head")
            
#             torch.cuda.empty_cache()
        
#         # Note that multi-gpu training could be slow because CLIP's size is
#         # big, which slows down the copy operation in DataParallel
#         device_count = torch.cuda.device_count()
#         if device_count > 1 and cfg.gpu is None:
#             print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#             self.model = nn.DataParallel(self.model)

#     def build_optimizer(self):
#         cfg = self.cfg

#         print("Turning off gradients in the model")
#         for name, param in self.model.named_parameters():
#             param.requires_grad_(False)
#         print("Turning on gradients in the tuner")
#         for name, param in self.tuner.named_parameters():
#             param.requires_grad_(True)
#         print("Turning on gradients in the head")
#         for name, param in self.head.named_parameters():
#             param.requires_grad_(True)

#         # print parameters
#         total_params = sum(p.numel() for p in self.model.parameters())
#         tuned_params = sum(p.numel() for p in self.tuner.parameters())
#         head_params = sum(p.numel() for p in self.head.parameters())
#         print(f"Total params: {total_params}")
#         print(f"Tuned params: {tuned_params}")
#         print(f"Head params: {head_params}")
#         # for name, param in self.tuner.named_parameters():
#         #     print(name, param.numel())

#         # NOTE: only give tuner and head to the optimizer
#         self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
#                                       {"params": self.head.parameters()}],
#                                       lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
#         self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
#         self.scaler = GradScaler() if cfg.prec == "amp" else None

#     def build_criterion(self):
#         cfg = self.cfg
#         cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

#         if cfg.loss_type == "CE":
#             self.criterion = nn.CrossEntropyLoss()
#         elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
#             self.criterion = FocalLoss()
#         elif cfg.loss_type == "LDAM": # https://arxiv.org/abs/1906.07413
#             self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
#         elif cfg.loss_type == "CB": # https://arxiv.org/abs/1901.05555
#             self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
#         elif cfg.loss_type == "GRW": # https://arxiv.org/abs/2103.16370
#             self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
#         elif cfg.loss_type == "BS": # https://arxiv.org/abs/2007.10740
#             self.criterion == BalancedSoftmaxLoss(cls_num_list=cls_num_list)
#         elif cfg.loss_type == "LA": # https://arxiv.org/abs/2007.07314
#             self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
#         elif cfg.loss_type == "LADE": # https://arxiv.org/abs/2012.00321
#             self.criterion = LADELoss(cls_num_list=cls_num_list)
        
#     def get_tokenized_prompts(self, classnames):
#         template = "a photo of a {}."
#         prompts = [template.format(c.replace("_", " ")) for c in classnames]
#         # print(f"Prompts: {prompts}")
#         prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         prompts = prompts.to(self.device)
#         return prompts

#     @torch.no_grad()
#     def init_head_text_feat(self):
#         cfg = self.cfg
#         classnames = self.classnames

#         print("Initialize head with text features")
#         prompts = self.get_tokenized_prompts(classnames)
#         text_features = self.model.encode_text(prompts)
#         text_features = F.normalize(text_features, dim=-1)

#         if cfg.backbone.startswith("CLIP-ViT"):
#             text_features = text_features @ self.model.image_encoder.proj.t()
#             text_features = F.normalize(text_features, dim=-1)

#         self.head.apply_weight(text_features)

#     @torch.no_grad()
#     def init_head_class_mean(self):
#         print("Initialize head with class means")
#         all_features = []
#         all_labels = []

#         for batch in tqdm(self.train_init_loader, ascii=True):
#             image = batch[0]
#             label = batch[1]

#             image = image.to(self.device)
#             label = label.to(self.device)

#             feature = self.model(image, use_tuner=False, return_feature=True)

#             all_features.append(feature)
#             all_labels.append(label)

#         all_features = torch.cat(all_features, dim=0)
#         all_labels = torch.cat(all_labels, dim=0)

#         sorted_index = all_labels.argsort()
#         all_features = all_features[sorted_index]
#         all_labels = all_labels[sorted_index]

#         unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

#         class_means = [None] * self.num_classes
#         idx = 0
#         for i, cnt in zip(unique_labels, label_counts):
#             class_means[i] = all_features[idx: idx+cnt].mean(dim=0, keepdim=True)
#             idx += cnt
#         class_means = torch.cat(class_means, dim=0)
#         class_means = F.normalize(class_means, dim=-1)

#         self.head.apply_weight(class_means)

#     @torch.no_grad()
#     def init_head_linear_probe(self):
#         print("Initialize head with linear probing")
#         all_features = []
#         all_labels = []

#         for batch in tqdm(self.train_init_loader, ascii=True):
#             image = batch[0]
#             label = batch[1]

#             image = image.to(self.device)
#             label = label.to(self.device)

#             feature = self.model(image, use_tuner=False, return_feature=True)

#             all_features.append(feature)
#             all_labels.append(label)

#         all_features = torch.cat(all_features, dim=0).cpu()
#         all_labels = torch.cat(all_labels, dim=0).cpu()

#         clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
#         class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
#         class_weights = F.normalize(class_weights, dim=-1)

#         self.head.apply_weight(class_weights)

#     def train(self):
#         cfg = self.cfg

#         # Initialize summary writer
#         writer_dir = os.path.join(cfg.output_dir, "tensorboard")
#         os.makedirs(writer_dir, exist_ok=True)
#         print(f"Initialize tensorboard (log_dir={writer_dir})")
#         self._writer = SummaryWriter(log_dir=writer_dir)

#         # Initialize average meters
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         loss_meter = AverageMeter(ema=True)
#         acc_meter = AverageMeter(ema=True)
#         cls_meters = [AverageMeter(ema=True) for _ in range(self.num_classes)]

#         # Remember the starting time (for computing the elapsed time)
#         time_start = time.time()

#         num_epochs = cfg.num_epochs
#         for epoch_idx in range(num_epochs):
#             self.tuner.train()
#             end = time.time()

#             num_batches = len(self.train_loader)
#             for batch_idx, batch in enumerate(self.train_loader):
#                 data_time.update(time.time() - end)

#                 image = batch[0]
#                 label = batch[1]
#                 image = image.to(self.device)
#                 label = label.to(self.device)

#                 if cfg.prec == "amp":
#                     with autocast():
#                         output = self.model(image)
#                         loss = self.criterion(output, label)
#                         loss_micro = loss / self.accum_step
#                         self.scaler.scale(loss_micro).backward()
#                     if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
#                         self.scaler.step(self.optim)
#                         self.scaler.update()
#                         self.optim.zero_grad()
#                 else:
#                     output = self.model(image)
#                     loss = self.criterion(output, label)
#                     loss_micro = loss / self.accum_step
#                     loss_micro.backward()
#                     if ((batch_idx + 1) % self.accum_step == 0) or (batch_idx + 1 == num_batches):
#                         self.optim.step()
#                         self.optim.zero_grad()

#                 with torch.no_grad():
#                     pred = output.argmax(dim=1)
#                     correct = pred.eq(label).float()
#                     acc = correct.mean().mul_(100.0)

#                 current_lr = self.optim.param_groups[0]["lr"]
#                 loss_meter.update(loss.item())
#                 acc_meter.update(acc.item())
#                 batch_time.update(time.time() - end)

#                 for _c, _y in zip(correct, label):
#                     cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
#                 cls_accs = [cls_meters[i].avg for i in range(self.num_classes)]

#                 mean_acc = np.mean(np.array(cls_accs))
#                 many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
#                 med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
#                 few_acc = np.mean(np.array(cls_accs)[self.few_idxs])

#                 meet_freq = (batch_idx + 1) % cfg.print_freq == 0
#                 only_few_batches = num_batches < cfg.print_freq
#                 if meet_freq or only_few_batches:
#                     nb_remain = 0
#                     nb_remain += num_batches - batch_idx - 1
#                     nb_remain += (
#                         num_epochs - epoch_idx - 1
#                     ) * num_batches
#                     eta_seconds = batch_time.avg * nb_remain
#                     eta = str(datetime.timedelta(seconds=int(eta_seconds)))

#                     info = []
#                     info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
#                     info += [f"batch [{batch_idx + 1}/{num_batches}]"]
#                     info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
#                     info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
#                     info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
#                     info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
#                     info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
#                     info += [f"lr {current_lr:.4e}"]
#                     info += [f"eta {eta}"]
#                     print(" ".join(info))

#                 n_iter = epoch_idx * num_batches + batch_idx
#                 self._writer.add_scalar("train/lr", current_lr, n_iter)
#                 self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
#                 self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
#                 self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
#                 self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
#                 self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
#                 self._writer.add_scalar("train/many_acc", many_acc, n_iter)
#                 self._writer.add_scalar("train/med_acc", med_acc, n_iter)
#                 self._writer.add_scalar("train/few_acc", few_acc, n_iter)
                
#                 end = time.time()

#             self.sched.step()
#             torch.cuda.empty_cache()
#             self.test()

#         print("Finish training")
#         print("Note that the printed training acc is not precise.",
#               "To get precise training acc, use option ``test_train True``.")

#         # show elapsed time
#         elapsed = round(time.time() - time_start)
#         elapsed = str(datetime.timedelta(seconds=elapsed))
#         print(f"Time elapsed: {elapsed}")

#         # save model
#         self.save_model(cfg.output_dir)

#         self.test()

#         # Close writer
#         self._writer.close()

#     @torch.no_grad()
#     def test(self, mode="test"):
#         if self.tuner is not None:
#             self.tuner.eval()
#         if self.head is not None:
#             self.head.eval()
#         self.evaluator.reset()

#         if mode == "train":
#             print(f"Evaluate on the train set")
#             data_loader = self.train_test_loader
#         elif mode == "test":
#             print(f"Evaluate on the test set")
#             data_loader = self.test_loader

#         for batch in tqdm(data_loader, ascii=True):
#             image = batch[0]
#             label = batch[1]

#             image = image.to(self.device)
#             label = label.to(self.device)

#             _bsz, _ncrops, _c, _h, _w = image.size()
#             image = image.view(_bsz * _ncrops, _c, _h, _w)

#             output = self.model(image)
#             output = output.view(_bsz, _ncrops, -1).mean(dim=1)

#             self.evaluator.process(output, label)

#         results = self.evaluator.evaluate()

#         for k, v in results.items():
#             tag = f"test/{k}"
#             if self._writer is not None:
#                 self._writer.add_scalar(tag, v)

#         return list(results.values())[0]

#     def save_model(self, directory):
#         tuner_dict = self.tuner.state_dict()
#         head_dict = self.head.state_dict()
#         checkpoint = {
#             "tuner": tuner_dict,
#             "head": head_dict
#         }

#         # remove 'module.' in state_dict's keys
#         for key in ["tuner", "head"]:
#             state_dict = checkpoint[key]
#             new_state_dict = OrderedDict()
#             for k, v in state_dict.items():
#                 if k.startswith("module."):
#                     k = k[7:]
#                 new_state_dict[k] = v
#             checkpoint[key] = new_state_dict

#         # save model
#         save_path = os.path.join(directory, "checkpoint.pth.tar")
#         torch.save(checkpoint, save_path)
#         print(f"Checkpoint saved to {save_path}")

#     def load_model(self, directory):
#         load_path = os.path.join(directory, "checkpoint.pth.tar")

#         if not os.path.exists(load_path):
#             raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

#         checkpoint = torch.load(load_path, map_location=self.device)
#         tuner_dict = checkpoint["tuner"]
#         head_dict = checkpoint["head"]

#         print("Loading weights to from {}".format(load_path))
#         self.tuner.load_state_dict(tuner_dict)
#         self.head.load_state_dict(head_dict)
        