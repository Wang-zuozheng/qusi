import torch
import torch.nn as nn
from torch.nn import functional as F

from .clip_text import *
from .peft_vit import Peft_ViT, ViT_Tuner
from .peft_rn import Peft_RN, RN_Tuner
from .classifiers import *
from .gcn import *

class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    @torch.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        return logit


class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes, classnames):
        super().__init__()

        if cfg.backbone.startswith("CLIP-ViT"):
            self.text_encoder = CLIP_Text(clip_model)
            self.prompt_text_encoder = TextEncoder(clip_model)
            self.image_encoder = Peft_ViT(clip_model.visual)
            self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)
            
        elif cfg.backbone.startswith("CLIP-RN"):
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_RN(clip_model.visual)
            self.tuner = RN_Tuner(cfg, clip_model.visual, num_classes)
        
        if cfg.is_gcn:
            self.gcn = GCN_feature(cfg.in_features, cfg.mid_features)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = clip_model.logit_scale
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features
    
    def forward(self, image, use_tuner=True, return_feature=False, gcn_relation=None):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        image_features = self.image_encoder(image, tuner, head)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.prompt_text_encoder(prompts, tokenized_prompts)
        if self.gcn is not None:
            identity = text_features
            text_features = self.gcn(text_features, gcn_relation)
        text_features += identity
        logit_scale = self.logit_scale.exp()
        image_features= image_features / image_features.norm(dim=-1,
                                                                keepdim=True)    
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        
        return logits


class PeftModelFromViT(nn.Module):
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()

        if cfg.backbone.startswith("IN21K-ViT"):
            self.image_encoder = Peft_ViT(vit_model)
            self.tuner = ViT_Tuner(cfg, vit_model, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head = eval(cfg.classifier)(feat_dim, num_classes, dtype, **cfg)

    def forward(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head)
