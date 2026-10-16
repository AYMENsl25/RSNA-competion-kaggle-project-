"""Shared 2D encoder MIL, frozen SSL encoder, and small MONAI 3D comparator."""
from pathlib import Path
import torch
from torch import nn
from .common import sha256


class StudyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight_hash = None
        family = cfg['model']
        if family == 'tiny':
            self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
                                         nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
            dim = 32
        elif family == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.encoder = resnet18(weights=None)
            if cfg['pretrained']:
                if cfg.get('weights_path'):
                    self.encoder.load_state_dict(torch.load(cfg['weights_path'], map_location='cpu', weights_only=True))
                    self.weight_hash = sha256(cfg['weights_path'])
                elif cfg.get('allow_downloads'):
                    weights = ResNet18_Weights.IMAGENET1K_V1
                    self.encoder.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
                    self.weight_hash = sha256(Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/', 1)[-1])
                else:
                    raise FileNotFoundError('Attach resnet18_imagenet.pth and set weights_path, or enable Internet and allow_downloads.')
            dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif family == 'dinov2':
            import timm
            self.encoder = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224)
            if not cfg.get('weights_path'):
                raise FileNotFoundError('DINO requires dinov2_vits14_timm.pth from the notebook weight-preparation cell.')
            self.encoder.load_state_dict(torch.load(cfg['weights_path'], map_location='cpu', weights_only=True))
            self.weight_hash = sha256(cfg['weights_path'])
            dim = self.encoder.num_features
        elif family == 'monai3d':
            from monai.networks.nets import DenseNet
            self.encoder = DenseNet(spatial_dims=3, in_channels=1, out_channels=128,
                                    init_features=16, growth_rate=8, block_config=(2, 2, 2), bn_size=2)
            dim = 128
        else:
            raise ValueError(f'Unknown model: {family}')
        # Plane identity is retained rather than treating sagittal/coronal/axial interchangeably.
        self.plane_embedding = nn.Embedding(3, dim)
        self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 12))
        self.dropout = nn.Dropout(cfg['dropout'])
        self.head = nn.Linear(dim * 2 if cfg['pooling'] == 'meanmax' else dim, 12)
        self.encoder_frozen = False

    def set_epoch(self, epoch):
        frozen = self.cfg['strategy'] == 'frozen' or (self.cfg['strategy'] == 'staged' and epoch < 2)
        self.encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        # Keep BatchNorm fixed with tiny study batches. Other encoder modules can train.
        if frozen:
            self.encoder.eval()
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def forward(self, images, mask):
        # images B,planes,slices,H,W; mask B,planes,slices
        b, planes, depth, h, w = images.shape
        if not mask.flatten(1).any(1).all():
            raise ValueError('Study has no valid MRI slice')
        if self.cfg['model'] == 'monai3d':
            valid = mask.any(-1).reshape(-1)
            x = images.reshape(b*planes, 1, depth, h, w)
            x = x * mask.reshape(b*planes, 1, depth, 1, 1)
            features = self.encoder(x[valid])
            all_features = features.new_zeros((b*planes, features.shape[-1]))
            all_features[valid] = features
            features = all_features.reshape(b, planes, -1)
            token_mask = mask.any(-1)
            plane_ids = torch.arange(planes, device=images.device)
        else:
            valid = mask.reshape(-1)
            x = images.reshape(b*planes*depth, 1, h, w)[valid].repeat(1, 3, 1, 1)
            if self.cfg['model'] == 'dinov2':
                x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            if self.cfg['pretrained']:
                mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
                std = x.new_tensor([.229, .224, .225])[None, :, None, None]
                x = (x-mean)/std
            # Frozen encoders use no autograd; no claim of precomputed feature caching.
            with torch.set_grad_enabled(self.training and not self.encoder_frozen):
                encoded = self.encoder(x)
            all_features = encoded.new_zeros((b*planes*depth, encoded.shape[-1]))
            all_features[valid] = encoded
            features = all_features.reshape(b, planes*depth, -1)
            token_mask = mask.reshape(b, planes*depth)
            plane_ids = torch.arange(planes, device=images.device).repeat_interleave(depth)
        features = features + self.plane_embedding(plane_ids)[None]
        if self.cfg['pooling'] == 'meanmax':
            mean = (features*token_mask[..., None]).sum(1)/token_mask.sum(1, keepdim=True)
            maximum = features.masked_fill(~token_mask[..., None], -torch.inf).amax(1)
            return self.head(self.dropout(torch.cat([mean, maximum], -1)))
        scores = self.attention(features).masked_fill(~token_mask[..., None], -torch.inf)
        pooled = torch.einsum('bnt,bnf->btf', scores.softmax(1), features)
        return (self.dropout(pooled)*self.head.weight[None]).sum(-1)+self.head.bias
