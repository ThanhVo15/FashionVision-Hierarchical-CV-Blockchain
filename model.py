import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EndToEndHierarchicalModel(nn.Module):
    def __init__(
        self,
        # Số lớp cho từng task:
        num_gender, num_master, num_usage,
        num_sub, num_article, num_base,
        num_brand,
        pretrained=True,
        mlp_hidden=32,
        embed_dim=16,
        teacher_forcing_p=0.5
    ):
        super().__init__()
        self.teacher_forcing_p = teacher_forcing_p

        # 1) Backbone CNN (ResNet18)
        self.backbone = models.resnet18(pretrained=pretrained)
        backbone_out = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 2) MLP numeric (3 input: [season, year, price])
        self.mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),    # Gợi ý: thêm dropout
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),  # Gợi ý: thêm batchnorm
            nn.ReLU()
        )

        self.base_feat_dim = backbone_out + mlp_hidden

        # ---------- Stage 1 ----------
        self.head_gender = nn.Linear(self.base_feat_dim, num_gender)
        self.head_master = nn.Linear(self.base_feat_dim, num_master)
        self.head_usage  = nn.Linear(self.base_feat_dim, num_usage)

        self.embed_gender = nn.Embedding(num_gender, embed_dim)
        self.embed_master = nn.Embedding(num_master, embed_dim)
        self.embed_usage  = nn.Embedding(num_usage,  embed_dim)

        # ---------- Stage 2 ----------
        stage1_embed_total = 3 * embed_dim
        stage2_input_dim = self.base_feat_dim + stage1_embed_total

        self.head_sub     = nn.Linear(stage2_input_dim, num_sub)
        self.head_article = nn.Linear(stage2_input_dim, num_article)
        self.head_base    = nn.Linear(stage2_input_dim, num_base)

        self.embed_sub     = nn.Embedding(num_sub,     embed_dim)
        self.embed_article = nn.Embedding(num_article, embed_dim)
        self.embed_base    = nn.Embedding(num_base,    embed_dim)

        # ---------- Stage 3 ----------
        stage2_embed_total = 3 * embed_dim
        stage3_input_dim   = self.base_feat_dim + stage1_embed_total + stage2_embed_total
        self.head_brand = nn.Linear(stage3_input_dim, num_brand)

    def forward(self, images, numeric_data=None, labels=None):
        """
        images: [B, 3, 224, 224]
        numeric_data: [B, 3] hoặc None
        labels: dict (hoặc None)
        """
        batch_size = images.size(0)

        # 1) CNN
        cnn_feat = self.backbone(images)

        # 2) Xử lý numeric_data = None
        if numeric_data is None:
            # Nếu user không cung cấp => ta tạo tensor zero
            numeric_data = torch.zeros(batch_size, 3, device=images.device)

        mlp_feat = self.mlp(numeric_data)
        base_feat = torch.cat([cnn_feat, mlp_feat], dim=1)  # [B, base_feat_dim]

        # =================== STAGE 1 ===================
        out_gender = self.head_gender(base_feat)
        out_master = self.head_master(base_feat)
        out_usage  = self.head_usage(base_feat)

        pred_gender_id = torch.argmax(out_gender, dim=1)
        pred_master_id = torch.argmax(out_master, dim=1)
        pred_usage_id  = torch.argmax(out_usage,  dim=1)

        # Mixed Teacher Forcing Stage 1
        if (labels is not None) and self.training:
            gt_gender_id = labels["gender"]
            gt_master_id = labels["masterCategory"]
            gt_usage_id  = labels["usage"]

            tf_mask = torch.rand(batch_size, device=images.device) < self.teacher_forcing_p

            used_gender_id = torch.where(tf_mask, gt_gender_id, pred_gender_id)
            used_master_id = torch.where(tf_mask, gt_master_id, pred_master_id)
            used_usage_id  = torch.where(tf_mask, gt_usage_id,  pred_usage_id)
        else:
            used_gender_id = pred_gender_id
            used_master_id = pred_master_id
            used_usage_id  = pred_usage_id

        embed_g = self.embed_gender(used_gender_id)
        embed_m = self.embed_master(used_master_id)
        embed_u = self.embed_usage(used_usage_id)
        stage1_embed = torch.cat([embed_g, embed_m, embed_u], dim=1)

        # =================== STAGE 2 ===================
        stage2_input = torch.cat([base_feat, stage1_embed], dim=1)
        out_sub     = self.head_sub(stage2_input)
        out_article = self.head_article(stage2_input)
        out_base    = self.head_base(stage2_input)

        pred_sub_id     = torch.argmax(out_sub,     dim=1)
        pred_article_id = torch.argmax(out_article, dim=1)
        pred_base_id    = torch.argmax(out_base,    dim=1)

        if (labels is not None) and self.training:
            gt_sub_id     = labels["subCategory"]
            gt_article_id = labels["articleType"]
            gt_base_id    = labels["baseColour"]

            tf_mask2 = torch.rand(batch_size, device=images.device) < self.teacher_forcing_p

            used_sub_id     = torch.where(tf_mask2, gt_sub_id,     pred_sub_id)
            used_article_id = torch.where(tf_mask2, gt_article_id, pred_article_id)
            used_base_id    = torch.where(tf_mask2, gt_base_id,    pred_base_id)
        else:
            used_sub_id     = pred_sub_id
            used_article_id = pred_article_id
            used_base_id    = pred_base_id

        embed_s = self.embed_sub(used_sub_id)
        embed_a = self.embed_article(used_article_id)
        embed_b = self.embed_base(used_base_id)
        stage2_embed = torch.cat([embed_s, embed_a, embed_b], dim=1)

        # =================== STAGE 3 ===================
        stage3_input = torch.cat([base_feat, stage1_embed, stage2_embed], dim=1)
        out_brand = self.head_brand(stage3_input)

        outputs = {
            "gender":         out_gender,
            "masterCategory": out_master,
            "usage":          out_usage,
            "subCategory":    out_sub,
            "articleType":    out_article,
            "baseColour":     out_base,
            "brand":          out_brand
        }
        return outputs
