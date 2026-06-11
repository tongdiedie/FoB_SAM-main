import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Res101Encoder
import sys
import cv2
import numpy as np
import torchvision.transforms as transforms
import math
from info_nce import InfoNCE


class IDR(nn.Module):
    def __init__(self, in_dim, num_points=10):
        super().__init__()
        self.num_points = num_points
        self.offset_pred = nn.Linear(in_dim * 2, num_points * 2)  
        self.scale_mod = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )
        self.attn_weight = nn.Linear(in_dim, num_points)

    def forward(self, query_feats, query_points, support_feats, feat_map, gcn_out):
        B, K, C = query_feats.shape
        H, W = feat_map.shape[2:]

        offset_input = torch.cat([query_feats, gcn_out], dim=-1)  
        offset = self.offset_pred(offset_input).view(B, K, self.num_points, 2)

        scale_input = torch.cat([query_feats, support_feats], dim=-1)
        scale = 2 * self.scale_mod(scale_input).view(B, K, 1, 1)

        coords = query_points.unsqueeze(2) + scale * offset            
        coords_grid = coords.view(B, 1, K * self.num_points, 2)
        grid = coords_grid * 2 - 1                                     

        feat_sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)
        feat_sampled = feat_sampled.view(B, feat_map.size(1), K, self.num_points)

        weight = self.attn_weight(query_feats)                         
        weight = F.softmax(weight, dim=-1)

        feat_weighted = (feat_sampled * weight.unsqueeze(1)).sum(dim=-1)  
        feat_weighted = feat_weighted.transpose(1, 2)                     

        new_qp = (coords * weight.unsqueeze(-1)).sum(dim=2)         

        return feat_weighted, new_qp


class SPG(nn.Module):
    def __init__(self, in_dim, use_learnable_alpha=True):
        super().__init__()
        self.W_theta = nn.Linear(in_dim, in_dim)
        self.W_phi = nn.Linear(in_dim, in_dim)
        self.W = nn.Linear(in_dim, in_dim, bias=False)
        self.mlp_mod = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )
        self.use_learnable_alpha = use_learnable_alpha
        if use_learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def build_ring_adj(self, K, B, device):
        A = torch.zeros(K, K, device=device)
        for i in range(K):
            A[i, (i - 1) % K] = 1
            A[i, (i + 1) % K] = 1
        A = A.unsqueeze(0).expand(B, -1, -1)
        return A

    def forward(self, query_feats, support_feats):
        B, K, C = query_feats.shape
        theta = self.W_theta(support_feats)
        phi = self.W_phi(support_feats)
        A_dyn = torch.matmul(theta, phi.transpose(-1, -2)) / (C ** 0.5)
        A_dyn = F.softmax(A_dyn, dim=-1)

        A_ring = self.build_ring_adj(K, B, query_feats.device)
        alpha = torch.clamp(self.alpha, 0, 1) if self.use_learnable_alpha else 0.5
        A = alpha * A_dyn + (1 - alpha) * A_ring
        A = A / A.sum(dim=-1, keepdim=True)

        M = torch.sigmoid(self.mlp_mod(query_feats))
        WQ = self.W(query_feats)
        out = torch.bmm(A, M * WQ)
        return F.relu(out)


class SPR(nn.Module):
    def __init__(self, in_dim, num_heads=4, num_points=10):
        super().__init__()
        self.gcn = SPG(in_dim)
        self.self_attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True)
        self.deform_attn = IDR(in_dim, num_points)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.iter = 3

    def forward(self, query_feats, query_points, support_feats, feat_map):
        """
        query_feats: [B, K, C]
        support_feats: [B, K, C]
        query_points: [B, K, 2]
        feat_map: [B, C, H, W]
        """
        gcn_out = self.gcn(query_feats, support_feats)
        query_feats = self.norm1(query_feats + gcn_out)

        attn_out, _ = self.self_attn(query_feats, query_feats, query_feats)
        query_feats = self.norm2(query_feats + attn_out)

        for _ in range(self.iter):
            visual_feat, query_points = self.deform_attn(
                query_feats, query_points, support_feats, feat_map, gcn_out
            )
            query_feats = self.norm3(query_feats + visual_feat)

        return query_points



class Head(nn.Module):
    def __init__(self, in_channels, out_size):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0 ),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=10,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)
    def forward(self, x):
        heatmap = self.head(x)  
        heatmap = self.upsample(heatmap)  
        return heatmap
    
    

class PromptMatching(nn.Module):

    def __init__(self, hidden_dim, proj_dim, self_update_dim):
        super().__init__()
        self.support_proj = nn.Linear(hidden_dim, proj_dim)
        self.query_proj = nn.Linear(hidden_dim, proj_dim)
        self.self_update_proj = nn.Sequential(
            nn.Linear(hidden_dim, self_update_dim), 
            nn.ReLU(),
            nn.Linear(self_update_dim, hidden_dim))
        self.tanh = nn.Tanh()
        self.dim = hidden_dim
        self.model_init()

    def forward(self, query, support, spatial_shape):
        """
        Args:
            support_subgraph: [n, bs, c]
            query_graph: [hw, bs, c]
            spatial_shape: h, w
        """
        h, w = spatial_shape
        query = query.transpose(0, 1)  
        support = support.transpose(0, 1) 

        fs_proj = self.support_proj(support)  
        fq_proj = self.query_proj(query)  
        channel_reweight = self.tanh(self.self_update_proj(fs_proj))  

        fs_feat = (channel_reweight + 1) * fs_proj  
        Phi = torch.bmm(fq_proj, fs_feat.transpose(1, 2)) 
        Phi = Phi.transpose(1, 2).reshape(-1, h, w)  
        return Phi
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True


class MaskedAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, ffn_expansion=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * ffn_expansion),
            nn.ReLU(),
            nn.Linear(feature_dim * ffn_expansion, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

        self.matching_head = PromptMatching(feature_dim, feature_dim, feature_dim * 2)
        self.conv = nn.Conv2d(10, 1, kernel_size=1, stride=1, padding=0)
        self.learnable_pos = nn.Parameter(torch.randn(1, feature_dim, 64, 64))
        self.sin_pos = self.get_sinusoid_encoding_table(10, feature_dim).cuda()  

    def get_sinusoid_encoding_table(self, K, C):
        position = torch.arange(K).unsqueeze(1)                 # [K, 1]
        div_term = torch.exp(torch.arange(0, C, 2) * (-math.log(10000.0) / C))  # [C/2]

        pe = torch.zeros(K, C)
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        return pe  
    
    def forward(self, skps, qry_fts):
        """
        skps: [N, C]
        qry_fts: [1, C, H, W]
        """
        N, C = skps.shape
        B, _, H, W = qry_fts.shape
        L = H * W
        qry_fts = qry_fts + self.learnable_pos  
        skps = skps + self.sin_pos
        sim = self.matching_head(qry_fts.view(B, C, L).permute(2, 0, 1), skps.unsqueeze(1), (H, W))  # [B, L, N]
        
        mask = F.relu(self.conv(sim))  # [1, L, 1]
        x = qry_fts.view(B, C, L).permute(0, 2, 1)  # [B, L, C]
        mask_flat = mask.view(1, L)
    
        attn_bias = mask_flat.transpose(1, 0) + mask_flat  # [L, L]

        attn_out, attn = self.mha(x, x, x, attn_mask=attn_bias)  # [B, L, C]
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out, sim

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class FewShotSeg(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args is None:
            args = {}


        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                    pretrained_weights="COCO")  
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            sys.exit("CUDA is not available.") 
        self.scaler = 20.0
        self.num_points = 10
        self.feature_dim = 512

        # ===== ABP: Adaptive Background Prompting configs =====
        self.use_multiring_bppc = bool(args.get("use_multiring_bppc", True))
        self.bppc_ring_kernel_pairs = [
            tuple(pair) for pair in args.get(
                "bppc_ring_kernel_pairs",
                [(17, 11), (21, 15), (25, 19)]
            )
        ]
        self.use_learnable_ring_fusion = bool(args.get("use_learnable_ring_fusion", True))
        if self.use_learnable_ring_fusion:
            self.ring_logits = nn.Parameter(torch.zeros(len(self.bppc_ring_kernel_pairs)))
        else:
            self.register_buffer(
                "ring_logits",
                torch.zeros(len(self.bppc_ring_kernel_pairs)),
                persistent=False
            )

        # Prompt Validity Filter is inference-only.
        self.use_prompt_validity_filter = bool(args.get("use_prompt_validity_filter", True))
        self.neg_fg_threshold = float(args.get("neg_fg_threshold", 0.85))
        self.neg_min_dist = float(args.get("neg_min_dist", 5.0))
        self.neg_topk = int(args.get("neg_topk", 256))
        self.neg_min_keep = int(args.get("neg_min_keep", 6))

        # Default single-ring setting, kept compatible with original FoB.
        self.default_outer_kernel_size = int(args.get("default_outer_kernel_size", 21))
        self.default_inner_kernel_size = int(args.get("default_inner_kernel_size", 15))
        self.pre_process = transforms.Compose([
            transforms.Resize((256, 256)),
        ])
        self.head = Head(self.feature_dim, (256, 256))
        self.criterion = JointsMSELoss(use_target_weight=False)
        self.masked_attention = MaskedAttention(self.feature_dim, num_heads=1, ffn_expansion=1)
        self.L2_loss = nn.MSELoss()
        self.nllloss = nn.NLLLoss(ignore_index=255, weight=torch.FloatTensor([0.1, 1.0]).cuda())
        self.refine = SPR(self.feature_dim, num_heads=1, num_points=8) 
        self.InfoNCE = InfoNCE(negative_mode='unpaired')

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_labels, train):

        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors  (1, 3, 257, 257)
        """
        
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]           
        supp_bs = supp_imgs[0][0].shape[0]      
        img_size = supp_imgs[0][0].shape[-2:]   

        supp_imgs[0][0] = self.pre_process(supp_imgs[0][0])
        supp_mask[0][0] = self.pre_process(supp_mask[0][0])
        qry_imgs[0] = self.pre_process(qry_imgs[0])
        qry_labels = self.pre_process(qry_labels) 

        img_size = supp_imgs[0][0].shape[-2:]   

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)  
        
        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])

        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]


        heatmap_loss = torch.zeros(1).to(self.device)
        rac_loss = torch.zeros(1).to(self.device)
        foreground_loss = torch.zeros(1).to(self.device)
        L2_loss = torch.zeros(1).to(self.device)
        if supp_mask[[0], 0, 0].max() > 0. and qry_labels.max() > 0.:

            # ***************************** Background Prompt Prototype Construction ********************************
            if self.use_multiring_bppc:
                skps, points_spt = self.build_multiring_background_prototypes(
                    supp_fts_one=supp_fts[0][0],
                    supp_mask_one=supp_mask[0],
                    img_size=img_size
                )
            else:
                points_spt = self.uniform_sample_contour(
                    supp_mask[0],
                    num_keypoints=self.num_points,
                    outer_kernel_size=self.default_outer_kernel_size,
                    inner_kernel_size=self.default_inner_kernel_size
                )  # [10, 2]
                heatmaps_spt = self.generate_keypoint_heatmaps(img_size, points_spt)
                heatmaps_spt = torch.from_numpy(heatmaps_spt).float().cuda()
                skps = []
                for i in range(self.num_points):
                    skp = [[self.getFeatures(supp_fts[0][0], heatmaps_spt[i])]]
                    skp = self.getPrototype(skp)[0].transpose(0, 1)
                    skps.append(skp)
                skps = torch.stack(skps).squeeze(2)  # [10, 512]

            # ***************************** Background-centric Context Modeling *****************************
            spt_fts_ = [[self.getFeatures(supp_fts[[[0], way, shot]], supp_mask[[0], way, shot])
                            for shot in range(self.n_shots)] for way in range(self.n_ways)]
            spt_fg_proto = self.getPrototype(spt_fts_)[0] # [1, 512]

            # obtain coarse mask of query *******************
            qry_pred = torch.stack(
                [self.getPred(qry_fts[way], spt_fg_proto[way], self.thresh_pred[way])
                    for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
            qry_pred_coarse = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

            qry_fts_suppressed = self.attention_suppress(qry_fts[0], spt_fg_proto)

            attended_query_fts, sim_heat = self.masked_attention(skps, qry_fts_suppressed)  
            heatmap = self.head(attended_query_fts)
            pred_point = self.get_keypoint_predictions(heatmap).squeeze(0)



            # ***************************** Structure-guided Prompt Refinement *****************************
            heatmaps_qry = self.generate_keypoint_heatmaps(img_size, pred_point) 
            qkps = []
            for i in range(self.num_points):
                qkp = [[self.getFeatures(qry_fts[0], torch.from_numpy(heatmaps_qry[i]).cuda())]] 
                qkp = self.getPrototype(qkp)[0].transpose(0, 1) 
                qkps.append(qkp)
            qkps = torch.stack(qkps).squeeze(2) # [10, 512]

            pred_point = self.refine(qkps.unsqueeze(0), torch.from_numpy(pred_point).cuda().unsqueeze(0), skps.unsqueeze(0), qry_fts[0]).squeeze(0)  # [1, 10, 2]
            pred_point = pred_point.squeeze(0).cpu().detach().numpy()  


            # ************************************* Optimization *************************************
            if train:
                gt = self.uniform_sample_contour(qry_labels.unsqueeze(0).float(), num_keypoints=self.num_points) # [10,2]

                heatmaps_gt = self.generate_keypoint_heatmaps(img_size, gt) #(10, 256, 256)
                heatmap_loss = self.criterion(heatmap.unsqueeze(0), torch.from_numpy(heatmaps_gt).unsqueeze(0).cuda(), None) 
                sim_heat = F.interpolate(sim_heat.unsqueeze(0), size=(img_size), mode='bilinear', align_corners=True)  # [10, 256, 256]
                heatmap_loss = heatmap_loss + self.criterion(sim_heat, torch.from_numpy(heatmaps_gt).unsqueeze(0).cuda(), None)
                
                L2_loss = self.L2_loss(torch.from_numpy(pred_point).float().cuda(), torch.from_numpy(gt).float().cuda()) 

                log_qry_pred_coarse = torch.cat([1 - qry_pred_coarse, qry_pred_coarse], dim=1).log()
                foreground_loss = self.nllloss(log_qry_pred_coarse, qry_labels) 

                for skp in skps:
                   cos_sim = F.cosine_similarity(spt_fg_proto.transpose(1,0), skp.unsqueeze(-1), dim=0)  
                   rac_loss += torch.clamp(0.5 + cos_sim, min=0) / self.num_points  



            # only test ****************************************
            if not train:
                pos_point = self.uniform_sample_from_prob(
                    qry_pred_coarse[0][0],
                    num_samples=10,
                    threshold=0.96
                )

                if self.use_prompt_validity_filter:
                    pred_point = self.filter_background_prompts(
                        pred_point=pred_point,
                        heatmap=heatmap.detach(),
                        fg_prob=qry_pred_coarse[0][0].detach(),
                        fg_threshold=self.neg_fg_threshold,
                        min_dist=self.neg_min_dist,
                        topk=self.neg_topk,
                        min_keep=self.neg_min_keep
                    )

                neg_point = np.array([pred_point])
                return neg_point, pos_point

        return heatmap_loss * 1000 + L2_loss / 10000, rac_loss, foreground_loss

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred
    
    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts
    
    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    

    # ========================= ABP helpers: Multi-ring BPPC =========================

    def _get_ring_weights(self, device):
        """Return normalized fusion weights for multi-ring BPPC."""
        logits = self.ring_logits.to(device)
        if self.use_learnable_ring_fusion:
            return torch.softmax(logits, dim=0)
        return torch.ones_like(logits, device=device) / max(1, logits.numel())

    def build_multiring_background_prototypes(self, supp_fts_one, supp_mask_one, img_size):
        """
        Multi-ring BPPC.

        Original FoB samples support background prompts from a single ring.
        This version samples multiple rings and fuses their prompt prototypes.
        Output prompt number is still self.num_points, so Head/MaskedAttention
        stay compatible with the original FoB implementation.
        """
        device = supp_fts_one.device
        ring_skps = []
        ref_points = None
        ref_idx = len(self.bppc_ring_kernel_pairs) // 2

        for ring_idx, kernel_pair in enumerate(self.bppc_ring_kernel_pairs):
            outer_kernel, inner_kernel = int(kernel_pair[0]), int(kernel_pair[1])

            points = self.uniform_sample_contour(
                supp_mask_one,
                num_keypoints=self.num_points,
                outer_kernel_size=outer_kernel,
                inner_kernel_size=inner_kernel
            )

            if ring_idx == ref_idx:
                ref_points = points

            heatmaps = self.generate_keypoint_heatmaps(img_size, points)
            heatmaps = torch.from_numpy(heatmaps).float().to(device)

            skps_this_ring = []
            for i in range(self.num_points):
                skp = [[self.getFeatures(supp_fts_one, heatmaps[i])]]
                skp = self.getPrototype(skp)[0].transpose(0, 1)
                skps_this_ring.append(skp)

            skps_this_ring = torch.stack(skps_this_ring).squeeze(2)  # [K, C]
            ring_skps.append(skps_this_ring)

        ring_skps = torch.stack(ring_skps, dim=0)  # [R, K, C]
        ring_weights = self._get_ring_weights(device).view(-1, 1, 1)
        skps = torch.sum(ring_skps * ring_weights, dim=0)  # [K, C]

        return skps, ref_points

    # ====================== ABP helpers: Prompt Validity Filter ======================

    def _clip_point(self, point, width, height):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        return np.array([x, y], dtype=np.int32)

    def _is_valid_negative_point(self, point, fg_prob_np, kept_points, fg_threshold, min_dist):
        """
        A negative prompt is invalid if:
        1. it falls into a high-confidence foreground region;
        2. it is too close to already selected negative prompts.
        """
        x, y = int(point[0]), int(point[1])

        if fg_prob_np[y, x] >= fg_threshold:
            return False

        for kept in kept_points:
            if np.linalg.norm(point.astype(np.float32) - kept.astype(np.float32)) < min_dist:
                return False

        return True

    def _pick_replacement_from_heatmap(self, heatmap_np, fg_prob_np, kept_points,
                                       fg_threshold, min_dist, topk):
        """Pick the highest-response valid location from a heatmap channel."""
        height, width = fg_prob_np.shape
        flat = heatmap_np.reshape(-1)
        topk = int(min(max(1, topk), flat.size))

        candidate_idx = np.argpartition(-flat, topk - 1)[:topk]
        candidate_idx = candidate_idx[np.argsort(-flat[candidate_idx])]

        for idx in candidate_idx:
            y = int(idx // width)
            x = int(idx % width)
            point = np.array([x, y], dtype=np.int32)

            if self._is_valid_negative_point(
                point,
                fg_prob_np,
                kept_points,
                fg_threshold,
                min_dist
            ):
                return point

        return None

    def filter_background_prompts(self, pred_point, heatmap, fg_prob,
                                  fg_threshold=0.85, min_dist=5.0,
                                  topk=256, min_keep=6):
        """
        Prompt Validity Filter.

        It corrects unreliable background prompts at inference time.
        If a predicted background point falls into high foreground probability
        or collapses into a cluster, replace it by a valid high-response point
        from the corresponding background heatmap channel.
        """
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.detach().cpu().float().numpy()
        else:
            heatmap_np = heatmap

        if heatmap_np.ndim == 4:
            heatmap_np = heatmap_np[0]

        if isinstance(fg_prob, torch.Tensor):
            fg_prob_np = fg_prob.detach().cpu().float().numpy()
        else:
            fg_prob_np = fg_prob

        if fg_prob_np.ndim == 3:
            fg_prob_np = fg_prob_np[0]

        height, width = fg_prob_np.shape
        pred_point = np.asarray(pred_point, dtype=np.float32)

        kept_points = []
        fallback_points = []

        for i, point in enumerate(pred_point):
            point = self._clip_point(point, width, height)

            if self._is_valid_negative_point(
                point,
                fg_prob_np,
                kept_points,
                fg_threshold,
                min_dist
            ):
                kept_points.append(point)
                continue

            heat_ch = min(i, heatmap_np.shape[0] - 1)
            replacement = self._pick_replacement_from_heatmap(
                heatmap_np[heat_ch],
                fg_prob_np,
                kept_points,
                fg_threshold,
                min_dist,
                topk
            )

            if replacement is not None:
                kept_points.append(replacement)
            else:
                fallback_points.append(point)

        while len(kept_points) < min_keep and len(fallback_points) > 0:
            kept_points.append(fallback_points.pop(0))

        if len(kept_points) == 0:
            kept_points = [self._clip_point(pred_point[0], width, height)]

        return np.asarray(kept_points, dtype=np.int32)

    def uniform_sample_from_prob(self, pred_map, num_samples=10, threshold=0.96):
        """
        Uniformly samples points from a probability map based on a given threshold.
        Args:
            pred_map (torch.Tensor): The probability map.
            num_samples (int, optional): The number of samples to be generated. 
            threshold (float, optional): The threshold value for selecting points. 
        Returns:
            np.ndarray: An array of sampled points in the format [[x, y]].
        """

        mask = (pred_map > threshold)

        coordinates = torch.nonzero(mask, as_tuple=False)
        

        if coordinates.shape[0] == 0: # no point is detected, sample the point with maximum similarity
            max_idx = torch.argmax(pred_map)  
            max_position = torch.unravel_index(max_idx, pred_map.shape)   
            pos_point = np.array([[[max_position[1].item(), max_position[0].item()]]])  # [[x, y]]
            return pos_point
        

        if coordinates.shape[0] <= num_samples:  # NOT ENOUGH POINTS
            sampled_points = coordinates
        else:
            indices = np.linspace(0, coordinates.shape[0] - 1, num_samples).astype(int)
            sampled_points = coordinates[indices]

        pos_points = np.array([[[point[1].item(), point[0].item()] for point in sampled_points]])

        return pos_points



    def dilate_label(self, label, kernel_size=9):
        label_dilate = F.max_pool2d(label, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return label_dilate
    def erode_label(self, label, kernel_size=9):
        label_erode = F.max_pool2d(1 - label, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return 1 - label_erode
    def get_ring(self, label, kernel_size=21, inner_kernel_size=15):
        outer = self.dilate_label(label, kernel_size)
        inner = self.dilate_label(label, inner_kernel_size)
        ring = torch.clamp(outer - inner, min=0.0, max=1.0)
        return ring

    def get_ring_inner(self, label, kernel_size=9):
        label_erode_9 = self.erode_label(label, kernel_size)
        ring = label - label_erode_9
        return ring

    def uniform_sample_contour(self, mask, num_keypoints=10,
                               outer_kernel_size=None, inner_kernel_size=None):
        """
        Uniformly sample points along a background ring contour.

        Args:
            mask: binary mask tensor.
            num_keypoints: number of sampled prompts.
            outer_kernel_size: outer dilation kernel.
            inner_kernel_size: inner dilation kernel.
        """
        if outer_kernel_size is None:
            outer_kernel_size = self.default_outer_kernel_size
        if inner_kernel_size is None:
            inner_kernel_size = self.default_inner_kernel_size

        raw_mask = mask.squeeze().detach().cpu().numpy()
        raw_mask = (raw_mask > 0).astype(np.uint8)

        def fallback_points(binary_mask):
            ys, xs = np.where(binary_mask > 0)
            if len(xs) == 0:
                return np.zeros((num_keypoints, 2), dtype=np.int32)

            coords = np.stack([xs, ys], axis=1).astype(np.float32)
            if coords.shape[0] == 1:
                return np.repeat(coords, num_keypoints, axis=0).astype(np.int32)

            idxs = np.linspace(0, coords.shape[0] - 1, num_keypoints).astype(int)
            return np.round(coords[idxs]).astype(np.int32)

        ring = self.get_ring(
            mask,
            kernel_size=outer_kernel_size,
            inner_kernel_size=inner_kernel_size
        )
        ring = ring.squeeze().detach().cpu().numpy()
        ring = (ring > 0).astype(np.uint8)

        height, width = ring.shape

        contours, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return fallback_points(raw_mask)

        contour = max(contours, key=cv2.contourArea)
        pts = contour[:, 0, :].astype(np.float32)

        if pts.shape[0] < 2:
            return fallback_points(raw_mask if raw_mask.max() > 0 else ring)

        clean_pts = [pts[0]]
        for pt in pts[1:]:
            if np.linalg.norm(pt - clean_pts[-1]) > 1e-6:
                clean_pts.append(pt)

        pts = np.asarray(clean_pts, dtype=np.float32)

        if pts.shape[0] < 2:
            return fallback_points(raw_mask if raw_mask.max() > 0 else ring)

        closed_pts = np.vstack([pts, pts[0]])
        seg_lens = np.linalg.norm(np.diff(closed_pts, axis=0), axis=1)
        total_len = float(seg_lens.sum())

        if total_len <= 1e-6:
            return fallback_points(raw_mask if raw_mask.max() > 0 else ring)

        cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
        desired = np.linspace(0, total_len, num_keypoints, endpoint=False)

        sampled_points = []
        for d in desired:
            idx = np.searchsorted(cumulative, d, side="right") - 1
            idx = max(0, min(idx, len(seg_lens) - 1))

            if seg_lens[idx] <= 1e-6:
                sampled_points.append(closed_pts[idx])
            else:
                ratio = (d - cumulative[idx]) / seg_lens[idx]
                sampled_points.append(closed_pts[idx] + ratio * (closed_pts[idx + 1] - closed_pts[idx]))

        sampled_points = np.round(np.asarray(sampled_points)).astype(np.int32)
        sampled_points[:, 0] = np.clip(sampled_points[:, 0], 0, width - 1)
        sampled_points[:, 1] = np.clip(sampled_points[:, 1], 0, height - 1)

        return sampled_points

    def sort_keypoints_clockwise(self, points):
        start_point = points[np.argmin(points[:, 0])]

        def calculate_angle(point):
            return np.arctan2(point[1] - start_point[1], point[0] - start_point[0])
        
        sorted_points = sorted(points, key=calculate_angle)
        
        return np.array(sorted_points)

   
    def generate_keypoint_heatmaps(self, image_size, keypoints, sigma=4):
        '''
        :param image_size: tuple (height, width) of the heatmap
        :param keypoints: array of shape [num_keypoints, 2], where each row is [x, y]
        :param sigma: standard deviation for the Gaussian
        :return: heatmap for keypoints, with shape [num_keypoints, height, width]
        '''
        num_keypoints = keypoints.shape[0]
        heatmap = np.zeros((num_keypoints, image_size[0], image_size[1]), dtype=np.float32)

        tmp_size = sigma * 3
        keypoints = self.sort_keypoints_clockwise(keypoints)
        for keypoint_id in range(num_keypoints):
            mu_x, mu_y = keypoints[keypoint_id]
            mu_x = int(mu_x + 0.5)
            mu_y = int(mu_y + 0.5)

            # Check if the keypoint is out of bounds
            if mu_x < 0 or mu_y < 0 or mu_x >= image_size[1] or mu_y >= image_size[0]:
                continue

            # Upper left and bottom right corners of the Gaussian
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            # Adjust if the Gaussian is partially out of bounds
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Determine the usable Gaussian range
            g_x = max(0, -ul[0]), min(br[0], image_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], image_size[0]) - ul[1]

            # Image range
            img_x = max(0, ul[0]), min(br[0], image_size[1])
            img_y = max(0, ul[1]), min(br[1], image_size[0])

            # Apply the Gaussian to the heatmap
            heatmap[keypoint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return heatmap


    def get_max_preds(self, batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals
    
    def get_keypoint_predictions(self, output):
        """
        Get the predicted keypoints from the output.
        Parameters:
            output (torch.Tensor): The output tensor containing the batch heatmaps.
        Returns:
            numpy.ndarray: The predicted keypoints.
        """

        batch_heatmaps = output.cpu().detach().numpy()

        preds, _ = self.get_max_preds(batch_heatmaps)
        
        return preds




    def compute_correlation(self, skp, query_feature_map):
        """
        Computes the attention weighted feature with the skp and query_feature_map.
        Args:
            skp (torch.Tensor): The skp tensor with shape (1, feature_dim, 1, 1).
            query_feature_map (torch.Tensor): The query feature map tensor with shape (1, feature_dim, H, W).
        Returns:
            torch.Tensor: The cosine similarity map between the skp and query_feature_map.
        """

        skp = skp.view(1, self.feature_dim, 1, 1)
        skp = F.normalize(skp, dim=1)  
        query_feature_map = F.normalize(query_feature_map, dim=1)  
        cosine_similarity_map = query_feature_map * skp
        
        return cosine_similarity_map

   

    
    def attention_suppress(self, qry_fts, spt_fg_proto):
        """
        Apply attention suppression to the query features based on the support foreground prototypes.
        Args:
            qry_fts (torch.Tensor): Query features of shape (b, c, h, w).
            spt_fg_proto (torch.Tensor): Support foreground prototypes of shape (b, c, 1, 1).
        Returns:
            torch.Tensor: Suppressed query features of shape (b, c, h, w).
        """


        b, c, h, w = qry_fts.shape
        proto_expanded = spt_fg_proto.view(b, c, 1, 1)  # [b, c, 1, 1]
        similarity = F.cosine_similarity(qry_fts, proto_expanded, dim=1)  # [b, h, w]
        attention_weights = 1 - similarity.unsqueeze(1)  # [b, 1, h, w]
        suppressed_qry_fts = qry_fts * attention_weights  # [b, c, h, w]
        
        return suppressed_qry_fts
