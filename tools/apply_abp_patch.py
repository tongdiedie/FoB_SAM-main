#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply Adaptive Background Prompting patch to FoB_SAM-main.

Patch includes:
1. Multi-ring BPPC
2. Prompt Validity Filter
3. SAM Mask Consistency Selector
4. env-config support in train.py / test.py

Run from repository root:
    python tools/apply_abp_patch.py
"""

from pathlib import Path
from datetime import datetime
import re
import shutil
import textwrap


ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_file(path: Path) -> None:
    if path.exists():
        backup_path = path.with_suffix(path.suffix + f".bak_abp_{STAMP}")
        shutil.copy2(path, backup_path)
        print(f"[backup] {path.relative_to(ROOT)} -> {backup_path.name}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"[write] {path.relative_to(ROOT)}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def replace_regex_once(text: str, pattern: str, repl, tag: str, flags=re.S) -> str:
    new_text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"Failed to patch block: {tag}")
    print(f"[patch] {tag}")
    return new_text


ABP_HELPERS = r'''
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
'''


SAM_CODE = r'''# -*- coding: utf-8 -*-
# Modified SAM wrapper for FoB_SAM-main.
# Adds SAM Mask Consistency Selector while preserving the original interface.

import os
import numpy as np
import torch.nn as nn
from segment_anything import sam_model_registry, SamPredictor


def _env_bool(name, default=True):
    value = os.environ.get(name, None)
    if value is None:
        return default
    return value.lower() in ["1", "true", "yes", "y", "on"]


class SAM(nn.Module):
    def __init__(
        self,
        sam_pretrained_path="checkpoints/sam_vit_h_4b8939.pth",
        use_consistency_selector=None,
        sam_score_weight=None,
        pos_consistency_weight=None,
        neg_consistency_weight=None,
    ):
        super().__init__()

        self.use_consistency_selector = (
            _env_bool("USE_SAM_SELECTOR", True)
            if use_consistency_selector is None
            else bool(use_consistency_selector)
        )

        self.sam_score_weight = (
            float(os.environ.get("SAM_SCORE_WEIGHT", 0.50))
            if sam_score_weight is None
            else float(sam_score_weight)
        )
        self.pos_consistency_weight = (
            float(os.environ.get("POS_CONSISTENCY_WEIGHT", 0.25))
            if pos_consistency_weight is None
            else float(pos_consistency_weight)
        )
        self.neg_consistency_weight = (
            float(os.environ.get("NEG_CONSISTENCY_WEIGHT", 0.25))
            if neg_consistency_weight is None
            else float(neg_consistency_weight)
        )

        self.get_sam(sam_pretrained_path)

    def get_sam(self, checkpoint_path):
        model_type = "vit_h"
        print(f"Using model type {model_type}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path).eval().cuda()
        self.predictor = SamPredictor(self.sam)
        self.sam.requires_grad_(False)

    def _fallback_best_idx(self, config):
        if config is not None and hasattr(config, "get") and config.get("dataset", None) == "isic":
            return 1
        return 0

    def _point_values_on_mask(self, mask, points):
        if points is None or len(points) == 0:
            return None

        height, width = mask.shape
        values = []

        for point in points:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            values.append(float(mask[y, x]))

        if len(values) == 0:
            return None

        return float(np.mean(values))

    def _select_mask_by_prompt_consistency(self, masks, sam_scores, points, point_labels, config=None):
        """
        Select the SAM candidate mask that is most consistent with FoB prompts.

        Score:
            alpha * SAM predicted IoU
          + beta  * positive prompt inclusion
          + gamma * negative prompt exclusion
        """
        if points is None or point_labels is None:
            return int(np.argmax(sam_scores))

        points = np.asarray(points)
        point_labels = np.asarray(point_labels)

        pos_points = points[point_labels == 1]
        neg_points = points[point_labels == 0]

        best_idx = self._fallback_best_idx(config)
        best_value = -1e9

        for idx in range(len(masks)):
            mask_bin = masks[idx] > 0

            sam_score = float(sam_scores[idx])

            pos_score = self._point_values_on_mask(mask_bin, pos_points)
            neg_inside = self._point_values_on_mask(mask_bin, neg_points)

            if pos_score is None:
                pos_score = 0.5

            if neg_inside is None:
                neg_score = 0.5
            else:
                neg_score = 1.0 - neg_inside

            value = (
                self.sam_score_weight * sam_score
                + self.pos_consistency_weight * pos_score
                + self.neg_consistency_weight * neg_score
            )

            if value > best_value:
                best_value = value
                best_idx = idx

        return int(best_idx)

    def predict_w_points_bbox(self, sam_input_points, bboxes, sam_neg_input_points,
                              qry_img, config=None, return_logits=False):
        masks, scores = [], []

        self.predictor.set_image(qry_img)
        bbox_xyxy = None

        all_points = []
        all_labels = []

        if sam_input_points is not None:
            for point in sam_input_points:
                assert qry_img.max() <= 255 and qry_img.min() >= 0 and qry_img.dtype == np.uint8
                if point is not None:
                    all_points.append(point)
                    all_labels.extend([1] * len(point))

        if sam_neg_input_points is not None:
            for neg_point in sam_neg_input_points:
                if neg_point is not None:
                    all_points.append(neg_point)
                    all_labels.extend([0] * len(neg_point))

        if len(all_points) > 0:
            points = np.vstack(all_points)
            point_labels = np.array(all_labels)
        else:
            points = None
            point_labels = None

        mask, score, logits = self.predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=bbox_xyxy if bbox_xyxy is not None else None,
            return_logits=return_logits,
            multimask_output=True
        )

        if self.use_consistency_selector:
            best_pred_idx = self._select_mask_by_prompt_consistency(
                masks=mask,
                sam_scores=score,
                points=points,
                point_labels=point_labels,
                config=config
            )
        else:
            best_pred_idx = self._fallback_best_idx(config)

        masks.append(mask[best_pred_idx])
        scores.append(score[best_pred_idx])

        return masks, scores

    def pre_process(self, image):
        image = image.permute(1, 2, 0).cpu().numpy()
        denom = image.max() - image.min()

        if denom < 1e-8:
            image = np.zeros_like(image)
        else:
            image = (image - image.min()) / denom

        image = (image * 255).astype(np.uint8)
        return image

    def forward(self, query_image, pos_point=None, neg_point=None,
                config=None, return_logits=False):
        query_image = self.pre_process(query_image)

        mask, score = self.predict_w_points_bbox(
            pos_point,
            None,
            neg_point,
            query_image,
            config,
            return_logits=return_logits
        )

        return mask[0]
'''


ABP_CONFIG_FUNCTION = r'''

def _abp_env_bool(name, default=False):
    value = os.environ.get(name, None)
    if value is None:
        return default
    return value.lower() in ["1", "true", "yes", "y", "on"]


def get_abp_model_config():
    return {
        # Multi-ring BPPC
        "use_multiring_bppc": _abp_env_bool("USE_MULTIRING_BPPC", True),
        "bppc_ring_kernel_pairs": [(17, 11), (21, 15), (25, 19)],
        "use_learnable_ring_fusion": _abp_env_bool("USE_LEARNABLE_RING_FUSION", True),

        # Prompt Validity Filter, inference-only
        "use_prompt_validity_filter": _abp_env_bool("USE_PROMPT_VALIDITY_FILTER", True),
        "neg_fg_threshold": float(os.environ.get("NEG_FG_THRESHOLD", 0.85)),
        "neg_min_dist": float(os.environ.get("NEG_MIN_DIST", 5.0)),
        "neg_topk": int(os.environ.get("NEG_TOPK", 256)),
        "neg_min_keep": int(os.environ.get("NEG_MIN_KEEP", 6)),
    }
'''


def patch_fob() -> None:
    path = ROOT / "models" / "FoB.py"
    require_file(path)
    backup_file(path)
    text = read_text(path)

    if "ABP: Adaptive Background Prompting configs" not in text:
        text = replace_regex_once(
            text,
            r"(def __init__\(self, args\):\s*\n\s*super\(\).__init__\(\))",
            r"\1\n        if args is None:\n            args = {}",
            "FewShotSeg.__init__ args default"
        )

        def repl_config(m):
            indent = m.group(1)
            original_line = m.group(0)
            return f'''{original_line}

{indent}# ===== ABP: Adaptive Background Prompting configs =====
{indent}self.use_multiring_bppc = bool(args.get("use_multiring_bppc", True))
{indent}self.bppc_ring_kernel_pairs = [
{indent}    tuple(pair) for pair in args.get(
{indent}        "bppc_ring_kernel_pairs",
{indent}        [(17, 11), (21, 15), (25, 19)]
{indent}    )
{indent}]
{indent}self.use_learnable_ring_fusion = bool(args.get("use_learnable_ring_fusion", True))
{indent}if self.use_learnable_ring_fusion:
{indent}    self.ring_logits = nn.Parameter(torch.zeros(len(self.bppc_ring_kernel_pairs)))
{indent}else:
{indent}    self.register_buffer(
{indent}        "ring_logits",
{indent}        torch.zeros(len(self.bppc_ring_kernel_pairs)),
{indent}        persistent=False
{indent}    )

{indent}# Prompt Validity Filter is inference-only.
{indent}self.use_prompt_validity_filter = bool(args.get("use_prompt_validity_filter", True))
{indent}self.neg_fg_threshold = float(args.get("neg_fg_threshold", 0.85))
{indent}self.neg_min_dist = float(args.get("neg_min_dist", 5.0))
{indent}self.neg_topk = int(args.get("neg_topk", 256))
{indent}self.neg_min_keep = int(args.get("neg_min_keep", 6))

{indent}# Default single-ring setting, kept compatible with original FoB.
{indent}self.default_outer_kernel_size = int(args.get("default_outer_kernel_size", 21))
{indent}self.default_inner_kernel_size = int(args.get("default_inner_kernel_size", 15))'''

        text = replace_regex_once(
            text,
            r"(?m)^(\s*)self\.feature_dim\s*=\s*512\s*$",
            repl_config,
            "FewShotSeg ABP config insertion",
            flags=0
        )

    # Insert helper methods before they are referenced by BPPC/inference patches.
    # The previous version checked the helper marker before insertion, which caused RuntimeError.
    if "def _get_ring_weights(self, device):" not in text:
        marker = "    def uniform_sample_from_prob(self, pred_map, num_samples=10, threshold=0.96):"
        if marker not in text:
            raise RuntimeError("Cannot find uniform_sample_from_prob marker in models/FoB.py.")
        text = text.replace(marker, ABP_HELPERS + "\n" + marker, 1)
        print("[patch] insert ABP helper methods")

    # Replace BPPC block.
    if "if self.use_multiring_bppc:" not in text:
        def repl_bppc(m):
            indent = m.group("indent")
            return f'''{indent}# ***************************** Background Prompt Prototype Construction ********************************
{indent}if self.use_multiring_bppc:
{indent}    skps, points_spt = self.build_multiring_background_prototypes(
{indent}        supp_fts_one=supp_fts[0][0],
{indent}        supp_mask_one=supp_mask[0],
{indent}        img_size=img_size
{indent}    )
{indent}else:
{indent}    points_spt = self.uniform_sample_contour(
{indent}        supp_mask[0],
{indent}        num_keypoints=self.num_points,
{indent}        outer_kernel_size=self.default_outer_kernel_size,
{indent}        inner_kernel_size=self.default_inner_kernel_size
{indent}    )  # [10, 2]
{indent}    heatmaps_spt = self.generate_keypoint_heatmaps(img_size, points_spt)
{indent}    heatmaps_spt = torch.from_numpy(heatmaps_spt).float().cuda()
{indent}    skps = []
{indent}    for i in range(self.num_points):
{indent}        skp = [[self.getFeatures(supp_fts[0][0], heatmaps_spt[i])]]
{indent}        skp = self.getPrototype(skp)[0].transpose(0, 1)
{indent}        skps.append(skp)
{indent}    skps = torch.stack(skps).squeeze(2)  # [10, 512]'''

        text = replace_regex_once(
            text,
            r"(?P<indent>\s*)# \*+\s*Background Prompt Prototype Construction\s*\*+\s*\n"
            r"(?P=indent)points_spt\s*=\s*self\.uniform_sample_contour\(supp_mask\[0\],\s*num_keypoints=self\.num_points\).*?"
            r"(?P=indent)skps\s*=\s*torch\.stack\(skps\)\.squeeze\(2\)\s*# \[10, 512\]",
            repl_bppc,
            "BPPC -> Multi-ring BPPC",
            flags=re.S
        )

    # Replace inference-only prompt return block.
    if "filter_background_prompts(" not in text:
        raise RuntimeError("ABP filter helper marker not inserted. Check helper insertion order.")

    if "self.use_prompt_validity_filter" in text and "pred_point = self.filter_background_prompts(" not in text:
        def repl_infer(m):
            indent = m.group("indent")
            return f'''{indent}if not train:
{indent}    pos_point = self.uniform_sample_from_prob(
{indent}        qry_pred_coarse[0][0],
{indent}        num_samples=10,
{indent}        threshold=0.96
{indent}    )

{indent}    if self.use_prompt_validity_filter:
{indent}        pred_point = self.filter_background_prompts(
{indent}            pred_point=pred_point,
{indent}            heatmap=heatmap.detach(),
{indent}            fg_prob=qry_pred_coarse[0][0].detach(),
{indent}            fg_threshold=self.neg_fg_threshold,
{indent}            min_dist=self.neg_min_dist,
{indent}            topk=self.neg_topk,
{indent}            min_keep=self.neg_min_keep
{indent}        )

{indent}    neg_point = np.array([pred_point])
{indent}    return neg_point, pos_point'''

        text = replace_regex_once(
            text,
            r"(?P<indent>\s*)if not train:\s*\n"
            r"(?P=indent)\s*pos_point\s*=\s*self\.uniform_sample_from_prob\(qry_pred_coarse\[0\]\[0\],\s*num_samples=10,\s*threshold=0\.96\)\s*\n"
            r"(?P=indent)\s*neg_point\s*=\s*np\.array\(\[pred_point\]\)\s*\n"
            r"(?P=indent)\s*return neg_point,\s*pos_point",
            repl_infer,
            "Inference -> Prompt Validity Filter",
            flags=re.S
        )

    # Insert helper methods before uniform_sample_from_prob.
    if "def _get_ring_weights(self, device):" not in text:
        text = text.replace(
            "    def uniform_sample_from_prob(self, pred_map, num_samples=10, threshold=0.96):",
            ABP_HELPERS + "\n    def uniform_sample_from_prob(self, pred_map, num_samples=10, threshold=0.96):",
            1
        )
        print("[patch] insert ABP helper methods")

    # Replace get_ring.
    if "inner_kernel_size=15" not in re.search(r"    def get_ring\(self.*?(?=    def get_ring_inner)", text, re.S).group(0):
        new_get_ring = r'''    def get_ring(self, label, kernel_size=21, inner_kernel_size=15):
        outer = self.dilate_label(label, kernel_size)
        inner = self.dilate_label(label, inner_kernel_size)
        ring = torch.clamp(outer - inner, min=0.0, max=1.0)
        return ring

'''
        text = replace_regex_once(
            text,
            r"    def get_ring\(self, label, kernel_size=9\):.*?(?=    def get_ring_inner)",
            new_get_ring,
            "get_ring supports outer/inner kernels",
            flags=re.S
        )

    # Replace uniform_sample_contour.
    if "outer_kernel_size=None" not in text:
        new_uniform_sample = r'''    def uniform_sample_contour(self, mask, num_keypoints=10,
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

'''
        text = replace_regex_once(
            text,
            r"    def uniform_sample_contour\(self, mask, num_keypoints=10\):.*?(?=    def sort_keypoints_clockwise)",
            new_uniform_sample,
            "uniform_sample_contour supports multi-ring kernels",
            flags=re.S
        )

    write_text(path, text)


def patch_train_or_test(filename: str) -> None:
    path = ROOT / filename
    require_file(path)
    backup_file(path)
    text = read_text(path)

    if not re.search(r"(?m)^import os$", text):
        text = text.replace("import shutil", "import os\nimport shutil", 1)
        print(f"[patch] {filename}: add import os")

    if "def get_abp_model_config():" not in text:
        if "import torchvision.transforms as transforms" in text:
            text = text.replace(
                "import torchvision.transforms as transforms",
                "import torchvision.transforms as transforms" + ABP_CONFIG_FUNCTION,
                1
            )
        else:
            text = text.replace("from utils import *", "from utils import *" + ABP_CONFIG_FUNCTION, 1)
        print(f"[patch] {filename}: insert get_abp_model_config")

    if "ABP model config" not in text:
        def repl_model_config(m):
            indent = m.group(1)
            return (
                f"{indent}model_config = get_abp_model_config()\n"
                f'{indent}_log.info(f"ABP model config: {{model_config}}")\n'
                f"{indent}model = FewShotSeg(model_config)"
            )

        text = replace_regex_once(
            text,
            r"(?m)^(\s*)model_config\s*=\s*\{\s*\}\s*\n\s*model\s*=\s*FewShotSeg\(model_config\)",
            repl_model_config,
            f"{filename}: model_config -> ABP config",
            flags=re.S
        )

    if filename == "test.py" and "SAM_CKPT" not in text:
        text = re.sub(
            r'sam\s*=\s*SAM\(sam_pretrained_path="([^"]+)"\)',
            r'sam = SAM(sam_pretrained_path=os.environ.get("SAM_CKPT", "\1"))',
            text,
            count=1
        )
        print("[patch] test.py: SAM_CKPT env support")

    write_text(path, text)


def write_sam_wrapper() -> None:
    path = ROOT / "SAM.py"
    backup_file(path)
    write_text(path, SAM_CODE)


def main() -> None:
    print(f"[root] {ROOT}")

    patch_fob()
    patch_train_or_test("train.py")
    patch_train_or_test("test.py")
    write_sam_wrapper()

    print("\nDone.")
    print("Recommended syntax check:")
    print("    python -m py_compile models/FoB.py train.py test.py SAM.py")
    print("\nAblation switches:")
    print("    USE_MULTIRING_BPPC=0/1")
    print("    USE_PROMPT_VALIDITY_FILTER=0/1")
    print("    USE_SAM_SELECTOR=0/1")
    print("    SAM_CKPT=/path/to/sam_vit_h_4b8939.pth")


if __name__ == "__main__":
    main()
