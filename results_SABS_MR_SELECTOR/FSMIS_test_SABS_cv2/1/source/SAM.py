# -*- coding: utf-8 -*-
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
