import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from sam2.build_sam2 import build_sam2
from sam2.predictor import SAM2ImagePredictor

from IMDLBenCo.registry import MODELS

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

@MODELS.register_module()
class SAMPrompting(nn.Module):
    def __init__(self, MyModel_Customized_param:int, pre_trained_weights:str) -> None:
        super().__init__()

        sam2_checkpoint = "sam2_hiera_small.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
        model_cfg = "sam2_hiera_s.yaml"  # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        predictor = SAM2ImagePredictor(sam2_model)

        # Train mask decoder.
        predictor.model.sam_mask_decoder.train(True)

        # Train prompt encoder.
        predictor.model.sam_prompt_encoder.train(True)

        self.loss_func_a = nn.BCEWithLogitsLoss()
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]


    def forward(self, image, mask, label, *args, **kwargs):
        gt_masks = mask

        batch_size = image.shape[0]


        backbone_out = self.visual_model.forward_image(image)
        _, image_embeddings, _, _ = self.visual_model._prepare_backbone_features(backbone_out)
        image_embeddings = [_.to(image.dtype) for _ in image_embeddings]
        if self.visual_model.directly_add_no_mem_embed:
            image_embeddings[-1] = image_embeddings[-1] + self.visual_model.no_mem_embed

        feats = [feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                 for feat, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}


        multimask_output = True

        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(batch_size):
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=None, boxes=None, masks=None,
            )

            high_res_features = [
                feat_level[img_idx].unsqueeze(0)
                for feat_level in _features["high_res_feats"]
            ]
            low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                image_embeddings=_features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=True,
                high_res_features=high_res_features,
            )

            # Upscale the masks to the original image resolution
            masks = self._transforms.postprocess_masks(
                low_res_masks, self._orig_hw[img_idx]
            )
            low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)


            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)


        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(all_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = all_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss


        output_dict = {
            "backward_loss": mask_loss,
            # "pred_mask": pred_mask,
            # "pred_label": pred_label,

            "visual_loss": {
                "mask_bce_loss": mask_bce_loss,
                'mask_dice_loss' : mask_dice_loss,
                "mask_loss": mask_loss,
            },

            # "visual_image": {
            #     "pred_mask": pred_mask,
            #     "reverse_mask" : inverse_mask,
            # }
        }
        
        return output_dict




























