import autorootcwd
import os
import glob
import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from med_agent_full.agents.segment_agent.segment_anything_CoMed import sam_model_registry
from med_agent_full.agents.segment_agent.IVDM3Seg.train.train import NpyDataset, CoMedSAM, args, join  # 기존 코드 그대로 사용

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_dice_score(pred, target, epsilon=1e-6):
    """Binary Dice score 계산."""
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return dice


class SegmentationAgent:
    """
    IVDM 4-phase (WAT, FAT, INN, OPP) 세그멘테이션을 수행하고,
    다음 에이전트에서 사용할 메타데이터를 반환하는 에이전트.
    """

    def __init__(
        self,
        data_root: str,
        sam_ckpt_path: str = "/home/minkyukim/mm-sam_tutorial/work_dir/SAM/sam_vit_b_01ec64.pth",
        mm_sam_ckpt_path: str = "/mnt/sda/minkyukim/pth/revision/MM_ivdm_2way/MM_2.pth",
        output_root: str = "ablation_images",
        bbox_shift: int = 0,
    ):
        self.data_root = data_root
        self.sam_ckpt_path = sam_ckpt_path
        self.mm_sam_ckpt_path = mm_sam_ckpt_path
        self.output_root = output_root
        self.bbox_shift = bbox_shift

        os.makedirs(self.output_root, exist_ok=True)

        # SAM 기반 CoMedSAM 초기화
        sam_model = sam_model_registry["vit_b"](checkpoint=self.sam_ckpt_path)

        def create_image_encoder():
            return copy.deepcopy(sam_model.image_encoder).to(device)

        self.mm_sam = CoMedSAM(
            image_encoder_factory=create_image_encoder,
            mask_decoder=sam_model.mask_decoder.to(device),
            prompt_encoder=sam_model.prompt_encoder.to(device),
            indicator=[1, 1, 1, 1],
        ).to(device)

        checkpoint = torch.load(self.mm_sam_ckpt_path, map_location=device)
        self.mm_sam.load_state_dict(checkpoint, strict=False)
        self.mm_sam.eval()

    def run(self, file_prefix: str):
        """
        file_prefix (예: '15-13') 를 입력받아,
        - 4-phase input 이미지
        - Final GT mask / Pred mask / Overlay / Overlay on pred
        - Dice score
        를 생성하고, 다음 에이전트가 사용할 dict를 반환.
        """
        matching_files = sorted(
            glob.glob(os.path.join(self.data_root, "gts", f"{file_prefix}_*.npy"))
        )

        if not matching_files:
            print(f"[SegmentationAgent] No matching files found for prefix: {file_prefix}")
            return None

        all_input_images = []     # 4-phase input 저장
        all_gt_masks = []
        all_predicted_masks = []

        for npy_file in matching_files:
            dataset = NpyDataset(self.data_root, bbox_shift=self.bbox_shift)
            dataset.gt_path_files = [npy_file]

            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            for step, (images, gt, bboxes, img_names) in enumerate(dataloader):
                images = images.to(device)
                gt = gt.to(device)

                with torch.no_grad():
                    pred_mask = self.mm_sam(images, bboxes.cpu().numpy())

                # GT / Pred를 1024x1024로 resize
                gt_resized = F.interpolate(gt.float(), size=(1024, 1024), mode="nearest").squeeze(1)
                gt_mask_np = np.clip(gt_resized[0].cpu().numpy(), 0, 1)

                pred_resized = F.interpolate(pred_mask.float(), size=(1024, 1024), mode="nearest").squeeze(0)
                pred_mask_np = np.clip(pred_resized[0].cpu().numpy(), 0, 1)

                all_gt_masks.append(gt_mask_np)
                all_predicted_masks.append(pred_mask_np)

                # 첫 4개 phase 입력만 저장 (WAT, FAT, INN, OPP 순서)
                for i in range(min(images.shape[1], 4)):
                    if len(all_input_images) < 4:
                        img_np = images[0, i].cpu().permute(1, 2, 0).numpy()
                        img_np = np.clip(img_np, 0, 1)
                        all_input_images.append(img_np)

                print(f"[SegmentationAgent] Inference completed for: {npy_file}")

        # Final GT / Pred mask (모든 slice 합)
        gt_final_mask = np.sum(all_gt_masks, axis=0)
        pred_final_mask = np.sum(all_predicted_masks, axis=0)
        pred_final_mask[pred_final_mask > 1] = pred_final_mask[pred_final_mask > 1] // 2

        # Dice score
        final_dice_score = calculate_dice_score(
            torch.tensor(pred_final_mask), torch.tensor(gt_final_mask)
        )
        dice_score_str = f"{final_dice_score:.4f}"

        # output 폴더
        output_folder = os.path.join(self.output_root, f"{file_prefix}_dice_{dice_score_str}")
        os.makedirs(output_folder, exist_ok=True)

        # 4-phase input 저장 (1: WAT, 2: FAT, 3: INN, 4: OPP)
        phase_names = ["WAT", "FAT", "INN", "OPP"]
        phase_image_paths = {}

        for i, img in enumerate(all_input_images):
            phase_name = phase_names[i] if i < len(phase_names) else f"PHASE_{i+1}"
            img_path = os.path.join(output_folder, f"input_{i+1}_{phase_name}.png")
            plt.imsave(img_path, img)
            phase_image_paths[phase_name] = img_path

        # Final GT / Pred 저장
        gt_path = os.path.join(output_folder, "gt_final.png")
        pred_path = os.path.join(output_folder, "predicted_final.png")
        plt.imsave(gt_path, gt_final_mask, cmap="gray")
        plt.imsave(pred_path, pred_final_mask, cmap="gray")
        print(f"[SegmentationAgent] Saved Final GT Mask: {gt_path}")
        print(f"[SegmentationAgent] Saved Final Predicted Mask: {pred_path}")

        # FP/FN overlay
        overlay = np.zeros((*gt_final_mask.shape, 3), dtype=np.uint8)
        false_positive = (pred_final_mask > 0) & (gt_final_mask == 0)
        false_negative = (pred_final_mask == 0) & (gt_final_mask > 0)
        overlay[false_positive] = [0, 0, 255]   # FP: 파란색
        overlay[false_negative] = [255, 0, 0]   # FN: 빨간색

        overlay_path = os.path.join(output_folder, "overlay.png")
        plt.imsave(overlay_path, overlay)
        print(f"[SegmentationAgent] Saved Overlay Image: {overlay_path}")

        # overlay_on_predicted
        predicted_rgb = np.stack([pred_final_mask] * 3, axis=-1) * 255
        mask_overlay = (overlay > 0).any(axis=-1)
        predicted_rgb[mask_overlay] = overlay[mask_overlay]

        overlay_on_pred_path = os.path.join(output_folder, "overlay_on_predicted.png")
        plt.imsave(overlay_on_pred_path, predicted_rgb / 255.0)
        print(f"[SegmentationAgent] Saved Overlay on Predicted Image: {overlay_on_pred_path}")

        # 다음 에이전트에 넘길 메타데이터
        seg_output = {
            "case_id": file_prefix,
            "output_dir": output_folder,
            "dice_score": float(final_dice_score),
            "phase_images": phase_image_paths,   # {"WAT": path, "FAT": path, ...}
            "masks": {
                "gt_final": gt_path,
                "pred_final": pred_path,
                "overlay": overlay_path,
                "overlay_on_predicted": overlay_on_pred_path,
            },
        }

        return seg_output
