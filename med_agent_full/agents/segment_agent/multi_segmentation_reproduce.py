import autorootcwd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import copy
from segment_anything_CoMed import sam_model_registry
import torch.nn.functional as F
from torch.utils.data import DataLoader
from IVDM3Seg.train.train import NpyDataset, CoMedSAM, show_mask, show_box, args, join
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Dice Score 계산 함수
def calculate_dice_score(pred, target, epsilon=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    
    return dice

def inference_on_npy(data_root, file_prefix, bbox_shift=0):
    """
    :param data_root: 데이터셋 루트 경로
    :param file_prefix: 처리할 파일의 prefix (예: "15-12")
    """
    # ✅ 해당 prefix를 포함하는 모든 npy 파일 검색
    matching_files = sorted(glob.glob(os.path.join(data_root, "gts", f"{file_prefix}_*.npy")))

    if not matching_files:
        print(f"No matching files found for prefix: {file_prefix}")
        return

    # ✅ 모델 로딩
    sam_model = sam_model_registry["vit_b"](checkpoint="/home/minkyukim/mm-sam_tutorial/work_dir/SAM/sam_vit_b_01ec64.pth")

    def create_image_encoder():
        return copy.deepcopy(sam_model.image_encoder).to(device)

    mm_sam = CoMedSAM(
        image_encoder_factory=create_image_encoder,
        mask_decoder=sam_model.mask_decoder.to(device),  
        prompt_encoder=sam_model.prompt_encoder.to(device), 
        indicator=[1, 1, 1, 1]
    ).to(device)

    checkpoint_path = "/mnt/sda/minkyukim/pth/revision/MM_ivdm_2way/MM_2.pth"
    checkpoint = torch.load(checkpoint_path)

    mm_sam.load_state_dict(checkpoint, strict=False)
    mm_sam.eval()

    all_input_images = []
    all_gt_masks = []
    all_predicted_masks = []

    for npy_file in matching_files:
        dataset = NpyDataset(data_root, bbox_shift=bbox_shift)
        dataset.gt_path_files = [npy_file]
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for step, (images, gt, bboxes, img_names) in enumerate(dataloader):
            images = images.to(device)
            gt = gt.to(device)

            with torch.no_grad():
                pred_mask = mm_sam(images, bboxes.cpu().numpy())

            # Resizing GT and predicted masks
            gt_resized = F.interpolate(gt.float(), size=(1024, 1024), mode='nearest').squeeze(1)
            gt_mask_np = np.clip(gt_resized[0].cpu().numpy(), 0, 1)

            pred_mask_resized = F.interpolate(pred_mask.float(), size=(1024, 1024), mode='nearest').squeeze(0)
            pred_mask_np = np.clip(pred_mask_resized[0].cpu().numpy(), 0, 1)

            # ✅ GT 및 Predicted Mask 저장 리스트에 추가
            all_gt_masks.append(gt_mask_np)
            all_predicted_masks.append(pred_mask_np)


            # 🔥 첫 4개 Input Image만 저장
            for i in range(min(images.shape[1], 4)):  
                if len(all_input_images) < 4:  # 첫 4개까지만 저장
                    image_np = images[0, i].cpu().permute(1, 2, 0).numpy()
                    image_np = np.clip(image_np, 0, 1)
                    all_input_images.append(image_np)

            print(f"Inference completed for: {npy_file}")

    # ✅ GT Masks 및 Predicted Masks를 하나로 합산
    gt_final_mask = np.sum(all_gt_masks, axis=0)
    pred_final_mask = np.sum(all_predicted_masks, axis=0)
    pred_final_mask[pred_final_mask > 1] = pred_final_mask[pred_final_mask > 1] // 2

    # ✅ Dice Score 계산 (최종 GT와 최종 예측 값 비교)
    final_dice_score = calculate_dice_score(torch.tensor(pred_final_mask), torch.tensor(gt_final_mask))
    dice_score_str = f"{final_dice_score:.4f}"

    # ✅ 폴더 생성 (폴더명에 Dice Score 추가)
    output_folder = f"ablation_images/{file_prefix}_dice_{dice_score_str}"
    os.makedirs(output_folder, exist_ok=True)

    # ✅ Input Image 4개 개별 저장
    for i, img in enumerate(all_input_images):
        plt.imsave(f"{output_folder}/input_{i+1}.png", img)

    # ✅ 최종 GT 및 Predicted Mask 저장
    plt.imsave(f"{output_folder}/gt_final.png", gt_final_mask, cmap='gray')
    plt.imsave(f"{output_folder}/predicted_final.png", pred_final_mask, cmap='gray')
    print(f"Saved Final GT Mask: {output_folder}/gt_final.png")
    print(f"Saved Final Predicted Mask: {output_folder}/predicted_final.png")

    # 🔥 오버레이 시각화: GT 기준으로 FP(파란색), FN(빨간색) 표시
    overlay = np.zeros((*gt_final_mask.shape, 3), dtype=np.uint8)

    false_positive = (pred_final_mask > 0) & (gt_final_mask == 0)  # FP: 있어야 할 곳에 없음
    false_negative = (pred_final_mask == 0) & (gt_final_mask > 0)  # FN: 없어야 할 곳에 있음

    overlay[false_positive] = [0, 0, 255]  # 🔵 파란색 (FP)
    overlay[false_negative] = [255, 0, 0]  # 🔴 빨간색 (FN)

    # ✅ Overlay 저장
    plt.imsave(f"{output_folder}/overlay.png", overlay)
    print(f"Saved Overlay Image: {output_folder}/overlay.png")

    # 🔥 `predicted_final.png` 위에 Overlay를 우선적으로 적용한 이미지 저장
    predicted_rgb = np.stack([pred_final_mask] * 3, axis=-1) * 255  # Grayscale을 RGB로 변환

    # ✅ Overlay가 있는 픽셀을 우선으로 표시 (FP/FN이 있는 곳은 Overlay 색상 유지)
    mask_overlay = (overlay > 0).any(axis=-1)  # FP/FN이 있는 부분 찾기
    predicted_rgb[mask_overlay] = overlay[mask_overlay]  # Overlay 적용

    plt.imsave(f"{output_folder}/overlay_on_predicted.png", predicted_rgb/255.0)
    print(f"Saved Overlay on Predicted Image: {output_folder}/overlay_on_predicted.png")

# 실행
data_root = "/mnt/sda/minkyukim/CoMed-sam_dataset/IVDM_/ivdm_npy_test_dataset_1024image"
file_prefix = "15-13"  # 🔥 Prefix를 입력하면 해당하는 모든 Phase를 자동으로 찾음
inference_on_npy(data_root, file_prefix)