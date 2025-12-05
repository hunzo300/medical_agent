import os
from typing import Dict
import google.generativeai as genai


class ReportDraftAgent:
    """
    phase별 소견 + segmentation 품질(Dice, mask, overlay)을 종합하여
    리포트 초안을 생성하는 에이전트.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def run(self, seg_output: Dict, phase_findings: Dict[str, str]) -> str:
        """
        seg_output + phase_findings를 받아 리포트 초안 텍스트를 반환.
        """
        case_id = seg_output["case_id"]
        dice_score = seg_output["dice_score"]
        masks = seg_output["masks"]
        phase_images = seg_output["phase_images"]

        # ⭐ f-string 안에 복잡한 문자열 연산이 들어가면 SyntaxError → 미리 생성
        phase_image_block = "\n".join(
            [f"- {phase}: {path}" for phase, path in phase_images.items()]
        )

        phase_findings_block = "\n".join(
            [f"=== {phase} ===\n{finding}\n" for phase, finding in phase_findings.items()]
        )

        # -----------------------------
        # 이제 f-string 안에는 변수만 넣는다 → SyntaxError 100% 해결
        # -----------------------------
        prompt = f"""
당신은 MRI IVDM multi-phase 세그멘테이션 연구를 위한 리포트 작성 어시스턴트입니다.
다음 정보를 종합하여 "초안 리포트"를 작성하세요.

[케이스 ID]
- {case_id}

[세그멘테이션 품질]
- Dice score (GT vs 모델 예측): {dice_score:.4f}
- GT mask 파일 경로: {masks['gt_final']}
- Predicted mask 파일 경로: {masks['pred_final']}
- Overlay (FP/FN 표시) 파일 경로: {masks['overlay']}
- Overlay on Predicted 파일 경로: {masks['overlay_on_predicted']}

[Phase별 입력 이미지 파일 경로]
{phase_image_block}

[Phase별 소견]
{phase_findings_block}

리포트 초안의 목적:
- 각 phase별 소견을 요약하고,
- multi-phase 관점에서 병변의 전체 양상과 세그멘테이션 품질을 기술적으로 설명하며,
- 임상 진단을 직접적으로 내리지 않고, 연구/모델 분석 관점에서 기술합니다.

리포트 초안 형식 (한국어):

[1. 검사 개요]
- IVDM multi-phase MRI (WAT/FAT/INN/OPP) 검사 배경 및 목적을 기술

[2. Phase별 영상 소견 요약]
- WAT phase:
- FAT phase:
- INN phase:
- OPP phase:

[3. 세그멘테이션 결과 및 품질 평가]
- 병변 분포 요약 (예: level, side, 범위)
- Dice score 기반의 전반적인 성능 평가
- Overlay/Overlay on Predicted를 기반으로 한 과다 탐지(FP)/미검출(FN) 경향 기술
  (이미지를 실제로 볼 수 없으므로, 파일 경로를 참고한 개념적 설명 수준으로 작성해도 됩니다)

[4. 종합 소견 (연구/기술적 관점)]
- multi-phase를 종합했을 때의 병변 양상에 대한 기술적 해석
- 모델이 잘 동작하는 상황과 어려워하는 상황에 대한 간단한 언급

※ 이 단계는 "초안"이며, 아직 임상 타당성 검토 전 단계입니다.
※ 진단명이나 특정 치료 방침을 단정적으로 제시하지 마세요.
"""

        response = self.model.generate_content(prompt)
        return response.text
