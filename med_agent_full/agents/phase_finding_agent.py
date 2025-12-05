import os
from typing import Dict
from PIL import Image
import google.generativeai as genai


class PhaseFindingAgent:
    """
    WAT / FAT / INN / OPP 4-phase input 이미지를 기반으로
    각 phase별 주요 소견을 생성하는 에이전트.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _analyze_single_phase(self, phase_name: str, image_path: str) -> str:
        """
        단일 phase 이미지에 대해 소견을 생성.
        """
        prompt = f"""
당신은 MRI multi-phase 영상에 대한 기술적 소견을 작성하는 의료 영상 연구용 어시스턴트입니다.
이 이미지는 IVDM 데이터셋에서 추출된 {phase_name} phase 영상입니다.

역할:
- 병변 위치/형태/강도 분포에 대한 기술적 소견을 작성합니다.
- 가능한 경우, 디스크/주변 연부조직/지방 분포 등의 특징을 언급합니다.
- 진단명은 단정적으로 내리지 말고, '의심', '가능성', '소견상 ~로 보임' 수준으로 적습니다.
- 이 내용은 연구·교육용이며 실제 임상 진단이나 치료 결정을 대체할 수 없음을 명시하지 않아도 됩니다
  (최종 리포트 단계에서 별도 경고가 추가됩니다).

출력 형식 (한국어):
[Phase: {phase_name}]
- 신호 특성:
- 의심되는 이상 소견:
- 추가적으로 확인이 필요한 부분:
"""

        img = Image.open(image_path).convert("RGB")
        response = self.model.generate_content([prompt, img])
        return response.text

    def run(self, seg_output: Dict) -> Dict[str, str]:
        """
        SegmentationAgent의 출력(seg_output)을 입력으로 받아,
        phase별 소견 딕셔너리를 반환.
        """
        phase_images: Dict[str, str] = seg_output["phase_images"]

        phase_findings: Dict[str, str] = {}
        for phase_name, img_path in phase_images.items():
            print(f"[PhaseFindingAgent] Analyzing phase: {phase_name} ({img_path})")
            finding = self._analyze_single_phase(phase_name, img_path)
            phase_findings[phase_name] = finding

        return phase_findings
