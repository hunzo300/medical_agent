import os
import google.generativeai as genai


class ReportFinalizerAgent:
    """
    리포트 초안을 입력으로 받아,
    임상 타당성/표현/안전성 관점에서 검토 후 최종 리포트를 작성하는 에이전트.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def run(self, draft_report: str) -> str:
        """
        초안 리포트를 입력으로 받아 최종 리포트를 반환.
        """
        prompt = f"""
당신은 신중한 의료 영상 리포트 감수자 역할을 합니다.

아래의 리포트 초안을 검토하여,
- 임상적으로 오해의 소지가 있는 표현을 완화하거나 수정하고,
- 과도하게 단정적인 진단 표현을 '의심', '~로 보임', '~ 가능성' 등으로 완화하며,
- 연구/교육용 리포트라는 점을 분명히 하고,
- 전체 흐름을 자연스럽게 다듬어 최종 리포트를 작성하세요.

[리포트 초안]
--------------------
{draft_report}
--------------------

최종 리포트 형식 (한국어):

[1. 검사 개요]

[2. Phase별 영상 및 세그멘테이션 소견]

[3. 세그멘테이션 모델 성능 및 한계]

[4. 종합 해석 (연구/기술적 관점)]

[5. 한계 및 주의사항]
- 데이터/모델/세그멘테이션의 한계
- 임상 적용 시 유의해야 할 점을 일반적인 수준에서 서술

[6. 비진단적 권고]
- '실제 환자 진료에서는 반드시 영상의학과 전문의 및 담당 주치의의 판단이 필요하다'는 내용을 포함

마지막에 반드시 다음과 유사한 경고 문구를 추가하세요:

"이 리포트는 연구 및 교육 목적의 참고용 자료로, 실제 진단이나 치료 결정을 대체할 수 없습니다.
환자의 임상 상태에 대한 평가는 반드시 의료 전문가와의 직접 상담을 통해 이루어져야 합니다."
"""

        response = self.model.generate_content(prompt)
        return response.text
