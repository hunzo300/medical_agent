# medical_agent

````markdown
# Medical Multi-Agent Pipeline for IVDM Multi-Phase Segmentation

> ⚠️ **연구·교육용 데모 코드입니다.**  
> 실제 환자 진료, 임상 의사결정, 치료 판단에는 **절대 사용하면 안 됩니다.**

이 프로젝트는 IVDM multi-phase MRI (WAT / FAT / INN / OPP) 데이터를 대상으로,

1. **Multi-phase 세그멘테이션 수행**
2. **Phase별 영상 소견 자동 작성**
3. **세그멘테이션 결과 + 소견을 종합한 리포트 초안 생성**
4. **임상 타당성을 검토한 최종 리포트 생성**

까지의 전 과정을 **여러 개의 “에이전트(Agent)”로 나누어 구성한 파이프라인**입니다.  
텍스트 생성에는 Google **Gemini 2.0 Flash** API를 사용합니다.

---

## 1. 전체 구조

폴더 구조 (핵심만):

```text
med_agent_full/
  agents/
    segment_agent/
      segmentation_agent.py     # 세그멘테이션 에이전트 (CoMedSAM 기반)
      segment_anything_CoMed.py # SAM 백본 래퍼
    phase_finding_agent.py      # Phase별 영상 소견 생성 에이전트
    report_draft_agent.py       # 세그멘테이션 + 소견 기반 리포트 초안 에이전트
    report_finalizer_agent.py   # 임상 타당성 검토 + 최종 리포트 에이전트
  README.md                     # (이 파일)
````

파이프라인 논리는 대략 다음 순서로 동작합니다.

1. **SegmentationAgent**

   * IVDM multi-phase MRI (`WAT/FAT/INN/OPP`) 4-channel npy 이미지를 입력으로 받아
     CoMedSAM 기반 multi-encoder 모델로 세그멘테이션 수행
   * 최종 Dice 점수, GT / Predicted / Overlay 이미지 파일 경로를 정리하여 반환

2. **PhaseFindingAgent**

   * SegmentationAgent가 만들어 낸 **phase별 입력 이미지 (WAT/FAT/INN/OPP)** 경로나 요약 정보를 바탕으로
   * Gemini 모델에게 각 phase의 특징적인 영상 소견을 **텍스트(한국어)**로 생성하도록 요청
   * 예:

     * WAT: 지방 조직 대비, 병변 대비/명암, 레벨 정보
     * FAT: 지방 억제 후 병변 대비 변화, 부종 여부 등

3. **ReportDraftAgent**

   * `seg_output` (Dice score, mask/overlay 경로 등)

     * `phase_findings` (phase별 텍스트 소견) 을 모두 입력으로 받아
   * Gemini에 다음과 같은 형태의 **리포트 초안** 작성을 요청:

     * [1. 검사 개요]
     * [2. Phase별 영상 소견 요약]
     * [3. 세그멘테이션 결과 및 품질 평가]
     * [4. 종합 소견 (연구/기술적 관점)]
   * 이 단계에서는 **세그멘테이션 성능 및 multi-phase 영상 해석을 기술적으로 서술**하며,
     진단명을 확정하거나 치료 방침을 직접 언급하지 않도록 설계됨.

4. **ReportFinalizerAgent**

   * 초안 리포트를 입력으로 받아,

     * 임상적으로 과도한 표현(“확진”, “치료 필요” 등)을 완화하고
     * 표현을 더 일관되고 읽기 좋게 정리
     * 연구/교육용 리포트라는 점을 다시 강조
   * 최종적으로 “인간 전문가가 검토하기에 앞서 보는 초벌 자동 리포트” 수준의 텍스트를 생성

---

## 2. SegmentationAgent (CoMedSAM 기반 Multi-Phase 세그멘테이션)

**파일:**
`med_agent_full/agents/segment_agent/segmentation_agent.py`
(+ 백본: `segment_anything_CoMed.py` 및 IVDM3Seg 관련 코드)

### 역할

* IVDM 데이터셋의 `imgs/` 폴더에 존재하는 **4-channel npy (WAT/FAT/INN/OPP)** 를 불러와,
* 각 phase를 독립적인 encoder에 넣어 임베딩을 얻고,
* 이들을 **Conv + Transformer 기반 Multi-Phase Fusion**으로 결합한 뒤,
* SAM mask decoder를 이용해 최종 마스크를 얻는 **CoMedSAM 변형 모델**을 사용합니다.

### 주요 입·출력

* 입력:

  * `data_root`: IVDM npy 데이터셋 루트 경로

    * `data_root/imgs/*.npy`  : 4-channel 이미지 (WAT/FAT/INN/OPP)
    * `data_root/gts/*.npy`   : GT 마스크
  * `file_prefix`: 예) `"15-13"`

    * `gts/15-13_*.npy` 에 해당하는 모든 slice를 하나의 case로 처리

* 출력(`seg_output` 딕셔너리 예시):

  ```python
  {
      "case_id": "15-13",
      "dice_score": 0.9123,
      "phase_images": {
          "WAT": "ablation_images/15-13_dice_0.9123/input_1.png",
          "FAT": "ablation_images/15-13_dice_0.9123/input_2.png",
          "INN": "ablation_images/15-13_dice_0.9123/input_3.png",
          "OPP": "ablation_images/15-13_dice_0.9123/input_4.png",
      },
      "masks": {
          "gt_final": "ablation_images/15-13_dice_0.9123/gt_final.png",
          "pred_final": "ablation_images/15-13_dice_0.9123/predicted_final.png",
          "overlay": "ablation_images/15-13_dice_0.9123/overlay.png",
          "overlay_on_predicted": "ablation_images/15-13_dice_0.9123/overlay_on_predicted.png",
      }
  }
  ```

이 결과는 이후 **PhaseFindingAgent → ReportDraftAgent → ReportFinalizerAgent**에서 그대로 사용됩니다.

---

## 3. PhaseFindingAgent

**파일:**
`med_agent_full/agents/phase_finding_agent.py`

### 역할

* `SegmentationAgent`가 출력한 `seg_output["phase_images"]` 정보를 기반으로
* 각 phase(WAT/FAT/INN/OPP)에 대해 **영상 의학적 소견 텍스트**를 생성
* 내부적으로 Google Gemini 2.0 Flash API를 사용하여,
  “IVDM multi-phase lumbar MRI에서 각 phase가 강조하는 조직 특성 및 병변 양상”을
  연구/교육 관점에서 기술하도록 프롬프트를 구성

### 출력 예시

```python
{
    "WAT": "... WAT phase에서 지방과 연부 조직 대비가 어떻게 보이는지, 병변이 어느 레벨에서 관찰되는지 ...",
    "FAT": "... fat-suppressed 상태에서 병변과 주변 부종이 어떻게 대비되는지 ...",
    "INN": "... INN phase에서 디스크 및 뼈 구조 대비와 병변 신호 변화 ...",
    "OPP": "... OPP phase에서 물/지방 상호작용과 병변 신호의 특징 ...",
}
```

---

## 4. ReportDraftAgent

**파일:**
`med_agent_full/agents/report_draft_agent.py`

### 역할

* 입력:

  * `seg_output` (Dice score, 마스크 및 overlay 경로)
  * `phase_findings` (각 phase별 텍스트 소견)
* 위 두 정보를 모두 **프롬프트에 포함**하여,
  Gemini 2.0 Flash 모델로부터 아래 형식의 **리포트 초안**을 생성합니다.

리포트 구조:

1. 검사 개요

   * IVDM multi-phase MRI 검사 목적 및 배경
2. Phase별 영상 소견 요약

   * WAT / FAT / INN / OPP 각각에 대한 핵심 포인트
3. 세그멘테이션 결과 및 품질 평가

   * Dice score 기반 성능
   * Overlay/Overlay on Predicted를 활용한 FP/FN 경향 설명 (개념적 수준)
4. 종합 소견 (연구/기술적 관점)

   * multi-phase를 종합한 병변 양상의 기술적 해석
   * 모델이 강점/약점을 보이는 상황에 대한 간단한 언급

> ⚠️ 이 단계에서는 여전히 **진단 확정이나 치료 방침 제시는 하지 않도록 설계**되어 있습니다.

---

## 5. ReportFinalizerAgent

**파일:**
`med_agent_full/agents/report_finalizer_agent.py`

### 역할

* `ReportDraftAgent`가 생성한 초안 리포트를 입력으로 받아,

  * 표현을 더 자연스럽고 일관되게 정리하고,
  * 지나치게 단정적인 표현, 임상적으로 오해될 수 있는 문장을 완화
  * “연구/교육용 자동 생성 리포트”라는 점을 다시 강조
* 최종적으로 **사람 전문가가 검토하기 좋은 형태의 정제된 리포트 텍스트**를 생성합니다.

---

## 6. 환경 설정 및 실행 방법 (요약)

### 6.1. 환경 변수 (.env)

프로젝트 루트(예: `ML_AL_Project/`)에 `.env` 파일 생성:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

파이썬 코드에서 `python-dotenv`로 자동 로드:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 6.2. Notebook에서 실행 예시

```python
import autorootcwd
from pathlib import Path
import sys
from dotenv import load_dotenv
import os

project_root = Path(".").resolve()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

load_dotenv()

# ipykernel의 --f 인자 충돌 방지를 위해 임시 argv 정리
import sys
argv_backup = sys.argv.copy()
sys.argv = [argv_backup[0]]

from med_agent_full.agents.segment_agent.segmentation_agent import SegmentationAgent
from med_agent_full.agents.phase_finding_agent import PhaseFindingAgent
from med_agent_full.agents.report_draft_agent import ReportDraftAgent
from med_agent_full.agents.report_finalizer_agent import ReportFinalizerAgent

sys.argv = argv_backup

DEFAULT_DATA_ROOT = "/mnt/sda/minkyukim/CoMed-sam_dataset/IVDM_/ivdm_npy_test_dataset_1024image"

def run_ivdm_agent_pipeline_in_notebook(file_prefix: str, data_root: str = DEFAULT_DATA_ROOT):
    seg_agent = SegmentationAgent(data_root=data_root)
    seg_output = seg_agent.run(file_prefix=file_prefix)

    phase_agent = PhaseFindingAgent()
    phase_findings = phase_agent.run(seg_output)

    draft_agent = ReportDraftAgent()
    draft_report = draft_agent.run(seg_output, phase_findings)

    final_agent = ReportFinalizerAgent()
    final_report = final_agent.run(draft_report)

    return seg_output, phase_findings, draft_report, final_report
```

---

## 7. 주의사항

* 본 코드는 **연구/교육용**으로,

  * 모델 성능 검증,
  * multi-phase 세그멘테이션 파이프라인 실험,
  * LLM 기반 리포트 자동 생성 연구
    등을 돕기 위한 데모입니다.
* **실제 환자 진단, 치료 결정, 임상 보고서 작성에는 절대 사용하면 안 됩니다.**
  항상 사람 의료 전문가의 판단이 최종적이어야 합니다.

---

```

원하는 느낌/톤(예: 더 포멀하게, 한국어 100%, 영어 100%, bilingual 등)으로 다시 다듬고 싶으면 말해줘.  
특정 agent 하나만 따로 상세 설명 섹션 추가하는 것도 가능해!
::contentReference[oaicite:0]{index=0}
```
