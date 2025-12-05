import autorootcwd
import os
import argparse

from med_agent_full.agents.segment_agent.segmentation_agent import SegmentationAgent
from med_agent_full.agents.phase_finding_agent import PhaseFindingAgent
from med_agent_full.agents.report_draft_agent import ReportDraftAgent
from med_agent_full.agents.report_finalizer_agent import ReportFinalizerAgent


def main():
    parser = argparse.ArgumentParser(description="IVDM Multi-phase Medical Agent Pipeline")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/sda/minkyukim/CoMed-sam_dataset/IVDM_/ivdm_npy_test_dataset_1024image",
        help="IVDM npy 데이터셋 루트 경로",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="15-13",
        help="케이스 prefix (예: 15-13)",
    )
    args = parser.parse_args()

    # 0. 환경 체크
    if os.getenv("GOOGLE_API_KEY") is None:
        raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")

    # 1. Segmentation Agent
    seg_agent = SegmentationAgent(data_root=args.data_root)
    seg_output = seg_agent.run(file_prefix=args.file_prefix)

    if seg_output is None:
        print("[Pipeline] Segmentation failed or no matching files.")
        return

    # 2. Phase Finding Agent (phase별 소견)
    phase_agent = PhaseFindingAgent()
    phase_findings = phase_agent.run(seg_output)

    # 3. Report Draft Agent (초안 리포트 생성)
    draft_agent = ReportDraftAgent()
    draft_report = draft_agent.run(seg_output, phase_findings)

    # 4. Report Finalizer Agent (임상 타당성 검토 + 최종 리포트)
    final_agent = ReportFinalizerAgent()
    final_report = final_agent.run(draft_report)

    # 5. 결과 출력 및 저장
    output_dir = seg_output["output_dir"]
    draft_path = os.path.join(output_dir, "report_draft.txt")
    final_path = os.path.join(output_dir, "report_final.txt")

    with open(draft_path, "w", encoding="utf-8") as f:
        f.write(draft_report)

    with open(final_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"[Pipeline] Draft report saved to: {draft_path}")
    print(f"[Pipeline] Final report saved to: {final_path}")
    print("\n========== 최종 리포트 ==========\n")
    print(final_report)


if __name__ == "__main__":
    main()
