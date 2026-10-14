"""Generate the reviewable report and static figure from completed P3 runs."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--natural", required=True)
    p.add_argument("--intervention", required=True)
    a = p.parse_args()
    natural_dir, intervention_dir = ROOT / a.natural, ROOT / a.intervention
    n = json.loads((natural_dir / "metrics.json").read_text())
    r = json.loads((intervention_dir / "metrics.json").read_text())
    with open(intervention_dir / "traces.csv", encoding="utf-8") as f:
        traces = list(csv.DictReader(f))
    # Cross-check numerical cost acceleration against the historical tracker,
    # at identical GT times, excluding the truncated excerpt's last frame.
    import yaml
    cfg = yaml.safe_load((natural_dir / "config.yaml").read_text(encoding="utf-8"))
    with open(ROOT / cfg["source"] / "frame_results.csv", encoding="utf-8") as f:
        cached = list(csv.DictReader(f))
    with open(natural_dir / "manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    lookup = {(t["performance"], round(float(t["time_sec"]), 6)): float(t["intact_error_sec"]) for t in traces}
    diffs, cursor = [], 0
    for m in manifest:
        count = int(m["cached_frames"])
        for row in cached[cursor:cursor+count]:
            t = float(row["perf_time_sec"])
            key = (m["performance"], round(t, 6))
            if t < 50 and key in lookup:
                diffs.append(abs(lookup[key] - float(row["online_error_sec"])))
        cursor += count
    audit = dict(compared_frames=len(diffs), max_error_difference_sec=max(diffs),
                 differing_frames=int(np.sum(np.array(diffs)>1e-8)))
    (intervention_dir / "baseline_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Baseline audit: " + json.dumps(audit), flush=True)
    if audit["differing_frames"]:
        raise RuntimeError("accelerated tracker differs from historical baseline; investigate before reporting")
    # First average repeated performances inside each composition at each time.
    grouped = defaultdict(list)
    for row in traces:
        grouped[(row["composition"], float(row["time_sec"]))].append(
            [float(row["intact_error_sec"]), float(row["erased_error_sec"])])
    time_groups = defaultdict(list)
    for (_, t), values in grouped.items():
        time_groups[t].append(np.mean(values, axis=0))
    time = np.array(sorted(time_groups))
    mean = np.array([np.mean(time_groups[t], axis=0) for t in time])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    hs = [2, 5, 10]
    effects, lows, highs = [], [], []
    for h in hs:
        s = n["natural"][f"{h}s"]
        effects.append(s["relative_improvement_percent"])
        lo, hi = np.array(s["bootstrap_ci95"]) / s["baseline_mae_log1p"] * 100
        lows.append(lo)
        highs.append(hi)
    ax[0].axhline(0, color="#8a8a8a", lw=1)
    ax[0].errorbar(hs, effects, yerr=[np.array(effects)-lows, np.array(highs)-effects],
                   fmt="o", capsize=5, color="#2463a5")
    ax[0].set(xticks=hs, xlabel="Forecast horizon (s)", ylabel="MAE improvement (%)",
              title="A. Natural ambiguity: inconclusive")
    ax[0].text(.97, .04, "23 compositions / 40 performances\nComposition bootstrap 95% CI",
                transform=ax[0].transAxes, va="bottom", ha="right", fontsize=9)
    ax[1].axvspan(30, 40, color="#ed835e", alpha=.18, label="Evidence erased")
    ax[1].plot(time, mean[:, 0], color="#2463a5", label="Intact input")
    ax[1].plot(time, mean[:, 1], color="#c65335", label="Erased at 30-40 s")
    ax[1].set(xlim=(20, 55), xlabel="Performance time (s)", ylabel="Mean error (score s)",
              title="B. Real MIDI: imposed evidence loss")
    ax[1].legend(frameon=False, fontsize=9)
    for axis in ax:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    paper = ROOT / "results/paper"
    paper.mkdir(exist_ok=True)
    fig.savefig(paper / "p3_revival.png", dpi=180)
    fig.savefig(paper / "p3_revival.pdf")
    plt.close(fig)
    table = []
    for h in hs:
        s = n["natural"][f"{h}s"]
        table.append(f"| {h}초{' (주 분석)' if h==5 else ''} | {s['relative_improvement_percent']:+.2f}% | "
                     f"{s['mae_improvement_log1p']:+.6f} | [{s['bootstrap_ci95'][0]:+.6f}, "
                     f"{s['bootstrap_ci95'][1]:+.6f}] | {s['shift_p_upper']:.3f} | "
                     f"{s['improved_compositions']}/{s['compositions']} |")
    synth = n["intervention"]
    primary = n["natural"]["5s"]
    text = f"""# P3 되살리기 결과 — 2026-09-17

**판정: 원래의 자연 연주 P3는 여전히 미판정. 대신 ‘위치 단서 제거 → 이후 추적 오차 증가’는
실제 ASAP MIDI 입력에 대한 짝지은 개입으로 검증했다.** 이는 인위적 개입의 효과이며
자연 모호성이 실제 위치 증거 상실을 유발했다는 매개 사슬 전체의 증명은 아니다.

## 실제 연주 소재 개입: {'통과' if r['pass'] else '미달'}

- {r['performances']}개 연주, 독립 작품 {r['compositions']}개. 모든 연주 포함, 오차 기반 필터 없음.
- 30~40초 구간에서 비용행의 위치별 차이만 제거. 같은 입력·정답·초기상태의 intact 실행과 비교.
- 개입 중 오차: intact **{r['intact_error_sec']:.3f}초**, erasure **{r['erased_error_sec']:.3f}초**.
- 짝지은 평균 증가: **{r['during_effect_sec']:+.3f}초**, 작품 bootstrap 95% CI
  **[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]초**.
- 증가 방향 작품 **{r['positive_compositions']}/{r['compositions']}**. 개입 전 최대 차이 **{r['pre_max_difference_sec']:.1f}초**.
- 관측 복원 후 40~50초 차이: **{r['after_effect_sec']:+.3f}초**,
  95% CI [{r['after_ci95'][0]:+.3f}, {r['after_ci95'][1]:+.3f}]초.
- onset은 오차를 보지 않고 고정했다. 따라서 원래 P3의 결과 기반 사건 선택 문제를 피한다.
- 단위는 원래 파이프라인과 같은 **악보 시간의 초**이며 연주 시간의 초와 구별한다.

## 자연 모호성의 추가 예측력: {'통과' if n['natural_pass'] else '미달'}

{primary['landmarks']}개 고정 시점에서 과거에 완료된 모호성 창만 사용했다.
현재 오차·2/5초 전 오차·경과 시간·직전 템포를 기준모델에 넣고, 모호성을 추가했다.
같은 작품의 모든 연주를 함께 제외하는 교차검증으로 비교했다.

| 예측 시차 | MAE 상대 개선 | MAE 개선(log1p) | 95% CI(log1p) | shift p | 개선 작품 |
|---|---:|---:|---|---:|---:|
{chr(10).join(table)}

5초에서 shift 대조는 낮은 p를 보였지만 작품 간 일반화의 CI가 0을 포함한다.
사전에 고정한 결합 기준을 통과하지 못했으므로 이를 성공으로 보고하지 않는다.
2·10초 보조 분석도 모두 공개하며 유리한 시차만 골라 쓰지 않는다.
이는 이 고정된 선형 예측모델의 실패이며 모든 가능한 시간 관계가 없다는 증명은 아니다.

## 합성 통제 실험

32개 seed × 템포 -20/0/+20% × intact/erased. 위치 단서 제거는 15~25초.
±20% 방향을 seed 안에서 평균한 오차 증가 **{synth['paired_effect_sec']:.3f}초**,
95% CI [{synth['bootstrap_ci95'][0]:.3f}, {synth['bootstrap_ci95'][1]:.3f}]초.
양의 효과 {synth['positive_seeds']}/32, 개입 전 차이 0. 템포 변화가 0인 조건도
증가 {synth['by_speed']['0.0']['during_effect_sec']:.3f}초여서,
‘드리프트에는 반드시 템포 교란이 필요하다’는 주장은 하지 않는다.

## 논문에서 쓸 주장과 남은 연결고리

쓸 수 있는 주장: “평가한 OLTW에서 위치별 관측 비용의 차이를 인위적으로 제거하면,
동일한 실제 MIDI 연주 입력의 개입 후 정렬 오차가 평균적으로 증가했다.”

쓰면 안 되는 주장: “자연 모호성 증가가 위치 상실에 선행하고 이것이 자연 드리프트를
유발함을 입증했다.” 자연 모호성 → 실제 관측 증거 약화 연결은 아직 검증되지 않았다.

후속으로는 과거 연주 창의 올바른 위치/경쟁 위치 간 관측 비용 margin을 직접 기록하고,
그 증거가 약해지는 정도를 예측하는 악보 모호성·연주 변형을 새 작품에서 검증하는 것이
필요하다. 현재 오차로 사건을 정하거나 GAP을 독립 사건으로 취급하지 않는다.
실제 마이크 오디오, 다른 tracker, 새로운 곡에서의 확인도 아직 남아 있다.

## 한계와 정정

- 악보 자기유사도는 잠재적 모호성이고 관측 증거 자체가 아니다.
- 기존 창은 앞으로 2/8초를 포함했고 중심 평활도 미래를 혼합했다. 새 분석은 창 끝을
  이미 지난 위치에 붙였다. 단 GT와 전곡 길이 기반 템포 정규화는 oracle 조건이다.
- 기존 40개 연주는 작품 23개다. 이번 CI는 프레임 수가 아니라 작품 수를 기준으로 한다.
- 기존 상관비 1.47≈미달폭비 1.42만으로 편향이 ‘정량 증명’됐다는 설명은 과도했다.
- 자연 예측 CI는 CV 학습셋 공유를 무시한 근사 bootstrap이며 shift는 비정상성 한계가 있다.
- 비용행 평균 치환은 위치 변별력과 함께 비용 분포도 바꾼다. 음향적 자연 모호성의
  정밀한 모사나 특정 음악적 조작의 효과로 해석할 수 없다.
- 실제 MIDI 개입은 합성 결과 후 설계한 후속 탐색이며 사전등록/새 독립 데이터가 아니다.
- 개입 전 차이가 0인 것은 인과적 실행의 구현 점검이다. 그것만으로 양의 시간 지연을
  추정하거나 자연 발생 사건의 시간순서를 증명한 것은 아니다.
- 개입 없는 재실행을 기존 P2 원본의 {audit['compared_frames']}개 프레임과 대조했다.
  최대 오차 차이 {audit['max_error_difference_sec']:.3g}초 (baseline_audit.json).

## 재현

```text
python scripts/revive_p3.py --config configs/p3_revival.yaml
python scripts/p3_asap_intervention.py --config configs/p3_asap_intervention.yaml
python scripts/report_p3_revival.py --natural {a.natural} --intervention {a.intervention}
python -m pytest tests -q
```

- 자연/합성 산출물: `{a.natural}` (설정, 해시, manifest, landmark, 예측, 작품별 효과, 개입 trace).
- 실제 MIDI 개입: `{a.intervention}` (설정, MIDI·코드 해시, 연주/작품별 효과, 전체 trace).
- 설계: [P3_REVIVAL_PROTOCOL.md](P3_REVIVAL_PROTOCOL.md).
- 그림: [PNG](results/paper/p3_revival.png), [PDF](results/paper/p3_revival.pdf).
- 확신도: 실행 수치와 개입 전 동일성 **확정**. 평가한 추적기의 단서 제거 효과 **잠정**
  (현재 코호트/조건의 탐색 결과). 자연 모호성 매개 사슬 **미판정**.
"""
    (ROOT / "P3_REVIVAL_REPORT.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
