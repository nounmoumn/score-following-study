"""Reviewable full P3 extension report, with all prespecified outcomes."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    args = p.parse_args()
    out = ROOT/args.result
    m = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    tests = {t["name"]:t for t in m["primary_tests"]}
    with open(out/"manifest.csv", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    with open(out/"diagnostics.csv", encoding="utf-8") as f:
        diagnostics = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    colors = {"oltw":"#2463a5", "bayes":"#d37827"}
    for ti, tracker in enumerate(["oltw", "bayes"]):
        values = m["natural"][tracker]["5s"]
        for j, model in enumerate(["ambiguity", "evidence", "joint"]):
            r = values[model]
            mean = r["relative_improvement_percent"]
            lo, hi = np.array(r["ci95"])/r["baseline_mae"]*100
            axes[0,0].errorbar(j+(ti-.5)*.14, mean, yerr=[[mean-lo],[hi-mean]],
                               fmt="o", capsize=4, color=colors[tracker], label=tracker.upper() if j==0 else None)
    axes[0,0].axhline(0, color="gray", linewidth=1)
    axes[0,0].set(xticks=range(3), xticklabels=["Score ambiguity", "Live evidence", "Both"],
                  ylabel="5s forecast MAE improvement (%)", title="New scores: extra predictive value")
    axes[0,0].legend(frameon=False)
    compare = ["target_vs_scattered", "prevention_vs_sham", "recovery_vs_sham"]
    for ti, tracker in enumerate(["oltw", "bayes"]):
        for j, name in enumerate(compare):
            r = m["intervention"]["holdout"][tracker][name]
            mean, (lo, hi) = r["mean"], r["ci95"]
            axes[0,1].errorbar(j+(ti-.5)*.14, mean, yerr=[[mean-lo],[hi-mean]],
                               fmt="o", capsize=4, color=colors[tracker], label=tracker.upper() if j==0 else None)
    axes[0,1].axhline(0, color="gray", linewidth=1)
    axes[0,1].set(xticks=range(3), xticklabels=["Distractor vs\nscattered", "Early cue vs\nsham", "Late cue vs\nsham"],
                  ylabel="Paired contrast (score seconds)", title="Specificity / rescue contrasts (positive expected)")
    axes[0,1].legend(frameon=False)
    styles = {"intact":("#2463a5", "Intact"), "targeted":("#c65335", "Distractor"),
              "scattered":("#777777", "Scattered control"), "prevention":("#209971", "Early cue"),
              "recovery":("#8153a3", "Late cue")}
    for ti, tracker in enumerate(["oltw", "bayes"]):
        ax = axes[1, ti]
        curves = {c:[] for c in styles}
        for case in cases:
            if case["cohort"]!="holdout":
                continue
            with np.load(out/"cases"/(case["case_id"]+".npz")) as data:
                time = data["time_sec"]+.1
                for condition in curves:
                    curves[condition].append(abs(data[condition+"__"+tracker]-data["truth_frames"])*float(data["hop_score_sec"]))
        ax.axvspan(30, 40, color="#ed835e", alpha=.12)
        for condition, (color, label) in styles.items():
            ax.plot(time, np.mean(curves[condition], axis=0), color=color, label=label, linewidth=1.6)
        ax.set(xlim=(25, 55), xlabel="Completed performance time (s)", ylabel="Mean error (score s)",
               title=tracker.upper()+": 40 held-out scores")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    paper = ROOT/"results/paper"
    fig.savefig(paper/"p3_extension.png", dpi=180)
    fig.savefig(paper/"p3_extension.pdf")
    plt.close(fig)
    labels = {
        "A_to_weakness":"악보 모호성 ↔ 실제 증거 약화 (rho)",
        "evidence_forecast":"실제 증거의 5초 예측 MAE 개선 (log1p)",
        "target_vs_intact":"경쟁 경로 − 원본 오차 (초)",
        "target_vs_scattered":"경쟁 경로 − 같은 크기 교란 (초)",
        "prevention_vs_target":"경쟁 경로 − 조기 단서 (초)",
        "prevention_vs_sham":"비식별 대조 − 조기 식별 단서 (초)",
        "recovery_vs_target":"경쟁 경로 − 후기 단서 (초)",
        "recovery_vs_sham":"비식별 대조 − 후기 식별 단서 (초)",
    }
    table = []
    for t in m["primary_tests"]:
        name = t["name"]
        tracker = "공통"
        for tr in ["oltw", "bayes"]:
            if name.startswith(tr+"_"):
                tracker, name = tr.upper(), name[len(tr)+1:]
                break
        table.append(f"| {tracker} | {labels[name]} | {t['mean']:+.5f} | [{t['ci95'][0]:+.5f}, {t['ci95'][1]:+.5f}] | {t['holm_p']:.4f} | {'통과' if t['passed'] else '미달'} |")
    status_rows = []
    for tracker in ["oltw", "bayes"]:
        flags = []
        for pair in [("target_vs_intact", "target_vs_scattered"),
                     ("prevention_vs_target", "prevention_vs_sham"),
                     ("recovery_vs_target", "recovery_vs_sham")]:
            flags.append("통과" if all(tests[tracker+"_"+v]["passed"] for v in pair) else "미달")
        status_rows.append(f"| {tracker.upper()} | {'통과' if tests[tracker+'_evidence_forecast']['passed'] else '미달'} | {' | '.join(flags)} |")
    forecast = []
    for tracker in ["oltw", "bayes"]:
        for h in ["2s", "5s", "10s"]:
            values = m["natural"][tracker][h]
            forecast.append(f"| {tracker.upper()} | {h} | {values['ambiguity']['relative_improvement_percent']:+.2f}% | {values['evidence']['relative_improvement_percent']:+.2f}% | {values['joint']['relative_improvement_percent']:+.2f}% |")
    baseline = []
    for cohort in ["development", "holdout"]:
        for tracker in ["oltw", "bayes"]:
            r = m["baseline"][cohort][tracker]
            baseline.append(f"| {cohort} | {tracker.upper()} | {r['mean_piece_median_sec']:.3f} | {r['median_piece_median_sec']:.3f} |")
    a_result = tests["A_to_weakness"]
    passed = [t["name"] for t in m["primary_tests"] if t["passed"]]
    report = f"""# P3 확장 실험 결과

**요청한 네 축을 모두 실행했다:** 실제 위치 증거 측정, 경쟁 위치 혼동 개입,
위치 식별 단서의 예방/회복, 새 악보와 별도 확률 추정기 검증.
새 검증군에서 주 검정 15개 중 **{len(passed)}개**가 고정한 기준을 통과했다.
통과 항목: {', '.join(passed) or '없음'}.
이 개수 자체가 전체 P3의 성공률이나 전체 자연 매개 인과 사슬의 증명은 아니다.

## 표본과 설계

- 기존 40연주/23악보에서만 예측모델 학습. 기존 악보와 겹치지 않는 **40개 악보의 40개 연주**에서 평가.
- 악보 단위 분리이며 같은 소나타/모음곡의 다른 악장이나 같은 작곡가·연주자가 겹칠 수 있다.
  새 녹음/새 데이터셋이 아니라 기존 ASAP의 이번 P3에 미사용 악보다.
- 모두 처음 90초, landmark 20~75초(5초 간격), 연주당 12시점.
  새 검증군의 자연 예측은 tracker당 480시점. 오차 기반 선별 없음.
- 7조건 × 2추적기 × 80연주 = **1,120회 경로 계산**.
- OLTW와 고정 모수의 인과적 Bayesian filter를 별도로 평가했다. 후자는 검증용 구현으로,
  출판된 HMM의 재현/최적 baseline이라고 부르지 않는다.
- [설계](P3_EXTENSION_PROTOCOL.md), `configs/p3_extension.yaml`에 기준 고정.

## 핵심 연결과 판정

악보 모호성 ↔ 실제 증거 약화의 새 악보별 Spearman rho 평균은 **{a_result['mean']:+.4f}**,
95% CI **[{a_result['ci95'][0]:+.4f}, {a_result['ci95'][1]:+.4f}]**,
Holm p={a_result['holm_p']:.4f}: **{'통과' if a_result['passed'] else '미달'}**.
이는 정답/경쟁 위치의 관측 유사도 차이를 직접 측정한 연결이며,
GT로 정답과 과거 템포를 표시하는 oracle 관찰 분석이다.

| 추적기 | 증거의 미래 예측 | 경쟁 경로 특이적 영향 | 조기 단서 | 후기 단서 |
|---|---|---|---|---|
{chr(10).join(status_rows)}

‘특이적 영향’은 원본보다 오차가 늘고 같은 크기의 흩어진 교란보다도 늘어야 통과한다.
‘단서’는 단서 없는 경쟁 경로와 동일 평균 비용의 비식별 단서 양쪽보다 개선돼야 통과한다.
모든 개별 판정은 예상방향, CI 하한>0, 주 검정 15개 Holm p≤.05를 모두 요구한다.

## 주 검정 전체 — 새 악보 40개

| 추적기 | 비교 | 작품 평균 효과 | 주변 95% CI | Holm p | 판정 |
|---|---|---:|---|---:|---|
{chr(10).join(table)}

효과 단위를 섞어 비교하지 말 것. 증거 예측의 효과는 log1p MAE 차이,
개입의 효과는 **악보 시간의 초**다. CI는 작품 bootstrap 10,000회이고 동시구간은 아니다.
sign-flip p는 작품 효과의 귀무 대칭성에 의존하는 근사 검정이다.

## 보조 시차를 포함한 예측 결과 전체

| 추적기 | 시차 | +악보 모호성 | +실제 위치 증거 | +둘 다 |
|---|---|---:|---:|---:|
{chr(10).join(forecast)}

수치는 기준모델 대비 MAE 개선율(음수는 악화). 5초의 증거 단독 추가만 주 판정이다.
다른 시차/조합의 좋은 결과로 주 판정을 뒤집지 않는다.

## 추적기 baseline 정확도

| 코호트 | 추적기 | 작품별 중앙오차의 평균(초) | 작품별 중앙오차의 중앙값(초) |
|---|---|---:|---:|
{chr(10).join(baseline)}

같은 개입의 효과 크기가 두 추적기에서 달라도 이 baseline과 함께 읽어야 한다.

## 구현 검증 및 조작 검사

- 개입 전 최대 경로 차이 **{m['audits']['pre_max_difference_frames']}프레임**.
- 정답 위치 주변 비용은 경쟁 개입에서 그대로 유지(모든 프레임 assertion).
- targeted/scattered 행별 총 비용 차이 최대 **{m['audits']['max_matched_budget_difference']:.3g}**.
- 국소 오답 후보의 순간 비용 개선량 평균 **{m['audits']['mean_weakness_manipulation']:.6f}**.
  이 조작 검사와 자연 연주의 과거창 증거는 다른 통계다.
- 원본 P2 **{m['audits']['historical_frames']}프레임** 대조,
  최대 baseline 오차 차이 **{m['audits']['max_historical_error_difference_sec']:.3g}초**.
- OLTW 가속 경로 일치, 과거 입력만 사용, 비용 예산 일치, maxsim 수식 일치,
  두 추적기의 인과성, Holm 계산을 단위 테스트로 검증했다.

## 주장 범위와 남은 한계

### 사후 강건성 진단 및 작업 범위

`python scripts/audit_p3_extension_results.py --result {args.result}`의
`posthoc_robustness.json`도 함께 읽을 것. Bayesian 후기 단서의 개선 중앙값은0,
14/40악보 개선·26/40변화 없음이며, Schumann/Kreisleriana_5가 총효과의85.2%를 차지한다.
그 악보 제외 시 평균+0.5017초, 95% CI [0.0615,1.0763]. 따라서 평균3.31초를
전형적 연주의 회복량으로 해석하지 않는다. 원래 판정은 유지하고 한계를 추가한 사후 진단이다.
모호성↔증거 약화는 작곡가11명 균등 평균도+0.3105, CI [0.1355,0.4790]였다.

P1·P2는 다시 검정하지 않았다. 기존 P2 cache 대조는 구현 동일성 점검이다.
사용자 지적에 따라 후속 실험 확대는 중단했다. 실제 기타 녹음의 추가 개입은
제안/경로 확인까지만 했으며 실행하지 않았다. 이번에 완료된 P3 범위만 보고한다.

자연 증거의 예측 결과는 시간상 먼저 이용할 수 있는 정보의 추가 예측력이며
자연 발생 매개 인과효과의 식별이 아니다. 악보 모호성과 위치 증거가 일부 음악 특징을
공유하고, GT 기반 과거 tempo warp도 포함하므로 기계적 연관/교란 가능성이 남는다.

경쟁 개입은 비용공간에서 정답 경로를 보존하며 가짜 경로를 만드는 oracle 조작이다.
실제 악보/음향을 바꾼 자연 음악 실험이 아니다. 흩어진 대조는 L1와 평균 비용을 맞추지만
모든 비용 분포/공간적 구조를 동일하게 맞추지는 않는다.

단서는 GT 위치를 알려주는 oracle cue다. 30~32초는 고정 onset 직후의 조기 단서이지
모든 연주에서 드리프트 전에 들어갔다고 보장하지 않는다. 38~40초 단서의 실패는
탐색 범위 이탈이나 추적기의 누적 상태 때문일 수도 있으며 정보 자체가 무효라는 뜻은 아니다.

전체 악보 후보, GT 주변 후보, OLTW 현재 행 후보에서의 증거를 모두 CSV에 남겼다.
주 분석은 GT 주변 후보였으며 actual-accessible 후보를 보고 유리한 정의로 바꾸지 않았다.
작품/악장별 CI는 동일 작곡가·모음곡·연주자에 따른 상관을 완전히 처리하지 못한다.
실제 마이크 녹음·다른 데이터셋·공개된 표준 확률 추적기에서의 외적 검증은 남아 있다.

## 재현과 산출물

```text
python scripts/extend_p3.py --config configs/p3_extension.yaml
python scripts/report_p3_extension.py --result {args.result}
python -m pytest tests -q
```

결과: `{args.result}`.
`manifest.csv`, `eligibility_exclusions.csv`, `config.yaml`, `source_snapshot/`, `provenance.json`,
`landmarks.csv`, `performance_results.csv`, `diagnostics.csv`, `natural_correlations.csv`,
`holdout_predictions.csv`, `composition_effects.csv`, `metrics.json`, `cases/*.npz`.
그림: [PNG](results/paper/p3_extension.png), [PDF](results/paper/p3_extension.pdf).

확신도: 실행 수치/구현 검증 **확정**. 새 악보로의 조건부 일반화는 **잠정**.
전체 자연 매개 인과 사슬은 **미판정**. 통과/미달을 동일하게 보존했다.
"""
    (ROOT/"P3_EXTENSION_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
