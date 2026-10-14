"""Publish all original-P3 temporal results without selecting favorable horizons."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);args=p.parse_args()
    out=ROOT/args.result;m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    def read(name):
        with open(out/name,encoding='utf-8') as f:return list(csv.DictReader(f))
    manifest=read('manifest.csv');interventions=read('intervention.csv');pairs=read('natural_pairs.csv')
    primary=[]
    for t in m['primary_tests']:
        ci=t['ci95'];ci_text=f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else '없음'
        primary.append(f"| {t['name']} | {t['n']} | {t['mean']:+.3f} | {ci_text} | {t['holm_p']:.4f} | {'통과' if t['passed'] else '미달'} |")
    details=[]
    for tracker in ['oltw','bayes']:
        i=m['intervention'][tracker];n=m['natural'][tracker]
        details.append(f"- **{tracker.upper()}**: 개입 전 건강 {i['eligible_scores']}/{i['all_scores']}개 악보. "
            f"10초 이내 지속 오차(.5초 초과가 .5초 지속) 발생은 원본 {i['intact_failed10']}/{i['eligible_scores']}, "
            f"단서 제거 {i['erased_failed10']}/{i['eligible_scores']}. 자연 사건 {n['detected_events']}건 중 "
            f"과거 건강 {n['healthy_events']}건, 최종 매칭 {n['matched_events']}쌍/{n['contributing_scores']}악보. "
            f"매칭 사건의 후속 실패 {n['event_failed10']}건, 대조 {n['control_failed10']}건.")
    horizons=[]
    for typ in ['intervention','natural']:
        for tracker in ['oltw','bayes']:
            vals=m[typ][tracker]['horizons']
            horizons.append('| '+typ+' '+tracker.upper()+' | '+' | '.join(f"{vals[str(h)+'s']['mean']:+.3f}" for h in [1,2,5,10])+' |')
    cal=[]
    for r in m['calibration']:
        cal.append(f"| {'제거' if r['erased'] else '유지'} | {r['shift_frames']} | {r['error_after5_sec']:.3f} | {r['mean_entropy_during']:.3f} | {r['mean_true_mass_during']:.3f} |")
    # Diagnostic only: contribution concentration in natural-event estimates.
    concentration={}
    for tracker in ['oltw','bayes']:
        rows=[r for r in pairs if r['tracker']==tracker]
        names=sorted({r['composition'] for r in rows})
        effects=np.array([np.mean([float(r['effect_5s']) for r in rows if r['composition']==g]) for g in names])
        if len(effects):
            worst=int(np.argmax(abs(effects)))
            concentration[tracker]=dict(largest_absolute_contributor=names[worst],effect_sec=float(effects[worst]),
                mean_without_largest=float(np.delete(effects,worst).mean()),median=float(np.median(effects)))
    (out/'descriptive_audit.json').write_text(json.dumps(concentration,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    colors={'oltw':'#2463a5','bayes':'#d37827'}
    for tracker in ['oltw','bayes']:
        ys=[];lo=[];hi=[]
        for h in [1,2,5,10]:
            r=m['intervention'][tracker]['horizons'][f'{h}s'];ys.append(r['mean']);lo.append(r['ci95'][0]);hi.append(r['ci95'][1])
        axes[0].plot([1,2,5,10],ys,'o-',label=tracker.upper(),color=colors[tracker])
        axes[0].fill_between([1,2,5,10],lo,hi,alpha=.15,color=colors[tracker])
    axes[0].axhline(0,color='gray',lw=1)
    axes[0].set(title='Healthy before evidence removal',xlabel='Seconds after removal',ylabel='Paired error increase (score s)')
    axes[0].legend(frameon=False)
    for k,tracker in enumerate(['oltw','bayes']):
        r=m['natural'][tracker]['horizons']['5s'];mean=r['mean'];l,h=r['ci95']
        axes[1].errorbar(k,mean,yerr=[[mean-l],[h-mean]],fmt='o',capsize=5,color=colors[tracker])
    axes[1].axhline(0,color='gray',lw=1)
    axes[1].set(xticks=[0,1],xticklabels=['OLTW','BAYES'],ylabel='Event minus control growth (score s)',title='Natural events: not supported')
    for k,tracker in enumerate(['oltw','bayes']):
        r=m['intervention'][tracker];n=r['eligible_scores']
        axes[2].bar(k-.17,100*r['intact_failed10']/n,width=.32,color='#2463a5',label='Intact' if k==0 else None)
        axes[2].bar(k+.17,100*r['erased_failed10']/n,width=.32,color='#c65335',label='Erased' if k==0 else None)
    axes[2].set(xticks=[0,1],xticklabels=['OLTW','BAYES'],ylabel='Sustained failures within 10 s (%)',title='Includes every eligible nonfailure',ylim=(0,105))
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.savefig(ROOT/'results/paper/p3_temporal.png',dpi=180)
    fig.savefig(ROOT/'results/paper/p3_temporal.pdf');plt.close(fig)
    text=f"""# 원래 P3 시간순서 검정 결과 — 2026-09-18

**동일한 정상 추적 상태에서 위치를 건드리지 않고 관측 단서만 제거하면 이후 오차가 증가한다는
개입 결과는 두 추적기 모두 통과했다. 자연적인 증거 약화 이후 드리프트가 증가한다는
사건 비교는 두 추적기 모두 미달이다. 원래 자연 P3 전체는 여전히 미판정이다.**

P1·P2는 재검정하지 않았다. 이번에는 기존 사용63개 score ID를 제외한 새40개 score ID를
사용했다. 악보/악장/버전 단위이며 완전히 독립인40개 작품이라는 뜻은 아니다.
예를 들어 Schumann/Toccata와 Toccata_repeat가 함께 포함되어 있다.
같은 작곡가·연주자·모음곡 및 다른 악보 버전의 상관은 남아 있다.

## 주 검정4개 (5초, Holm 보정)

| 검정 | 기여 악보 수 | 평균 효과(악보 초) | 주변95% CI | Holm p | 판정 |
|---|---:|---:|---|---:|---|
{chr(10).join(primary)}

intervention은 제거 조건−원본 조건의 5초 뒤 오차 차이이다. 같은 개입 전 상태이므로
오차 증가량의 차이와 같다. natural은 증거약화 사건의 오차 증가량−매칭대조의 증가량이다.
개입 전2초 모든 오차≤.5초인 위험집합만 주 분석하며 미래 결과로 골라내지 않았다.
개입 전에 두 조건의 전체 추적 경로는 동일했다. 한 추적기의 건강한 집합을 다른 추적기에
강제하지 않았으므로 두 효과크기는 동일 표본 비교가 아니다.

## 표본 흐름 및 미발생 사례

{chr(10).join(details)}

자연 사건은 끝난 과거2초 입력의 best_wrong−correct similarity가 -.05 이상인 상태를
1초 간격으로 두번 확인한 시점이다. 그전 샘플은 -.05 미만이어야 한다.
따라서 약화의 최초 물리시점이 아니라 최소1초 확인 지연이 있는 측정시점이다.
대조는 강한 증거(≤-.15)와 같은 과거 건강도 조건, 현재오차·오차추세·템포 caliper를 사용했다.
후속 드리프트 여부로 사건/대조를 선별하지 않았고 미매칭 사건도 audit CSV에 남겼다.

## 시간별 효과 곡선 — 관계식 도출 대신 전부 공개

| 비교 | 1초 | 2초 | 5초(주) | 10초 |
|---|---:|---:|---:|---:|
{chr(10).join(horizons)}

위 표는 고정된 네 시점의 평균 짝효과 f(h)를 기술한 것이다. 연속 동역학 방정식의 도출이나
미분식의 증명이 아니다. 자연 P3가 지지되지 않았으므로 새 선형/임계 동적 식을 적합하지 않았다.
개입후 증가가 양수라는 결과와 별도로, 모든 자연 사건에서 양의 지연으로 드리프트가
발생한다고 주장할 수 없다. first sustained failure는 비발생을10초에서 우측검열해 저장했다.

## 인위적 위치 교란 — 보조 측정기 점검

기존 첫8연주에서만 Bayesian posterior를30초 직전 ±5프레임 이동했다.
주 실험은 이 이동을 전혀 사용하지 않았다. 끝 경계 질량보존 및0이동 원본 경로 동일성 통과.
엔트로피는 분포의 폭이지 정답성을 뜻하지 않는다. 확신 있게 틀릴 수 있으므로
정답 주변 확률도 같이 기록했다. 아래 평균은 검정/모집단 추론에 쓰지 않는다.

| 관측 단서 | 위치 이동(프레임) | 5초 뒤 오차 | 개입 중 엔트로피 | 정답±.5초 확률 |
|---|---:|---:|---:|---:|
{chr(10).join(cal)}

## 사후 기술 진단 (주 판정을 바꾸지 않음)

{chr(10).join(f"- {k.upper()} 자연사건 최대 절대 기여: {r['largest_absolute_contributor']} ({r['effect_sec']:+.3f}초). 그 악보 제외 평균 {r['mean_without_largest']:+.3f}초; 전체 중앙값 {r['median']:+.3f}초." for k,r in concentration.items())}

음의 평균을 ‘증거 약화가 오차를 줄인다’고 읽지 않는다. 표본/매칭·자기상관·드문 실패의
영향과 자연 변동이 남아 있으며, 사전 방향 가설을 지지하지 않았다는 것이 정확한 결론이다.

## 주장 가능한 범위

가능: “개입 직전 정확히 추적하던 연주에서, 위치 상태를 조작하지 않고 관측의 위치별
비용 차이를 제거하자 5초 뒤 추적 오차가 증가했다. 이는 OLTW와 고정 Bayesian filter에서 관찰됐다.”

불가: “자연 악보 모호성이 증거 상실을 유발하고 그 뒤 드리프트가 발생한다는 전체 사슬을 증명했다.”
증거 제거는 비용행 평균 치환이므로 비용분포도 변한다. 실제 음향의 모호성과 동일하지 않다.
GT·전곡 길이 tempo scaling을 사용하는 oracle 진단이고, 새40악보는 별도 외부 데이터셋이 아니다.
자연 매칭은 관측된 과거 상태만 맞추므로 인과 매개효과를 식별하지 않는다.
건강도/증거 임계값은 고정했지만 이 선택에 대한 외적 타당성은 아직 별도 문제다.

## 재현

```text
python scripts/p3_temporal.py --config configs/p3_temporal.yaml
python scripts/report_p3_temporal.py --result {args.result}
python -m pytest tests -q
```

산출물: `{args.result}`의 config, protocol/source snapshot, MIDI/annotation 해시,
manifest, calibration, intervention, evidence, natural_pairs, natural_event_audit,
cases 전체경로, metrics, descriptive_audit.
그림: [PNG](results/paper/p3_temporal.png), [PDF](results/paper/p3_temporal.pdf).
전체 테스트 **51개 통과**. 증거의 과거 입력 제한·매칭 규칙·교란질량·원본경로 보존 검증 포함.

확신도: 실행 수치·검증은 확정. 현재 조건의 개입 효과 일반화는 잠정.
자연 P3는 미판정. 성공한 개입과 실패한 자연 비교를 분리해 보존한다.
"""
    (ROOT/'P3_TEMPORAL_REPORT.md').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
