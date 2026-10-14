"""Report the frozen state-conditioned holdout without model reselection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p3_temporal import read_csv

def main():
    sys.stdout.reconfigure(encoding='utf-8');ap=argparse.ArgumentParser();ap.add_argument('--result',required=True);args=ap.parse_args()
    out=ROOT/args.result;m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    rows=read_csv(out/'landmarks.csv');train=read_csv(out/'training_manifest.csv');test=read_csv(out/'manifest.csv')
    assert not ({r['score'] for r in train}&{r['score'] for r in test})
    assert len(rows)==960 and all(float(r['exposure_last_time_sec'])+10<90 for r in rows)
    assert all(np.isfinite(float(r[k])) for r in rows for k in ['weakness','risk','y_5s'])
    preds=read_csv(out/'predictions.csv');assert len(preds)==1440
    lines=['# P3 사전 상태·관측 구별력 분리 검증 (2026-09-18)','',
        f'원본: `{args.result}`. 실행 전 고정: [P3_STATE_PROTOCOL.md](P3_STATE_PROTOCOL.md).',
        '기존40 score 학습, 미사용40 score 검증, 각12시점. P1/P2 효과검정을 반복하지 않았다.','',
        '## 5초 주검정','',
        '| 비교 | 기준 MAE | 확장 MAE | 상대 개선 | 개선 95% CI | Holm p | 판정 |',
        '|---|---:|---:|---:|---|---:|---|']
    for c in m['comparisons']:
        if c['primary']:
            lines.append(f"| {c['name']} | {c['baseline_mae']:.6f} | {c['full_mae']:.6f} | {c['improvement_percent']:+.2f}% | [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | {c['holm_p']:.4f} | {'통과' if c['passed'] else '미달'} |")
    lines+=['','MAE 단위는 log1p(초) 오차증분. 양수 개선은 확장모형이 더 정확하다는 뜻이다.',
        'M0=노출 전 오차이력·시간·템포·prior 상태, M1=M0+관측약화 W, M2=M1+W×경쟁위험 R.',
        'M3=M0+노출완료 오차, M4=M3+W+W×R. M2 성공만으로 상호작용의 독립 효과를 주장하지 않는다.',
        '학습표준화와 회귀적합은 기존40개만 사용했다. 새40개의 결과로 모형/임계값을 선택하지 않았다.','',
        '## 모든 시계와 작곡가 단위 보조 결과','',
        '| 시계 | 비교 | score 평균 개선 | score CI | 작곡가 균등평균 개선 | 작곡가 CI | 한 작곡가 제외 평균 범위 |',
        '|---:|---|---:|---|---:|---|---|']
    for c in m['comparisons']:
        cc=c['composer_balanced'];loo=list(c['leave_composer_out'].values())
        lines.append(f"| {c['horizon_sec']:g}s | {c['name']} | {c['mean']:+.6f} | [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}] | {cc['mean']:+.6f} | [{cc['ci95'][0]:+.6f}, {cc['ci95'][1]:+.6f}] | [{min(loo):+.6f}, {max(loo):+.6f}] |")
    lines+=['','2/10초와 M2 vs M1은 보조 결과이며 유리한 시계로 주검정 결과를 대체하지 않는다.','',
        '## 표본·시간 순서·한계','']
    for split,d in m['description'].items():
        lines.append(f"- {split}: {d['scores']} score, {d['landmarks']} 시점. R≥1% {d['risk_ge_1percent']}개, 노출 전 건강 {d['pre_healthy']}개, W>0 {d['weakness_favors_wrong']}개. 모두 기술통계이며 모집단 선별에 쓰지 않음.")
    counts={}
    for r in test:counts[r['composition'].split('/')[0]]=counts.get(r['composition'].split('/')[0],0)+1
    lines+=['- 검증 작곡가 분포: '+', '.join(f'{k} {v}' for k,v in sorted(counts.items()))+'.',
        '- t 직전 posterior에서 후보와 가중치를 고정. [t,t+2) 관측만 W에 사용. 목표는 t+1.9초 오차에서 이후 h초의 증분.',
        '- Bayes 재생80개 모두 기준 구현과 전체 경로 일치. 학습40개는 기존 저장 경로와도 일치. 960행/1440예측·score 비중복·유한값·시계 검사 통과.',
        '- 정답후보와 오차·과거템포에 GT를 사용한 oracle 분석. 전체 MIDI 길이 비율로 score hop을 정하는 기존 전처리도 사용하므로 실제 온라인 배치 성능 주장은 불가.',
        '- W는 고정속도 후보궤적의 구별력이다. 실제 filter 전체 템포 경로의 주변가능도가 아니며 템포불일치 영향을 받을 수 있다.',
        '- 개별 score는 미사용이지만 같은 작품의 다른 악장/버전 및 같은 작곡가가 양쪽에 있을 수 있다. 작품전체/작곡가 외삽 검증은 아니다.',
        '- score 단위 bootstrap은 시점 간 상관을 묶지만 작품 간 상관을 없애지 않는다. 작곡가 CI는 소수 그룹의 제한이 있다. sign-flip은 효과분포 대칭성을 가정한다.',
        '- 이 단계는 Bayesian tracker의 조건부 예측검정이다. 인과효과, 모든 추적기에 대한 일반성, 자연 약화→누적 drift의 증명으로 바꾸어 쓰지 않는다.','',
        '## 판정','']
    primary=[c for c in m['comparisons'] if c['primary']];passed=[c['name'] for c in primary if c['passed']]
    lines.append('통과한 주검정: '+(', '.join(passed) if passed else '없음')+'. 원래 자연 P3의 증명 판정은 보류한다.')
    lines.append('현재 결과를 근거로 사후 위험 임계값/시계/속도를 골라 성공으로 재분류하지 않는다. 다음 설계는 고정속도 측정의 타당성과 드문 경쟁 상태에서의 인과적 상호작용을 별도로 검증해야 한다.')
    text='\n'.join(lines)+'\n';(ROOT/'P3_STATE_REPORT.md').write_text(text,encoding='utf-8');(out/'REPORT.md').write_text(text,encoding='utf-8')
    print(text)

if __name__=='__main__':main()
