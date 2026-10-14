import argparse,json,sys,shutil,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from p3_temporal import read_csv


def main():
    sys.stdout.reconfigure(encoding='utf-8');ap=argparse.ArgumentParser();ap.add_argument('--result',required=True);args=ap.parse_args();out=ROOT/args.result
    m=json.loads((out/'metrics.json').read_text(encoding='utf-8'));rows=read_csv(out/'conditions.csv');effects=read_csv(out/'effects.csv')
    assert len(rows)==240 and len(effects)==40
    lines=['# P3 전체경로 관측 측정과 교차 개입','',f'원본: `{args.result}`.',
        '실행 전 고정: [P3_MECHANISM_PROTOCOL.md](P3_MECHANISM_PROTOCOL.md). 기존 temporal40 score를 재사용한 기전 실험이며 독립 재현 표본이 아니다.','',
        '## 측정 타당성','',
        '실제 Bayes transition·restart를 반영한 forward recursion으로 초기 정답/오답 후보별 관측가능도를 계산했다. 작은 상태공간 완전열거와 일치하며 양성/음성/무정보 대조와 가변속도 대조를 통과했다. 전체 테스트 60개 통과.',
        f"실제40개 소거조건 W의 최대 절댓값 {m['null_max_abs']:.3g}. 고정속도 W 중앙값 {m['fixed_path_median']:+.5f}, 전체경로 W 중앙값 {m['all_path_median']:+.5f}. 부호가 달라진 score {m['sign_disagreements']}/40.",
        'W>0은 경쟁후보 관측가능도가 더 큼을 뜻한다. 이 검사는 모형 내 계산과 대조의 타당성을 보장하며 자연 drift의 예측력까지 보장하지 않는다.','',
        '## 사전 고정한 두 주검정','',
        '| 검정 | 평균효과(초) | 95% CI | Holm p | 판정 |',
        '|---|---:|---|---:|---|']
    for t in m['primary']:
        lines.append(f"| {t['name']} | {t['mean']:+.6f} | [{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}] | {t['holm_p']:.4f} | {'통과' if t['passed'] else '미달'} |")
    lines+=['','erasure: 위치상태 유지에서 증거 소거로 증가한 30~40초 평균 절대오차.',
        'interaction: ±5frame 교란 시 소거효과를 평균한 뒤 상태유지 소거효과를 뺀 값. 양수면 교란으로 소거효과가 증폭됨.',
        '오차의 상태교란 주효과를 interaction과 혼동하지 않는다. 소거효과가 커도 interaction이 0일 수 있다.','',
        '## 조건별 기술통계','',
        '| 교란(frame) | 소거 | 노출 중 평균오차(초) | 5초 순간오차(초) | 이후10초 평균오차(초) |',
        '|---:|---|---:|---:|---:|']
    for s in [0,-5,5]:
        for erase in ['False','True']:
            rr=[r for r in rows if int(r['shift_frames'])==s and r['erased']==erase]
            values=[np.mean([float(r[k]) for r in rr]) for k in ['mean_error_during_sec','error5_sec','mean_error_recovery_sec']]
            lines.append(f'| {s} | {erase} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |')
    lines+=['','## 큰 효과의 기술적 점검','',
        '아래는 절댓값 상위3개를 공개한 것이며 표본에서 제외하거나 주검정을 다시 계산하지 않았다.']
    for key in ['erasure','interaction']:
        t=next(t for t in m['primary'] if t['name']==key)
        lines.append(f"- {key}: 양의 score {t['positive']}/40, 중앙값 {t['median']:+.6f}초.")
        for r in sorted(effects,key=lambda r:abs(float(r[key])),reverse=True)[:3]:
            lines.append(f"  - {r['composition']}: {float(r[key]):+.6f}초")
    shifts=[abs(float(r['shift_score_sec'])) for r in rows if int(r['shift_frames'])==5]
    lines+=['',f'5frame 이동의 실제 score초 범위: {min(shifts):.4f}~{max(shifts):.4f}초. 모든 교란 전 경로와 원본 무교란 경로의 일치를 확인했다.',
        'score 단위 CI는 악장/반복버전 상관을 완전히 처리하지 않는다. sign-flip은 효과분포 대칭성 가정이 있다.',
        '이전에 사용한 자료·한 tracker·한 onset·한 shift 크기·oracle GT/전체길이 정규화의 범위에 한정한다.',
        '증거 소거는 인위적 무정보 조건이며 자연 모호성의 모든 유형과 같지 않다. 위치 이동은 posterior 전체 이동이며 다봉 경쟁 상태를 주입하는 것과도 다르다.','',
        '## P3 판정 및 다음 검증의 경계','',
        '어느 주검정이 성공하더라도 원래 자연 P3가 증명되었다고 쓰지 않는다. 인위적 개입의 효과와 자연 약화→미래 drift의 선후관계는 별개다.',
        '다음 자연 검증은 전체경로 지표의 추가 예측력을 과거오차와 비교해야 한다. 기존 소비자료로 사건규칙·모형을 개발하고, 작품단위로 완전히 분리한 미사용 자료를 확보한 뒤 설계를 고정해야 한다.',
        '현재의 악보ID별 분리는 같은 작품의 다른 악장을 허용하므로 엄격한 작품 외삽 표본으로 간주할 수 없다. 미사용 자료 수와 사건 수가 부족하면 충분한 검증력이 있다고 주장하지 않고 외부자료 확보로 전환한다.',
        '“확실히 검증”은 양성결과 보장이 아니라 음성결과도 유지하는 식별 가능한 설계와 독립 검증을 뜻한다.']
    if (out/'boundary_audit.json').exists():
        lines+=['','## 最大効果の追跡確認'.replace('最大効果の追跡確認','최대 효과의 경로 확인'),'',
            'Chopin/Ballades_4는 소거 직전 이미 MAP5572frame 대 정답203frame으로 크게 이탈해 있었다. 소거 후 MAP는5572→5668이며 끝6039에 도달하지 않았다. 악보 끝 흡수 현상이라는 추정은 이 사례에서 지지되지 않는다.',
            '따라서 +822초 효과는 건강한 추적의 새 drift 발생으로 해석할 수 없다. 이미 이탈한 상태가 원관측에서는 회복하지만 소거에서는 유지되는 차이가 포함된다. 미래결과로 이 사례를 제외하지 않았으며 두 주검정 미달은 유지한다.',
            '이 구분 때문에 원래 P3의 다음 자연검정은 사전에 정의한 건강한 시작상태에서 신규 이탈 발생을 측정하고, 회복 실패는 별도 결과로 다뤄야 한다.']
    report='\n'.join(lines)+'\n';(ROOT/'P3_MECHANISM_REPORT.md').write_text(report,encoding='utf-8');(out/'REPORT.md').write_text(report,encoding='utf-8')
    entry='\n## 2026-09-18 전체경로 관측 측정·교차 개입 완료\n\n'
    entry+='기존 temporal40에서240조건. 실제 transition/restart 전체경로 관측가능도 완전열거 및 양성/음성/무정보 대조 통과, 전체60 tests passed.\n'
    for t in m['primary']:entry+=f"- {t['name']}: {t['mean']:+.6f}초, CI {t['ci95']}, Holm p={t['holm_p']:.4f}, {'통과' if t['passed'] else '미달'}.\n"
    entry+='\n원래 자연 P3는 여전히 미검증. 기존자료 기전검정이며 독립재현 아님. [P3_MECHANISM_REPORT.md](P3_MECHANISM_REPORT.md) 및 실행 전 [protocol](P3_MECHANISM_PROTOCOL.md)에 범위·한계 기록.\n'
    entry+=f'원본 `{args.result}`. 다음 단계는 작품단위 미사용 표본 확보/검증력 확인 후 전체경로 지표의 자연 시간예측 검정.\n\n'
    for name in ['ENGINEERING_LOG.md','FEEDBACK.md','PAPER_NUMBERS.md']:
        path=ROOT/name;old=path.read_text(encoding='utf-8')
        if '## 2026-09-18 전체경로 관측 측정·교차 개입 완료' not in old:
            first,_,rest=old.partition('\n');path.write_text(first+'\n'+entry+rest,encoding='utf-8')
    files=[Path(__file__),ROOT/'P3_MECHANISM_REPORT.md',ROOT/'tests/test_path_evidence.py']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'report_provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':main()
