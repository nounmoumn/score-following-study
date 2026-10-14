# -*- coding: utf-8 -*-
"""
표준(제약 없는) 전역 DTW — P2(온라인 vs 오프라인 격차) 검증용.
설계: FEEDBACK.md "P2 설계 (온라인 vs 오프라인 격차)" 절 (전략 트랙, 2026-09-08).

OnlineDTW(Dixon 2005, band-limited, causal — online_dtw.py)와 **같은 국소비용·
스텝가중치**를 쓰되, band 제약 없이 전체 (m x n) 비용행렬에 대해 표준 DP +
백트레이스로 최적 경로를 구한다 — "온라인이 미래 정보 없이 실시간으로 내놓는 답"과
"모든 정보를 다 보고 사후에 고른 최선의 답"을 같은 조건에서 비교하기 위한 상한선.

⚠️ 스텝가중치를 OnlineDTW와 반드시 맞춘 이유: 수평/수직=1, 대각=2 (Sakoe-Chiba류
대칭 스텝패턴). 예전 `librosa.sequence.dtw` 퇴화 버그(ENGINEERING_LOG.md "DTW 퇴화
버그" 절)의 원인은 정확히 "수평/수직/대각 이동에 비용 차이가 전혀 없어서(무가중치),
스코어의 한 지점에 눌러앉아 있는 게 실제 정렬 시도보다 수학적으로 더 쌌다"는 것이었다.
가중치를 두면(대각이 더 "저렴"하게 설계됨) 제자리 정체가 항상 손해라 이 문제가
재발하지 않는다 — 아래 test_offline_dtw.py의 회귀 테스트로 직접 확인.

메모리: 전체 D(누적비용) 행렬은 저장하지 않는다(m*n이 커지면 GB 단위 — 예: ASAP에서
가장 긴 곡은 8000프레임대). 대신 이전 행만 유지하는 rolling DP + 백트레이스용
방향 포인터 행렬(int8, m*n 바이트 — 8000x8000이어도 64MB)만 저장한다.

open_end 옵션: week4처럼 **연주가 스코어의 접두어(prefix)뿐**이라 스코어 전체를
커버하지 못하는 경우, 끝점을 (m-1, n-1)로 강제하면 안 된다(그러면 "억지로 끝까지
채우는" 문제가 재발할 수 있음 — OnlineDTW.real_path()가 온라인 쪽에서 대응한 것과
같은 문제의 오프라인 버전). open_end=True면 시작은 (0,0)로 고정하되(week4는
onset-detection으로 시작점이 곡 시작임을 이미 확인했으므로) 끝은 마지막 행(i=m-1)
전체에서 누적비용이 최소인 열을 고른다.

⚠️ band_width 옵션 — **2026-09-08 week4 실행 중 발견한 새 퇴화 모드에 대한 대응**.
open_end=True + band 없음으로 bansanka(만찬가)에 돌렸더니, 61.58초 전체 연주가
스코어의 **12.6초 지점 근처에 완전히 눌러앉는** 해가 나왔다(offline_err가 시간에
거의 선형으로 증가 — 실측: t=56s에서 offline_err=43.4s, 즉 추정 스코어 위치가
줄곧 j≈126프레임=12.6초에 고정됨). 원인: 인트로가 4마디×6반복인 짧은 코드 루프라,
"그 12.6초 구간이 연주 전체와 어중간하게 계속 비슷하다"는 게 대각선을 타고 스코어
끝까지 실제로 진행하는 것보다 총비용이 더 쌌다(week4 스코어 크로마는 진짜 대응
지점에서도 평균 코사인유사도가 0.5대로 낮음 — ENGINEERING_LOG.md "P1 테스트 진단"
절 — 그래서 "제대로 진행"이 별로 안 싸다). **이건 예전 librosa 퇴화 버그와 결이
같다** — 전체 정보를 다 보는 게 "물리적으로 말이 안 되는 해"까지 자유롭게 허용하면
안 된다는 교훈. 해법: band_width를 주면(온라인과 같은 값 등) i-j 차이가 그 폭을
넘는 셀은 계산 자체를 안 해서(비용 inf) 먼 곳의 반복이 애초에 도달 불가능해진다 —
"미래 정보는 다 보되, 터무니없는 템포 이탈(=먼 반복 구간으로 점프)은 배제한다"는
것이 오히려 더 정확한 "오프라인"의 정의. band_width=None(기본)이면 기존 동작(제약
없음, ASAP에서 이미 검증된 그대로) 유지 — ASAP은 MIDI라 진짜 대응 지점의 국소비용이
거의 0에 가까워 이 퇴화가 안 생겼음(재실행 불필요).
"""
import numpy as np

DIAG, VERT, HORIZ, START = 0, 1, 2, -1


def _cost_row(u, V):
    """u: (d,), V: (n, d) — 전부 L2 정규화돼 있다고 가정.
    반환: (n,) = 1 - cosine_similarity(u, V[j])  (OnlineDTW._cost와 동일 정의,
    벡터가 0(무음)이면 최대비용 2.0)."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(V, axis=1)
    denom = nu * nv
    safe_denom = np.where(denom == 0, 1.0, denom)
    sim = (V @ u) / safe_denom
    cost = 1.0 - sim
    return np.where(denom > 0, cost, 2.0)


class OfflineDTW:
    """
    U: 연주 크로마 (m, d), V: 스코어 크로마 (n, d). 프레임별 L2 정규화 가정
    (OnlineDTW와 동일 전제).

    open_end=False (기본): 고전적 전역 DTW — 시작 (0,0), 끝 (m-1,n-1) 고정.
    open_end=True: 시작 (0,0) 고정, 끝은 행 m-1에서 자유(week4류 접두어 연주용).
    band_width=None (기본): 제약 없음(ASAP에서 검증된 원래 동작).
    band_width=C (정수): |i-j|<=C인 셀만 계산 — 먼 반복 구간이 통째로 도달 불가능해짐
      (위 모듈 docstring "band_width 옵션" 참조).
    """

    def __init__(self, U, V, open_end=False, diag_weight=2.0, step_weight=1.0, band_width=None):
        self.U = np.asarray(U)
        self.V = np.asarray(V)
        self.m, self.n = len(self.U), len(self.V)
        self.open_end = open_end
        self.wd = float(diag_weight)
        self.ws = float(step_weight)
        self.band = band_width
        self._path = None
        self.end_j_ = None
        self.total_cost_ = None

    def _band(self, i):
        n = self.n
        if self.band is None:
            return 0, n - 1
        return max(0, i - self.band), min(n - 1, i + self.band)

    def run(self):
        m, n, wd, ws = self.m, self.n, self.wd, self.ws
        if m == 0 or n == 0:
            self._path = []
            return self._path

        pointers = np.zeros((m, n), dtype=np.int8)
        D_prev = np.full(n, np.inf)
        D_curr = np.full(n, np.inf)

        for i in range(m):
            j_lo, j_hi = self._band(i)
            row_cost = _cost_row(self.U[i], self.V)
            D_curr.fill(np.inf)

            if i == 0:
                # band_width가 있으면 i=0일 때 j_lo는 항상 0(max(0,-band)=0)이라
                # 아래 START 처리가 밴드 유무와 무관하게 그대로 성립한다.
                D_curr[0] = row_cost[0]
                pointers[0, 0] = START
                for j in range(1, j_hi + 1):
                    D_curr[j] = D_curr[j - 1] + ws * row_cost[j]
                    pointers[0, j] = HORIZ
            else:
                diag_cand = np.full(n, np.inf)
                diag_cand[1:] = D_prev[:-1] + wd * row_cost[1:]
                vert_cand = D_prev + ws * row_cost
                base_best = np.minimum(diag_cand, vert_cand)
                base_ptr = np.where(diag_cand <= vert_cand, DIAG, VERT)

                D_curr[j_lo] = base_best[j_lo]
                pointers[i, j_lo] = base_ptr[j_lo]
                # 수평(같은 행, 왼쪽 이웃) 의존성만 순차 스캔 — 나머지는 위에서 벡터화함
                for j in range(j_lo + 1, j_hi + 1):
                    horiz_cand = D_curr[j - 1] + ws * row_cost[j]
                    if horiz_cand < base_best[j]:
                        D_curr[j] = horiz_cand
                        pointers[i, j] = HORIZ
                    else:
                        D_curr[j] = base_best[j]
                        pointers[i, j] = base_ptr[j]
            D_prev, D_curr = D_curr, D_prev

        # 루프 종료 시점에 스왑이 한 번 더 일어났으므로 D_prev가 마지막 행(i=m-1)의 값
        last_row = D_prev
        if self.open_end:
            end_j = int(np.argmin(last_row))
        else:
            end_j = n - 1
        if not np.isfinite(last_row[end_j]):
            raise ValueError(
                f"band_width={self.band}가 너무 좁아서 끝점(j={end_j})에 도달하는 경로가 "
                f"없음 — band_width를 늘리거나 open_end=True를 확인할 것."
            )
        self.end_j_ = end_j
        self.total_cost_ = float(last_row[end_j])

        path = []
        i, j = m - 1, end_j
        while True:
            path.append((i, j))
            p = pointers[i, j]
            if p == START:
                break
            if p == DIAG:
                i, j = i - 1, j - 1
            elif p == VERT:
                i, j = i - 1, j
            elif p == HORIZ:
                i, j = i, j - 1
        path.reverse()
        self._path = path
        return path

    def path(self):
        if self._path is None:
            self.run()
        return self._path

    def j_of_t(self):
        """{연주 프레임 t: 최종 추정 스코어 프레임 j} — OnlineDTW 사용 관례와 동일하게
        같은 t에 여러 j가 나오면(수평 이동 여러 번) 마지막 것(가장 진행된 j)을 취한다."""
        out = {}
        for t, j in self.path():
            out[t] = j
        return out
