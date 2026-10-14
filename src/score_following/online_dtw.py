# -*- coding: utf-8 -*-
"""
Dixon (2005) On-Line Time Warping — 실데이터용 일반화.

원본: ismir2026_lbd/scripts/online_dtw.py (2026-09-06 작성, 계산 로직 변경 없이 이동).
탐색 스크립트들이 참조하는 원본은 scripts/exploratory/online_dtw.py에도 그대로 남겨둔다
(임포트 경로만 다르고 내용은 동일 — 사본이지 리다이렉트가 아님, 수정 시 양쪽 동기화 필요).

이건 "우리 알고리즘"이 아니다. 표준 실시간 정렬 알고리즘을
검증용 추적기로 재사용하는 것 — 논문 §6 "통 B: 검증용 엔지니어링"에
해당한다.
"""
import numpy as np

INF = float("inf")


class OnlineDTW:
    """
    U: 라이브(연주) 특징 시퀀스, shape (m, d)
    V: 참조(스코어) 특징 시퀀스, shape (n, d)
    각 프레임은 이미 L2 정규화되어 있다고 가정 (내적 = 코사인 유사도).
    local_cost = 1 - cosine_similarity  (0=완전 일치, 2=정반대)
    """

    def __init__(self, U, V, band_width=60, max_run_count=3):
        self.U = U
        self.V = V
        self.m, self.n = len(U), len(V)
        self.C = band_width
        self.MAX_RUN_COUNT = max_run_count

        # 전체 m x n 행렬을 만들지 않는다 — 딕셔너리로 계산된 셀만 저장.
        self.D = {}
        self.L = {}
        self.t = 0
        self.j = 0
        self.previous = None
        self.run_count = 0

        # 진단/검증용 로그: 매 스텝의 (t, j) 및 그 시점의 결정
        self.path = []  # (t, j) — 라이브 프레임 t가 확정된 시점의 최종 j

    def _cost(self, i, j):
        u, v = self.U[i], self.V[j]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 2.0
        return 1.0 - float(np.dot(u, v) / (nu * nv))

    def _eval(self, i, j):
        if i < 0 or j < 0 or i >= self.m or j >= self.n:
            return
        if (i, j) in self.D:
            return
        d = self._cost(i, j)
        if i == 0 and j == 0:
            self.D[(i, j)] = d
            self.L[(i, j)] = 1
            return
        best_cost, best_len = INF, 0
        for pi, pj, w in ((i, j - 1, 1), (i - 1, j, 1), (i - 1, j - 1, 2)):
            if pi < 0 or pj < 0:
                continue
            if (pi, pj) not in self.D:
                continue
            cost = self.D[(pi, pj)] + w * d
            if cost < best_cost:
                best_cost, best_len = cost, self.L[(pi, pj)] + w
        if best_cost < INF:
            self.D[(i, j)] = best_cost
            self.L[(i, j)] = best_len

    def _norm(self, i, j):
        if (i, j) not in self.D or self.L[(i, j)] == 0:
            return INF
        return self.D[(i, j)] / self.L[(i, j)]

    def _get_inc(self):
        if self.t < self.C:
            return "Both"
        if self.run_count > self.MAX_RUN_COUNT:
            return "Column" if self.previous == "Row" else "Row"

        best = INF
        x, y = self.t, self.j
        lo_k = max(0, self.j - self.C)
        for k in range(lo_k, self.j + 1):
            v = self._norm(self.t, k)
            if v < best:
                best, x, y = v, self.t, k
        lo_i = max(0, self.t - self.C)
        for k in range(lo_i, self.t + 1):
            v = self._norm(k, self.j)
            if v < best:
                best, x, y = v, k, self.j

        if x == self.t and y < self.j:
            return "Row"
        elif y == self.j and x < self.t:
            return "Column"
        return "Both"

    def run(self, stop_at_performance_end=False):
        """Align sequences; optionally stop before the artificial score tail.

        Default behavior is preserved for historical experiments. The early
        stop is equivalent to scoring real_path(), but avoids computing cells
        after the last live frame during paired evidence-ablation replays.
        """
        self._eval(0, 0)
        while self.t < self.m - 1 or self.j < self.n - 1:
            if stop_at_performance_end and self.t >= self.m - 1:
                break
            inc = self._get_inc()

            if inc != "Column" and self.t + 1 < self.m:
                self.t += 1
                for k in range(self.j - self.C + 1, self.j + 1):
                    if k >= 0:
                        self._eval(self.t, k)
            elif inc != "Column":
                pass

            if inc != "Row" and self.j + 1 < self.n:
                self.j += 1
                for k in range(self.t - self.C + 1, self.t + 1):
                    if k >= 0:
                        self._eval(k, self.j)
            elif inc != "Row":
                pass

            if inc == self.previous:
                self.run_count += 1
            else:
                self.run_count = 1
            if inc != "Both":
                self.previous = inc

            self.path.append((self.t, self.j))

            # 종료 조건: 둘 다 끝에 도달
            if self.t >= self.m - 1 and self.j >= self.n - 1:
                break

        return self.path

    def real_path(self):
        """연주가 처음 마지막 프레임(t=m-1)에 도달하는 지점까지만 잘라서 반환.

        week4에서 발견된 문제(ENGINEERING_LOG.md "DTW 퇴화 버그" 절 "부수 발견"):
        연주가 스코어보다 짧은 접두어일 때, run()의 결과 뒤쪽은 연주 종료 후
        스코어 인덱스만 억지로 채우는 인위적 꼬리다. 이 꼬리를 제거한다.
        run() 호출 후에만 유효하다(self.path가 채워져 있어야 함).
        """
        if not self.path:
            self.run()
        first_end_idx = next(
            (i for i, (t, j) in enumerate(self.path) if t == self.m - 1),
            len(self.path) - 1,
        )
        return self.path[: first_end_idx + 1]
