# -*- coding: utf-8 -*-
"""
Dixon(2005) On-Line Time Warping 데모.

밴드(band)가 어떻게 움직이는지 눈으로 보는 것이 목적.

  - 세로축 t = 라이브 연주 (실시간으로 하나씩 도착)
  - 가로축 j = 저장된 연주 (처음부터 다 알고 있음)
  - 전체 행렬을 채우지 않고, 폭 c 짜리 '띠'만 계산한다
  - 그 띠는 "제일 싼 셀"을 따라 매 스텝 움직인다

표시:
  .  아직 계산 안 함 (건너뛴 셀)
  #  이전 스텝에서 계산됨
  N  이번 스텝에 새로 계산됨   <- 이게 '밴드'가 움직이는 모습
  @  현재 위치 (t, j)
"""

import sys

# Windows 콘솔(cp949)에서 한글 출력 시 UnicodeEncodeError 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

INF = float("inf")

# 저장된 연주 V: 음 10개를 일정한 속도로
V = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 라이브 연주 U: 같은 곡인데 5~7번 음에서 느려짐(각 음이 2프레임 지속)
U = [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10]

C = 4              # 밴드 폭 (논문의 c)
MAX_RUN_COUNT = 3  # 한 방향 연속 허용 횟수 = 기울기 제한


def local_cost(i, j):
    """d(i,j) — u_i 와 v_j 를 짝지을 때의 비용. 같으면 0, 다를수록 큼."""
    return abs(U[i] - V[j])


class OLTW:
    def __init__(self):
        self.m, self.n = len(U), len(V)
        self.D = [[INF] * self.n for _ in range(self.m)]   # 누적 비용
        self.L = [[0] * self.n for _ in range(self.m)]     # 경로 길이
        self.computed = [[False] * self.n for _ in range(self.m)]
        self.t = 0
        self.j = 0
        self.previous = None
        self.run_count = 0
        self.new_cells = []      # 이번 스텝에 계산된 셀
        self.path_log = []

    def evaluate_path_cost(self, i, j):
        """식 (3): 대각선에 가중치 2. 이미 계산된 칸만 참조."""
        if i < 0 or j < 0 or i >= self.m or j >= self.n:
            return
        d = local_cost(i, j)
        best_cost, best_len = INF, 0

        if i == 0 and j == 0:                     # 시작점
            best_cost, best_len = d, 1
        else:
            # (i, j-1) 왼쪽 / (i-1, j) 위 / (i-1, j-1) 대각선
            for pi, pj, w in ((i, j - 1, 1), (i - 1, j, 1), (i - 1, j - 1, 2)):
                if pi < 0 or pj < 0:
                    continue
                if not self.computed[pi][pj]:
                    continue
                cost = self.D[pi][pj] + w * d
                if cost < best_cost:
                    best_cost, best_len = cost, self.L[pi][pj] + w

        if best_cost < INF:
            self.D[i][j] = best_cost
            self.L[i][j] = best_len
            self.computed[i][j] = True
            self.new_cells.append((i, j))

    def norm_cost(self, i, j):
        """경로 길이로 정규화 — 길이가 다른 경로를 공평하게 비교하기 위해."""
        if not self.computed[i][j] or self.L[i][j] == 0:
            return INF
        return self.D[i][j] / self.L[i][j]

    def get_inc(self):
        """밴드를 어느 쪽으로 옮길지 결정: Row / Column / Both"""
        # ① 초기 단계: 아직 밴드 폭만큼도 안 채워짐
        if self.t < C:
            return "Both"

        # ② 한 방향으로 너무 오래 갔으면 강제로 반대 방향 (기울기 제한)
        if self.run_count > MAX_RUN_COUNT:
            return "Column" if self.previous == "Row" else "Row"

        # ③ 현재 행(k==t)과 현재 열(l==j) 위에서 제일 싼 셀을 찾는다
        best = INF
        x, y = self.t, self.j
        for k in range(self.n):                    # 현재 행
            v = self.norm_cost(self.t, k)
            if v < best:
                best, x, y = v, self.t, k
        for k in range(self.m):                    # 현재 열
            v = self.norm_cost(k, self.j)
            if v < best:
                best, x, y = v, k, self.j

        # 최소점이 현재 행에서 j보다 왼쪽 -> 저장본 j가 너무 앞서감 -> t만 전진
        if x == self.t and y < self.j:
            return "Row"
        # 최소점이 현재 열에서 t보다 위 -> 라이브 t가 너무 앞서감 -> j만 전진
        elif y == self.j and x < self.t:
            return "Column"
        return "Both"

    def render(self, step, inc):
        lines = []
        header = "        " + " ".join(f"{j:2d}" for j in range(self.n))
        lines.append(f"\n[step {step:2d}]  t={self.t}  j={self.j}  "
                     f"결정={inc}  runCount={self.run_count}")
        lines.append(header + "   <- j (저장된 연주)")
        newset = set(self.new_cells)
        for i in range(self.m):
            row = []
            for j in range(self.n):
                if (i, j) == (self.t, self.j):
                    ch = " @"
                elif (i, j) in newset:
                    ch = " N"
                elif self.computed[i][j]:
                    ch = " #"
                else:
                    ch = " ."
                row.append(ch)
            mark = "  <- t (라이브)" if i == self.t else ""
            lines.append(f"   t={i:2d} " + "".join(row) + mark)
        return "\n".join(lines)

    def run(self, max_steps=40, show_every=1):
        print("저장된 연주 V (다 알고 있음):", V)
        print("라이브 연주 U (하나씩 도착) :", U)
        print(f"밴드 폭 c = {C},  MaxRunCount = {MAX_RUN_COUNT}")
        print("=" * 64)

        self.new_cells = []
        self.evaluate_path_cost(0, 0)

        step = 0
        while step < max_steps:
            inc = self.get_inc()
            self.new_cells = []

            # --- t 증가 (= 새 오디오 프레임 소비 + '행' 계산) ---
            if inc != "Column":
                if self.t + 1 >= self.m:
                    break
                self.t += 1
                for k in range(self.j - C + 1, self.j + 1):
                    if k >= 0:
                        self.evaluate_path_cost(self.t, k)

            # --- j 증가 (= 저장본 전진 + '열' 계산) ---
            if inc != "Row":
                if self.j + 1 >= self.n:
                    break
                self.j += 1
                for k in range(self.t - C + 1, self.t + 1):
                    if k >= 0:
                        self.evaluate_path_cost(k, self.j)

            # --- runCount 갱신 ---
            if inc == self.previous:
                self.run_count += 1
            else:
                self.run_count = 1
            if inc != "Both":
                self.previous = inc

            step += 1
            self.path_log.append((self.t, self.j, inc))
            if step % show_every == 0:
                print(self.render(step, inc))

        # --- 통계 ---
        total = self.m * self.n
        done = sum(1 for i in range(self.m) for j in range(self.n)
                   if self.computed[i][j])
        print("\n" + "=" * 64)
        print(f"전체 행렬 칸수      : {self.m} x {self.n} = {total}")
        print(f"실제 계산한 칸수    : {done}  ({done/total*100:.0f}%)")
        print(f"건너뛴 칸수         : {total - done}")
        print("\n최종 정렬 (라이브 t -> 저장본 j):")
        for t, j, inc in self.path_log:
            print(f"   라이브 {t:2d}번 프레임(음={U[t]:2d})  ->  "
                  f"저장본 {j:2d}번(음={V[j]:2d})   [{inc}]")


if __name__ == "__main__":
    OLTW().run()
