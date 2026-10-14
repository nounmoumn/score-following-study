# -*- coding: utf-8 -*-
"""
Dannenberg(1984) 매칭 알고리즘 데모.

핵심: 행렬의 두 축은 성격이 다르다.
  - 행(row) = 악보 이벤트  -> 처음부터 전부 알고 있음
  - 열(col) = 연주 이벤트  -> 시간이 지나야 하나씩 도착함

연주 이벤트가 하나 도착할 때마다 '열 하나'만 계산한다.
그 계산에 필요한 건 (1) 방금 도착한 이벤트, (2) 바로 이전 열 = 전부 과거 정보.
미래 연주 정보는 단 한 번도 쓰지 않는다.
"""

import sys

# Windows 콘솔(cp949)에서 한글 출력 시 UnicodeEncodeError 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCORE = "AGEGABC"        # 악보: 처음부터 전부 알고 있는 것
PERFORMANCE = "AGEDGBC"  # 연주: 시간에 따라 하나씩 도착할 것


def run(score: str, performance: str) -> None:
    n_rows = len(score)
    cols: list[list[int]] = []   # 지금까지 계산된 열들 (과거 + 현재)
    max_so_far = 0               # 지금까지 본 최댓값 (보고 판정용)

    print(f"악보(전부 알고 있음): {' '.join(score)}")
    print("=" * 62)

    for c, event in enumerate(performance):
        # --- 새 연주 이벤트 도착: 열 하나를 새로 계산 ---
        prev = cols[-1] if cols else [0] * n_rows   # 이전 열 (없으면 0)
        cur = [0] * n_rows

        for r in range(n_rows):
            up   = cur[r - 1] if r > 0 else 0        # [r-1, c] : 같은 열, 위 칸
            left = prev[r]                           # [r,   c-1] : 이전 열
            cur[r] = max(up, left)

            if event == score[r]:
                diag = prev[r - 1] if r > 0 else 0   # [r-1, c-1]
                cur[r] = max(cur[r], 1 + diag)

        cols.append(cur)

        # --- 보고(reporting): 최댓값이 커졌으면 "방금 매치됐다" ---
        col_max = max(cur)
        report_row = None
        if col_max > max_so_far:
            max_so_far = col_max
            # 휴리스틱 3: 동점이면 악보상 가장 이른 위치
            report_row = cur.index(col_max)

        # --- 출력: 지금까지 계산된 행렬만 (오른쪽은 아직 존재하지 않음) ---
        print(f"\n[t={c+1}] 연주 이벤트 '{event}' 도착 -> 열 {c+1} 계산")
        header = "        " + "  ".join(performance[:c + 1])
        print(header + "     <- 여기까지만 도착함")
        for r in range(n_rows):
            row_vals = "  ".join(str(cols[k][r]) for k in range(c + 1))
            mark = ""
            if report_row is not None and r == report_row:
                mark = "  <== 보고: 악보 " + str(r + 1) + "번째('" + score[r] + "')에 있음"
            print(f"   {score[r]}    {row_vals}{mark}")
        if report_row is None:
            print(f"   -> 최댓값 그대로({max_so_far}). 보고 없음 = 악보에 없는 음으로 판단")

    print("\n" + "=" * 62)
    print("최종 최선의 매치 길이:", max_so_far)


if __name__ == "__main__":
    run(SCORE, PERFORMANCE)
