"""Figure 5 (Walker2d) — 세 줄 파노라마 스택 (2026-11-05).

사용자 (2026-11-05): "walker2d는 파노라마 3개를 쌓아서 stack해서 만들어서 올릴거야. rho0.7로 만든 버전이랑
rho 0.1인 버전 두개를 각각 3stack panorama" — 줄 = CD / Safety / 낙상(fall) 에피소드.
입력은 scripts/render_cd_panorama.sh (src.viz.render_video._save_panorama) 가 만든 에피소드별 PNG 그대로다
(렌더 SSOT 재사용, 새 렌더 경로 없음). 각 PNG 에서 패널 N_PICK 장을 골라 머리글(t·x 글자)과 하늘 일부를
잘라내고 같은 크기로 위아래로 쌓는다. 한 줄 = 한 에피소드 (에피소드끼리 섞지 않음).

사용: python -m experiments.paper_fig5.compose_walker_stack --rho 0.7 --rows CD:<png> Safety:<png> Fall:<png>
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 320          # = render_video 기본 width 640 × pano_crop 0.5
PANEL_SEP = 4          # = _save_panorama 의 sep
HEAD_H = 28            # = _save_panorama 의 head (t·x 글자 줄)
N_PICK = 5             # 사용자 스케치: 줄마다 5 칸
SKY_CUT = 110          # 프레임 위쪽 하늘 (몸통이 닿지 않는 높이) 잘라냄
SIDE_CUT = 30          # 패널 좌우 여백 잘라냄 (몸통은 카메라 중앙)
# 2026-11-06 (카메라레디 TODO 3): 사용자 "같은 파노라마에 쓰인 이미지들은 딱 붙여서 여백을 없애서 파노라마가 하나의
#   연속되어진 파노라마처럼 보이게 … 서로 다른 파노라마끼리는 여전히 공간 여백을 줘서 확실히 분리" → 줄 안 간격 0, 줄 사이 간격 확대.
#   줄 순서는 --rows 순서 그대로 (b1 CD 잘 걷기 → b2 같은 정책의 낙상 → b3 safety).
GAP = 0                # 같은 파노라마 안 패널 사이 간격 (0 = 연속 파노라마)
ROW_GAP = 28           # 파노라마 (줄) 사이 흰 간격
LABEL_W = 150          # --labels 일 때만 (기본: 라벨 없음 — 사용자 2026-11-05: "레이블 필요없어 … figure 설명란에 넣을거라서")


def _panels(path):
    im = Image.open(path).convert("RGB")
    n = (im.width + PANEL_SEP) // (PANEL_W + PANEL_SEP)
    assert n * PANEL_W + (n - 1) * PANEL_SEP == im.width, (path, im.width, n)
    return [im.crop((j * (PANEL_W + PANEL_SEP) + SIDE_CUT, HEAD_H + SKY_CUT,
                     j * (PANEL_W + PANEL_SEP) + PANEL_W - SIDE_CUT, im.height)) for j in range(n)]


def _pick(panels, kind, n_pick=N_PICK):
    n = len(panels)
    # 걷기 줄은 t=0 (모두 같은 초기 자세) 을 빼고 고르게, 낙상 줄은 마지막(넘어진) 칸을 반드시 포함.
    lo = 0 if kind == "fall" else 1
    idx = np.unique(np.linspace(lo, n - 1, n_pick).round().astype(int))
    assert len(idx) == n_pick, (n, idx)
    return [panels[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs=3, required=True,
                    help="LABEL:png, 위→아래 순서 그대로 쌓는다 (fall 줄은 라벨이 Fall 로 시작)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", action="store_true", help="오른쪽에 줄 이름 표시 (검토용)")
    ap.add_argument("--n_pick", type=int, default=N_PICK,
                    help="줄마다 패널 수 (2026-11-06 사용자: \"frame을 6개라던가 7개까지 합쳐서 … 초안 비교\")")
    args = ap.parse_args()
    rows = []
    for spec in args.rows:
        label, path = spec.split(":", 1)
        rows.append((label, _pick(_panels(path), "fall" if label.lower().startswith("fall") else "walk", args.n_pick)))
    pw, ph = rows[0][1][0].size
    W = args.n_pick * pw + (args.n_pick - 1) * GAP + (LABEL_W if args.labels else 0)
    H = len(rows) * ph + (len(rows) - 1) * ROW_GAP
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 40)
    for r, (label, panels) in enumerate(rows):
        y = r * (ph + ROW_GAP)
        for j, p in enumerate(panels):
            assert p.size == (pw, ph)
            canvas.paste(p, (j * (pw + GAP), y))
        if args.labels:
            draw.text((args.n_pick * pw + (args.n_pick - 1) * GAP + 14, y + ph // 2), label, fill="black", font=font, anchor="lm")
    canvas.save(args.out)
    print("saved", args.out, canvas.size)


if __name__ == "__main__":
    main()
