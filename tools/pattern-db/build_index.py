#!/usr/bin/env python3
"""build_index.py — 型データベースの取り込み・集計。

manifest（動画ID・ジャンル・metrics.json の場所の一覧）を読み、
動画ごとの型プロファイル（voice_profile.profile_from_metrics）を
index.json に蓄積し、ジャンル別／全体の分布統計を aggregate.json
に出力します。

設計方針:
- 言語ごとにフォルダを分ける（pattern-db/<lang>/）。今は ja のみ。
  将来の多言語はフォルダと manifest の追加だけで対応する。
- 「AIの強みは人間にできない量の学習データ」（ボブ方針）。
  1本のお手本ではなく、多数の動画の型の分布（中央値・四分位）を
  「自然な範囲」の基準にするため、件数を増やすことが前提の設計。

使い方:
  python3 build_index.py --lang ja
  （pattern-db/ja/manifest.json を読み、同じ場所に index.json と
    aggregate.json を書き直します）
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import voice_profile as vp

FIELDS = ("speed_speech", "speed_overall", "gap_median", "f0_median_hz",
          "delta_median_st", "span_median_st")


def dist(values):
    """分布の要約（件数・中央値・四分位・最小最大）。"""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    q = statistics.quantiles(vals, n=4) if len(vals) >= 4 else None
    return {"n": len(vals), "median": round(statistics.median(vals), 3),
            "q1": round(q[0], 3) if q else None,
            "q3": round(q[2], 3) if q else None,
            "min": round(vals[0], 3), "max": round(vals[-1], 3)}


def main():
    ap = argparse.ArgumentParser(description="型データベースの取り込み・集計")
    ap.add_argument("--lang", default="ja", help="言語フォルダ（既定 ja）")
    args = ap.parse_args()
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        args.lang)
    manifest_path = os.path.join(base, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    entries = []
    for item in manifest["videos"]:
        with open(item["metrics"], encoding="utf-8") as f:
            metrics = json.load(f)
        prof = vp.profile_from_metrics(metrics)
        entry = {"video_id": item["video_id"], "title": metrics.get("title"),
                 "genre": item["genre"], "lang": args.lang,
                 "metrics": item["metrics"]}
        for k in FIELDS:
            entry[k] = prof.get(k)
        entry["final_ratios"] = prof.get("final_ratios")
        entry["delta_source"] = prof.get("delta_source")
        entries.append(entry)

    index = {"lang": args.lang, "count": len(entries), "videos": entries}
    with open(os.path.join(base, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    agg = {"lang": args.lang, "count": len(entries), "by_genre": {},
           "overall": {}}
    genres = sorted({e["genre"] for e in entries})
    for g in genres + [None]:
        subset = [e for e in entries if g is None or e["genre"] == g]
        target = agg["overall"] if g is None else agg["by_genre"].setdefault(g, {})
        target["count"] = len(subset)
        for k in FIELDS:
            target[k] = dist([e.get(k) for e in subset])
    with open(os.path.join(base, "aggregate.json"), "w",
              encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"OK: {args.lang} に {len(entries)} 件を取り込みました"
          f" -> {base}/index.json, aggregate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
