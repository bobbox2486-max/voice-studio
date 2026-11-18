#!/usr/bin/env python3
"""voice_profile.py — 文単位の「型」プロファイル抽出と比較（共有モジュール）。

video_analyzer.py（測定側）と voice_adjust.py（生成側）の対になる
中核モジュールです。従来の全体統計（話速・間・ピッチ中央値）に加えて、
**文単位の特徴**を「型プロファイル」として扱います。

- 文ごとのピッチの動き：開始（冒頭0.3秒）→終了（末尾0.3秒）の変化
  （delta、半音）、文末の上がり／下がり／横ばいの分類
- 文ごとの抑揚の幅：文内ピッチレンジ（p10〜p90、半音）
- 文の長さと間の分布（中央値＋四分位）

測定の物差しは video_analyzer.py と同一に揃えてあります
（F0は Praat／parselmouth、強度52dB超のフレームのみ、統計は半音の
相対値、文末判定は ±1.0半音）。

提供するもの:
  extract_f0_frames(wav)          有声フレーム列の抽出
  sentence_features(frames, sents) 文単位の特徴リスト
  aggregate_profile(...)          特徴＋全体統計 → プロファイル
  profile_from_metrics(metrics)   analyzer の metrics.json → プロファイル
  profile_from_generated(wav, script) 生成音声 → プロファイル
  similarity(ref, gen)            類似度スコア（項目別の距離つき）
  prosody_for_sheet(profile)      キャラシート用 prosody 辞書の生成

終了コード（CLIとして使った場合）: 0=成功 / 2=入力エラー / 5=測定エラー
"""

import json
import math
import os
import re
import statistics
import sys
import wave

# ---- 測定モデル定数（video_analyzer.py と同一に保つこと） ----
INTENSITY_GATE_DB = 52.0  # F0統計に使う声フレームの強度下限
PITCH_FLOOR = 75
PITCH_CEILING = 600
TIME_STEP = 0.01
FINAL_MOVE_THRESHOLD_ST = 1.0  # 文末の上がり／下がり判定の閾値（半音）
HEAD_TAIL_SEC = 0.3  # 文の開始／終了を測る窓（秒）

PUNCT_RE = re.compile(r"[\s、。！？!?・…「」『』（）()]")


class ProfileError(Exception):
    def __init__(self, message, code=5):
        super().__init__(message)
        self.code = code


def count_chars(text):
    """文字数（空白と句読点・記号を除く）。video_analyzer と同一基準。"""
    return len(PUNCT_RE.sub("", text))


def hz_to_st(freq, ref):
    return 12.0 * math.log2(freq / ref)


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def classify_final(delta_st):
    """文末の動き（開始→終了のピッチ変化、半音）→ rise/fall/flat。"""
    if delta_st is None:
        return None
    if delta_st >= FINAL_MOVE_THRESHOLD_ST:
        return "rise"
    if delta_st <= -FINAL_MOVE_THRESHOLD_ST:
        return "fall"
    return "flat"


# ---------------------------------------------------------------------------
# F0フレーム抽出（video_analyzer.extract_f0_frames と同一手法）
# ---------------------------------------------------------------------------

def extract_f0_frames(wav_path):
    """音声 → {"times": [...], "f0s": [...], "duration": float}。
    強度ゲート（INTENSITY_GATE_DB 超）を通った有声フレームのみ。"""
    try:
        import parselmouth
    except ImportError:
        raise ProfileError(
            "praat-parselmouth が導入されていません。README.md の導入手順を"
            "参照してください", 5)

    snd = parselmouth.Sound(wav_path)
    if snd.n_channels > 1:
        snd = snd.extract_channel(1)
    duration = snd.duration
    pitch = snd.to_pitch(time_step=TIME_STEP, pitch_floor=PITCH_FLOOR,
                         pitch_ceiling=PITCH_CEILING)
    intensity = snd.to_intensity(time_step=TIME_STEP)

    import bisect
    freqs = pitch.selected_array["frequency"]
    i_times = [intensity.get_time_from_frame_number(i)
               for i in range(1, intensity.n_frames + 1)]
    i_vals = intensity.values[0]
    times, f0s = [], []
    for i in range(1, pitch.n_frames + 1):
        f0 = float(freqs[i - 1])
        if f0 <= 0:
            continue
        t = pitch.get_time_from_frame_number(i)
        j = bisect.bisect_left(i_times, t)
        nearest = None
        for k in (j - 1, j):
            if 0 <= k < len(i_vals):
                if nearest is None or abs(i_times[k] - t) < abs(i_times[nearest] - t):
                    nearest = k
        if nearest is None or float(i_vals[nearest]) <= INTENSITY_GATE_DB:
            continue
        times.append(t)
        f0s.append(f0)

    if len(f0s) < 50:
        raise ProfileError(
            f"声とみなせるフレームが不足しています（{len(f0s)}件）。"
            "BGMが大きい・無音が多い等の可能性があります", 5)
    return {"times": times, "f0s": f0s, "duration": duration}


# ---------------------------------------------------------------------------
# 文単位の特徴
# ---------------------------------------------------------------------------

def sentence_features(frames, sentences):
    """有声フレームと文リスト → 文単位の特徴リスト。

    sentences: [{"start": float, "est_end": float}, ...]
    戻り値: 文と同じ並びのリスト。各要素は None（F0不足）か
    {median_st, span_st, start_st, end_st, delta_st, final_class}。
    半音は全体F0中央値からの相対値。
    """
    times = frames["times"]
    f0s = frames["f0s"]
    if not f0s:
        return [None] * len(sentences)
    median_hz = statistics.median(f0s)

    def frames_in(t0, t1):
        return [f for tt, f in zip(times, f0s) if t0 <= tt <= t1]

    out = []
    for s in sentences:
        vals = frames_in(s["start"], s["est_end"])
        if len(vals) < 5:
            out.append(None)
            continue
        v_st = [hz_to_st(f, median_hz) for f in vals]
        head = frames_in(s["start"],
                         min(s["start"] + HEAD_TAIL_SEC, s["est_end"]))
        tail = frames_in(max(s["est_end"] - HEAD_TAIL_SEC, s["start"]),
                         s["est_end"])
        start_st = (hz_to_st(statistics.median(head), median_hz)
                    if len(head) >= 2 else None)
        end_st = (hz_to_st(statistics.median(tail), median_hz)
                  if len(tail) >= 2 else None)
        delta = (end_st - start_st
                 if start_st is not None and end_st is not None else None)
        out.append({
            "median_st": hz_to_st(statistics.median(vals), median_hz),
            "span_st": pct(v_st, 0.90) - pct(v_st, 0.10),
            "start_st": start_st,
            "end_st": end_st,
            "delta_st": delta,
            "final_class": classify_final(delta),
        })
    return out


def aggregate_profile(per_sentence, speed_speech=None, gap_values=None,
                      len_values=None, f0_median_hz=None):
    """文単位の特徴＋全体統計 → 型プロファイル（dict）。"""
    valid = [p for p in per_sentence if p is not None]
    deltas = [p["delta_st"] for p in valid if p["delta_st"] is not None]
    spans = [p["span_st"] for p in valid if p["span_st"] is not None]
    classes = [p["final_class"] for p in valid if p["final_class"]]
    ratios = None
    if classes:
        n = len(classes)
        ratios = {k: classes.count(k) / n for k in ("rise", "fall", "flat")}
    gaps = [g for g in (gap_values or []) if g is not None]
    lens = [x for x in (len_values or []) if x is not None]
    return {
        "speed_speech": speed_speech,
        "gap_median": statistics.median(gaps) if gaps else None,
        "gap_p25": pct(gaps, 0.25),
        "gap_p75": pct(gaps, 0.75),
        "len_median": statistics.median(lens) if lens else None,
        "len_p25": pct(lens, 0.25),
        "len_p75": pct(lens, 0.75),
        "f0_median_hz": f0_median_hz,
        "delta_median_st": statistics.median(deltas) if deltas else None,
        "span_median_st": statistics.median(spans) if spans else None,
        "final_ratios": ratios,
        "n_sentences": len(per_sentence),
        "n_sentences_with_f0": len(valid),
    }


# ---------------------------------------------------------------------------
# プロファイルの作り方（1）: analyzer の metrics.json から
# ---------------------------------------------------------------------------

def profile_from_metrics(metrics):
    """video_analyzer.py の metrics.json → 型プロファイル。

    新形式（f0.per_sentence と sentences を含む）では文単位の特徴から
    集計し、旧形式では全体統計（final_rise/fall/flat の件数、span_all）
    から可能な範囲で組み立てる（delta_median_st は旧形式では
    「です・ます」文のみの desumasu_final_median で代用し、
    delta_source に出所を明記する）。
    """
    cap = metrics.get("captions") or {}
    f0 = metrics.get("f0") or {}
    per = f0.get("per_sentence")
    sentences = metrics.get("sentences") or []
    gap_values = [s.get("gap_after") for s in sentences
                  if s.get("gap_after") is not None]
    len_values = [s.get("n_chars") for s in sentences
                  if s.get("n_chars") is not None]
    if per:
        prof = aggregate_profile(
            per,
            speed_speech=cap.get("speed_speech"),
            gap_values=gap_values or None,
            len_values=len_values or None,
            f0_median_hz=f0.get("median_hz"))
        prof["delta_source"] = "per_sentence"
        if prof["gap_median"] is None:
            prof["gap_median"] = cap.get("gap_median")
            prof["gap_p25"] = cap.get("gap_p25")
            prof["gap_p75"] = cap.get("gap_p75")
        if prof["len_median"] is None:
            prof["len_median"] = cap.get("len_median")
            prof["len_p25"] = cap.get("len_p25")
            prof["len_p75"] = cap.get("len_p75")
        return prof
    # 旧形式フォールバック
    n_rise = f0.get("final_rise") or 0
    n_fall = f0.get("final_fall") or 0
    n_flat = f0.get("final_flat") or 0
    total = n_rise + n_fall + n_flat
    ratios = ({"rise": n_rise / total, "fall": n_fall / total,
               "flat": n_flat / total} if total else None)
    return {
        "speed_speech": cap.get("speed_speech"),
        "gap_median": cap.get("gap_median"),
        "gap_p25": cap.get("gap_p25"),
        "gap_p75": cap.get("gap_p75"),
        "len_median": cap.get("len_median"),
        "len_p25": cap.get("len_p25"),
        "len_p75": cap.get("len_p75"),
        "f0_median_hz": f0.get("median_hz"),
        "delta_median_st": f0.get("desumasu_final_median"),
        "delta_source": ("desumasu_only" if f0.get("desumasu_final_median")
                         is not None else None),
        "span_median_st": f0.get("span_all"),
        "final_ratios": ratios,
        "n_sentences": cap.get("n_sentences"),
        "n_sentences_with_f0": f0.get("n_sentences_with_f0"),
    }


# ---------------------------------------------------------------------------
# プロファイルの作り方（2）: 生成音声（WAV）から
# ---------------------------------------------------------------------------

def _voiced_flags(wav_path):
    """10ms RMS走査 → (有声フラグ列, フレーム長秒)。"""
    with wave.open(wav_path, "rb") as w:
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    import array
    samples = array.array("h")
    samples.frombytes(raw)
    frame = rate // 100  # 10ms
    n_frames = len(samples) // frame
    rms = []
    for i in range(n_frames):
        chunk = samples[i * frame:(i + 1) * frame]
        if not chunk:
            break
        rms.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)))
    if not rms:
        return [], 0.01
    threshold = max(max(rms) * 0.04, 60.0)
    return [r > threshold for r in rms], 0.01


def voiced_seconds(wav_path):
    """有声フレームの合計秒数（voice_adjust の VAD基準と同一）。"""
    flags, flen = _voiced_flags(wav_path)
    return sum(flags) * flen


def segment_utterances(wav_path, min_silence_sec=0.25, expected_count=None):
    """WAV → 発話区間 [(start, end), ...]。

    10ms RMS走査で有声を判定し、min_silence_sec 以上の無音で区切る。
    expected_count（台本の文数）が分かっている場合は、区間が文中の
    ポーズで割れて数が多すぎるときに「最も短い間の両側を結合」を
    繰り返して文数に揃える（文間の挿入無音は文中ポーズより長いはず、
    という生成音声の前提に基づく）。
    """
    voiced, _flen = _voiced_flags(wav_path)
    if not voiced:
        return []
    min_sil = int(min_silence_sec / 0.01)
    spans = []
    start = None
    silence = 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_sil:
                spans.append((start * 0.01, (i - silence + 1) * 0.01))
                start = None
                silence = 0
    if start is not None:
        spans.append((start * 0.01, len(voiced) * 0.01))
    # ごく短い断片（0.15秒未満）はノイズとして落とす
    spans = [(a, b) for a, b in spans if b - a >= 0.15]
    if expected_count and len(spans) > expected_count:
        while len(spans) > expected_count:
            gaps = [spans[i + 1][0] - spans[i][1]
                    for i in range(len(spans) - 1)]
            j = gaps.index(min(gaps))
            spans[j:j + 2] = [(spans[j][0], spans[j + 1][1])]
    return spans


def profile_from_generated(wav_path, script_sentences):
    """生成音声と台本の文リスト → 型プロファイル。

    発話区間を検出して台本の文と順番に対応づけ、話速（VAD基準）・
    間・文単位のピッチ特徴を測る。検出区間数と台本の文数が違う場合は
    先頭から min(区間数, 文数) 件だけ対応づけ、n_mismatch に差を記録する。
    """
    spans = segment_utterances(wav_path,
                               expected_count=len(script_sentences))
    if not spans:
        raise ProfileError("生成音声から発話区間を検出できませんでした", 5)
    frames = extract_f0_frames(wav_path)
    n = min(len(spans), len(script_sentences))
    sents = [{"start": spans[i][0], "est_end": spans[i][1]}
             for i in range(n)]
    feats = sentence_features(frames, sents)
    # 話速は生成側（voice_adjust）と同一の物差し：有声フレーム合計
    # （VAD基準）で全文字数を割る
    speech_sec = voiced_seconds(wav_path)
    total_chars = sum(count_chars(t) for t in script_sentences)
    speed = (total_chars / (speech_sec / 60.0)) if speech_sec > 0 else None
    gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
    lens = [count_chars(t) for t in script_sentences[:n]]
    f0_med = (statistics.median(frames["f0s"]) if frames["f0s"] else None)
    prof = aggregate_profile(feats, speed_speech=speed, gap_values=gaps,
                             len_values=lens, f0_median_hz=f0_med)
    prof["delta_source"] = "per_sentence"
    prof["n_utterances_detected"] = len(spans)
    prof["n_mismatch"] = abs(len(spans) - len(script_sentences))
    return prof


# ---------------------------------------------------------------------------
# 類似度（物差し）
# ---------------------------------------------------------------------------

# 項目ごとの重みと、「距離1.0（＝0点）」になるズレの大きさ。
# 距離は ズレ / スケール を 0〜1 にクランプしたもの。
SIMILARITY_SPEC = {
    "speed": {"weight": 0.20, "scale": 0.30, "label": "話速（相対差）"},
    "gap": {"weight": 0.15, "scale": 0.60, "label": "文間の間（秒）"},
    "delta": {"weight": 0.20, "scale": 3.0, "label": "文のピッチ変化・中央値（半音）"},
    "span": {"weight": 0.15, "scale": 4.0, "label": "文内の抑揚の幅・中央値（半音）"},
    "final_ratios": {"weight": 0.20, "scale": 1.0, "label": "文末パターン比率（L1距離/2）"},
    "f0_median": {"weight": 0.10, "scale": 6.0, "label": "ピッチ中央値（半音差）"},
}


def _ratio_distance(a, b):
    if a is None or b is None:
        return None
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0))
               for k in ("rise", "fall", "flat")) / 2.0


def similarity(ref, gen):
    """お手本プロファイルと生成プロファイルの類似度。

    戻り値: {"score": 0〜100, "dimensions": {項目: {ref, gen, distance,
    score}}, "missing": [測れなかった項目]}。
    測れない項目は重みから外し、残りの重みで正規化する。
    """
    raw = {}
    if ref.get("speed_speech") and gen.get("speed_speech"):
        raw["speed"] = (ref["speed_speech"], gen["speed_speech"],
                        abs(gen["speed_speech"] - ref["speed_speech"])
                        / ref["speed_speech"])
    if ref.get("gap_median") is not None and gen.get("gap_median") is not None:
        raw["gap"] = (ref["gap_median"], gen["gap_median"],
                      abs(gen["gap_median"] - ref["gap_median"]))
    if (ref.get("delta_median_st") is not None
            and gen.get("delta_median_st") is not None):
        raw["delta"] = (ref["delta_median_st"], gen["delta_median_st"],
                        abs(gen["delta_median_st"] - ref["delta_median_st"]))
    if (ref.get("span_median_st") is not None
            and gen.get("span_median_st") is not None):
        raw["span"] = (ref["span_median_st"], gen["span_median_st"],
                       abs(gen["span_median_st"] - ref["span_median_st"]))
    rd = _ratio_distance(ref.get("final_ratios"), gen.get("final_ratios"))
    if rd is not None:
        raw["final_ratios"] = (ref["final_ratios"], gen["final_ratios"], rd)
    if ref.get("f0_median_hz") and gen.get("f0_median_hz"):
        raw["f0_median"] = (ref["f0_median_hz"], gen["f0_median_hz"],
                            abs(hz_to_st(gen["f0_median_hz"],
                                         ref["f0_median_hz"])))
    dims = {}
    missing = []
    wsum = 0.0
    acc = 0.0
    for key, spec in SIMILARITY_SPEC.items():
        if key not in raw:
            missing.append(key)
            continue
        r, g, dist_raw = raw[key]
        dist = min(1.0, dist_raw / spec["scale"])
        score = 100.0 * (1.0 - dist)
        dims[key] = {"label": spec["label"], "ref": r, "gen": g,
                     "distance": dist, "score": score}
        wsum += spec["weight"]
        acc += spec["weight"] * score
    total = acc / wsum if wsum > 0 else None
    return {"score": total, "dimensions": dims, "missing": missing}


# ---------------------------------------------------------------------------
# キャラシート連携
# ---------------------------------------------------------------------------

def prosody_for_sheet(profile):
    """プロファイル → キャラシートの prosody セクション。

    文単位の型（文末パターンの比率、文ごとのピッチ変化と抑揚の幅の
    中央値）をシートに持たせるための辞書。生成側（voice_adjust）は
    apply_contour が真で、目標値がある場合に文単位のピッチ整形を行う。

    安全策（2026-11-18、デモ実証の教訓）: apply_contour は既定オフ。
    文単位整形は local 土台で不自然さ（ボブ評価「怖い」）を招いたため、
    型データは記録するが整形の有効化は opt-in（シートで明示、または
    synthesize の --apply-contour）に限る。
    """
    delta = profile.get("delta_median_st")
    span = profile.get("span_median_st")
    return {
        "delta_median_st": (round(delta, 2) if delta is not None else None),
        "span_median_st": (round(span, 2) if span is not None else None),
        "final_ratios": profile.get("final_ratios"),
        "delta_source": profile.get("delta_source"),
        "apply_contour": False,
    }


# ---------------------------------------------------------------------------
# CLI（単体確認用）: metrics.json → プロファイル表示
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="metrics.json から文単位の型プロファイルを表示する")
    ap.add_argument("metrics", help="video_analyzer.py の metrics.json")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.metrics):
        print(f"エラー: metrics.json が見つかりません: {args.metrics}",
              file=sys.stderr)
        return 2
    with open(args.metrics, encoding="utf-8") as f:
        metrics = json.load(f)
    prof = profile_from_metrics(metrics)
    print(json.dumps(prof, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
