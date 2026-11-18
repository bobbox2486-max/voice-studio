#!/usr/bin/env python3
"""video_analyzer.py — YouTube動画の「音声抽出＋読み分析」ツール

URL（または動画ID）を渡すと、字幕と音声を取得し、既存のベンチマーク分析
（~/workspace/ria-study/akachan-mama-analysis*.md）と同一の物差しで
読みの型を定量測定して Markdown レポートを出力する。

測定手法の出典と前提（既存レポート akachan-mama-analysis.md /
akachan-mama-analysis-2.md, 2026-11-17 のベンチマーク分析と同一の物差し）:
- 字幕は日本語（公式字幕があれば公式、なければ自動字幕 ja）の json3 を
yt-dlp で取得する。字幕に単語単位の時刻が無い形式（pb3等）では、
音声を faster-whisper で書き起こし、その単語タイムスタンプで単語時刻を
復元する（実音声の無声区間が「間」になる）。
- 日本語字幕自体が存在しない動画では、--asr auto（既定）の場合に限り
音声全体を faster-whisper で書き起こして文面と時刻の土台にする
（ASRフォールバック。align_mode="asr_full" としてレポートと
metrics.json に明記し、字幕由来の数値とは区別する）。
--asr never では従来通り字幕なしをエラーにする。
- 単語の推定発話長モデル: 0.13秒/字 + 0.06秒。推定無音 0.50秒以上を
文の区切りとし、文間の「間」はその推定無音で測る（誤差 ±0.2〜0.3秒）。
- F0 は Praat（parselmouth）で抽出。強度 52dB 超のフレームのみを声として
ゲートし、統計は中央値と「半音」単位の相対値で取る（Hz絶対値は使わない）。
- 測定モデル定数は本ファイル冒頭の定数ブロックに分離してある。
既存レポートとの横比較が前提のため、定数の変更は分析の物差しを
変えることと同義（変更時はレポート側にもその旨を明記すること）。

使い方:
python3 video_analyzer.py https://www.youtube.com/watch?v=XXXX [--out DIR]

終了コード: 0=成功 / 2=入力エラー / 3=メタ情報取得失敗
4=字幕なし（--asr never時）・字幕取得失敗 / 5=音声取得失敗 / 6=解析失敗
"""

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime

# ---- 測定モデル定数（既存レポートと同一に保つこと） ----
WORD_CHAR_SEC = 0.13 # 単語の推定発話長：1字あたり秒
WORD_BASE_SEC = 0.06 # 単語の推定発話長：固定分
SENTENCE_SILENCE = 0.50 # 文の区切りとする推定無音（秒）
SMALL_GAP_MIN = 0.30 # 文中の小さな間とみなす下限（秒）
LONG_GAP_SEC = 1.2 # 「長い間」として一覧化する下限（秒）
SCENE_GAP_SEC = 2.5 # 場面級の間の下限（秒）
INTENSITY_GATE_DB = 52.0 # F0統計に使う声フレームの強度下限
PITCH_FLOOR = 75
PITCH_CEILING = 600
TIME_STEP = 0.01

VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})")
BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


FULL_L = chr(12304)  # 全角の開き括弧
FULL_R = chr(12305)  # 全角の閉じ括弧


def is_annotation(text):
    """字幕注釈（音楽 等）だけの断片かどうか。"""
    t = text.strip()
    return len(t) >= 2 and ((t[0] == "[" and t[-1] == "]")
                            or (t[0] == FULL_L and t[-1] == FULL_R))


class AnalyzeError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def extract_video_id(target):
    if BARE_ID_RE.match(target):
        return target
    m = VIDEO_ID_RE.search(target)
    if m:
        return m.group(1)
    raise AnalyzeError(
        f"動画IDを特定できませんでした: {target}／対処: YouTubeの動画URL"
        "（watch / youtu.be / shorts 形式）か、11字の動画IDを指定してください",
        2)


def find_ytdlp():
    """yt-dlp の起動コマンド（リスト）を返す。見つからなければエラー。"""
    exe = shutil.which("yt-dlp") or os.path.expanduser("~/.local/bin/yt-dlp")
    if os.path.isfile(exe):
        return [exe]
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        raise AnalyzeError(
            "yt-dlp が見つかりません。導入方法は README.md を参照してください"
            "（python3 -m pip install --user --break-system-packages yt-dlp）", 3)


def run(cmd, timeout, what):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AnalyzeError(f"{what} がタイムアウトしました（{timeout}秒）", 3)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else "詳細不明"
        raise AnalyzeError(f"{what} に失敗しました: {detail}", 3)
    return proc.stdout


PLAYER_CLIENT_ARGS = ["--extractor-args", "youtube:player_client=android"]


def run_ytdlp(ytdlp, args, timeout, what):
    """yt-dlp 実行。androidクライアント指定で試し、失敗したら既定で再試行する。
    （環境によって既定クライアントが 'not available' になるための対策）"""
    try:
        return run(ytdlp + PLAYER_CLIENT_ARGS + args, timeout, what)
    except AnalyzeError:
        return run(ytdlp + args, timeout, what)


def fetch_metadata(ytdlp, video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        out = run_ytdlp(ytdlp, ["-J", "--no-download", "--no-playlist", url],
                        180, "動画メタ情報の取得")
    except AnalyzeError as e:
        raise AnalyzeError(
            f"{e}／対処: URLが正しいか、動画が公開中（削除・非公開でない）か、"
            "ネットワーク接続を確認してください", 3)
    try:
        info = json.loads(out)
    except json.JSONDecodeError:
        raise AnalyzeError("動画メタ情報の解析に失敗しました", 3)
    return info


def pick_subtitle_track(info):
    """(種別, 言語コード) を返す。公式字幕 ja > 自動字幕 ja-orig > 自動 ja。"""
    subs = info.get("subtitles") or {}
    for lang in ("ja", "ja-JP"):
        if lang in subs:
            return ("公式字幕", lang)
    auto = info.get("automatic_captions") or {}
    for lang in ("ja-orig", "ja", "ja-JP"):
        if lang in auto:
            return ("自動字幕", lang)
    return (None, None)


def fetch_subtitles(ytdlp, video_id, kind, lang, workdir):
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = ytdlp + ["--skip-download", "--sub-format", "json3",
                   "--sub-langs", lang, "-o", os.path.join(workdir, "%(id)s"),
                   url]
    if kind == "公式字幕":
        cmd.insert(1, "--write-subs")
    else:
        cmd.insert(1, "--write-auto-subs")
    try:
        # run_ytdlp は ytdlp プレフィックス込みで組み立てるため差し替える
        rest = cmd[len(ytdlp):]
        run_ytdlp(ytdlp, rest, 300, "字幕の取得")
    except AnalyzeError as e:
        raise AnalyzeError(str(e), 4)
    for name in sorted(os.listdir(workdir)):
        if name.endswith(".json3"):
            return os.path.join(workdir, name)
    raise AnalyzeError(
        f"日本語字幕（{kind}: {lang}）のファイルが取得できませんでした", 4)


def parse_json3(path):
    """json3字幕 → (単語列, ウィンドウ列)。

    単語列: [(text, start_or_None), ...]（出現順）。
    ウィンドウ列: [{"start": float, "end": float, "texts": [text, ...]}]。
    字幕形式によっては単語単位の時刻（tOffsetMs）が無い（pb3形式等）。
    その場合は start=None になり、後段で音声の発話区間へ割り付けて
    単語時刻を復元する。注釈のみの断片は除外する。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    word_seq = []
    windows = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        t0 = event.get("tStartMs", 0) / 1000.0
        dur = (event.get("dDurationMs") or 0) / 1000.0
        texts = []
        for seg in segs:
            text = (seg.get("utf8") or "").replace("\n", "").strip()
            if not text or is_annotation(text):
                continue
            off = seg.get("tOffsetMs")
            start = t0 + off / 1000.0 if off is not None else None
            texts.append(text)
            word_seq.append((text, start))
        if texts:
            windows.append({"start": t0, "end": t0 + dur, "texts": texts})
    return word_seq, windows


def transcribe_words_asr(wav_path, model_name, vad_filter=True):
    """faster-whisper で単語タイムスタンプ付き書き起こしを行う。
    戻り値: [(start, text, end), ...]（秒）。
    vad_filter は既定で有効（従来の測定との互換のため）。字幕なし動画の
    全文書き起こし経路（align_mode="asr_full"）では、VAD が歌唱の
    持続発声を非音声として落としてしまうため無効化して呼ぶ。
    モデルは ~/.cache/faster-whisper/<名前>/ に model.bin があれば
    そのローカルパスを優先する（HF Hub からの自動ダウンロードが
    使えない環境向け）。音声は PyAV を避けて stdlib の wave で読む。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise AnalyzeError(
            "faster-whisper が導入されていません。音声認識による"
            "書き起こし／単語時刻の復元が必要です／対処: "
            "python3 -m pip install --user --break-system-packages "
            "faster-whisper を実行してください（日本語字幕がある動画では "
            "--asr never で字幕ウィンドウ内分の近似測定にも切り替えられます）",
            6)
    import wave as wave_mod

    import numpy as np

    with wave_mod.open(wav_path, "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1:
            raise AnalyzeError(
                "音声が 16kHz モノラルではありません"
                f"（{wf.getframerate()}Hz / {wf.getnchannels()}ch）", 6)
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    local = os.path.expanduser(
        os.path.join("~", ".cache", "faster-whisper", model_name))
    model_ref = local if os.path.isfile(
        os.path.join(local, "model.bin")) else model_name
    try:
        model = WhisperModel(model_ref, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(
            audio, language="ja", word_timestamps=True,
            vad_filter=vad_filter)
        words = []
        for seg in segments:
            for w in seg.words or []:
                text = (w.word or "").strip()
                if text:
                    words.append((float(w.start), text, float(w.end)))
    except AnalyzeError:
        raise
    except Exception as e:
        raise AnalyzeError(
            f"音声認識（faster-whisper）に失敗しました: {e}／対処: "
            "モデルファイルの取得状態を確認してください"
            "（~/.cache/faster-whisper/<モデル名>/ に model.bin 等を置くと"
            "ローカルから読みます）", 6)
    if len(words) < 10:
        raise AnalyzeError(
            f"音声認識で得られた単語数が不足しています（{len(words)}件）。"
            "／対処: 声が主体の動画か確認してください", 6)
    words.sort(key=lambda w: w[0])
    return words

def align_words_window(windows):
    """フォールバック：字幕ウィンドウ内で単語を推定発話長で並べる。
    戻り値: [(start, text, None), ...]"""
    aligned = []
    for w in windows:
        t = w["start"]
        for text in w["texts"]:
            aligned.append((t, text, None))
            t += est_speech_len(text)
    aligned.sort(key=lambda x: x[0])
    return aligned


def fetch_audio(ytdlp, video_id, workdir):
    """音声のみをダウンロードし、ffmpeg でモノラル16kHz WAVに変換して返す。
    （yt-dlp の -x 変換は環境依存で失敗することがあるため自前で変換する）"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    outtmpl = os.path.join(workdir, "audio_src.%(ext)s")
    try:
        run_ytdlp(ytdlp, ["-f", "bestaudio/best", "--no-playlist",
                          "-o", outtmpl, url], 900, "音声の取得")
    except AnalyzeError as e:
        raise AnalyzeError(
            f"{e}／対処: ネットワーク接続と動画の公開状態を確認してください", 5)
    src = None
    for name in sorted(os.listdir(workdir)):
        if name.startswith("audio_src."):
            src = os.path.join(workdir, name)
            break
    if src is None:
        raise AnalyzeError("音声ファイルがダウンロードされませんでした", 5)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AnalyzeError(
            "ffmpeg が見つかりません。音声のWAV変換に必要です", 5)
    mono = os.path.join(workdir, "audio_mono.wav")
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", src,
         "-ac", "1", "-ar", "16000", mono],
        capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(mono):
        detail = (proc.stderr or "").strip().splitlines()
        raise AnalyzeError(
            "音声のWAV変換に失敗しました: "
            + (detail[-1] if detail else "詳細不明"), 5)
    return mono

# ---------------------------------------------------------------------------
# 文分割と字幕ベース測定（既存レポートと同一モデル）
# ---------------------------------------------------------------------------

PUNCT_RE = re.compile(r"[\s、。！？!?・…「」『』（）()]")


def count_chars(text):
    """文字数（空白と句読点・記号を除く）。既存レポートの字数基準に合わせる
    （字幕文には句読点がなく、ASR文には付くため、基準を揃える）。"""
    return len(PUNCT_RE.sub("", text))


def est_speech_len(text):
    return WORD_CHAR_SEC * count_chars(text) + WORD_BASE_SEC


def segment_sentences(words, speech_time_override=None):
    """単語列 [(start, text), ...] → 文リスト。
    各文: text / start / est_end / n_chars / gap_after。
    あわせて発話時間の合計と文中の小さな間の回数を返す。
    speech_time_override があれば発話時間はその値（音声の発話区間の実測）を
    使い、単語モデルによる推定は使わない。"""
    sentences = []
    speech_time = 0.0
    small_gaps = 0
    cur_texts = []
    cur_start = None
    cur_est_end = None

    for i, (start, text, wend) in enumerate(words):
        elen = est_speech_len(text)
        est_end = wend if wend is not None else start + elen
        next_start = words[i + 1][0] if i + 1 < len(words) else None
        if next_start is not None:
            if wend is not None:
                speech_time += max(0.0, wend - start)
            else:
                speech_time += max(0.0, min(elen, next_start - start))
            silence = next_start - est_end
        else:
            speech_time += (wend - start) if wend is not None else elen
            silence = None

        if cur_start is None:
            cur_start = start
            cur_texts = []
        cur_texts.append(text)
        cur_est_end = est_end

        if silence is not None and SMALL_GAP_MIN <= silence < SENTENCE_SILENCE:
            small_gaps += 1

        if silence is None or silence >= SENTENCE_SILENCE:
            joined = "".join(cur_texts)
            sentences.append({
                "text": joined,
                "start": cur_start,
                "est_end": cur_est_end,
                "n_chars": count_chars(joined),
            })
            cur_start = None
            cur_texts = []

    # 文間の間
    for i, s in enumerate(sentences):
        if i + 1 < len(sentences):
            s["gap_after"] = sentences[i + 1]["start"] - s["est_end"]
        else:
            s["gap_after"] = None
    if speech_time_override is not None:
        speech_time = speech_time_override
    return sentences, speech_time, small_gaps


ENDING_PATTERNS = [
    ("丁寧（です・ます）", re.compile(
        r"(です|ます|でした|ました|でしょう|ましょう|ません|ませんでした)"
        r"(ね|よ|よね|わ)?$")),
    ("ね・よ系", re.compile(r"(ね|よ|よね|わね|わよ|かな|かしら|な)$")),
    ("過去・断定（た・だ）", re.compile(r"(た|だ|だった|のだ|んだ)$")),
    ("言い切り", re.compile(r"(る|う|く|す|つ|ぬ|む|ぶ|ぐ|い)$")),
]


def classify_ending(text):
    tail = re.sub(r"[。！？!?、,\s]+$", "", text)
    for name, pat in ENDING_PATTERNS:
        if pat.search(tail):
            return name
    return "その他"


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


def fmt(x, digits=2):
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def caption_metrics(sentences, speech_time, small_gaps, duration):
    total_chars = sum(s["n_chars"] for s in sentences)
    lengths = [s["n_chars"] for s in sentences]
    gaps = [s["gap_after"] for s in sentences if s["gap_after"] is not None]
    dur_min = duration / 60.0
    m = {}
    m["n_sentences"] = len(sentences)
    m["total_chars"] = total_chars
    m["len_median"] = statistics.median(lengths) if lengths else None
    m["len_mean"] = statistics.mean(lengths) if lengths else None
    m["len_max"] = max(lengths) if lengths else None
    m["pct_le10"] = (100.0 * sum(1 for x in lengths if x <= 10) / len(lengths)
                     if lengths else None)
    m["pct_le20"] = (100.0 * sum(1 for x in lengths if x <= 20) / len(lengths)
                     if lengths else None)
    m["speed_overall"] = total_chars / dur_min if dur_min else None
    m["speech_time"] = speech_time
    m["speech_ratio"] = 100.0 * speech_time / duration if duration else None
    m["speed_speech"] = (total_chars / (speech_time / 60.0)
                         if speech_time > 0 else None)
    m["gap_median"] = statistics.median(gaps) if gaps else None
    m["gap_mean"] = statistics.mean(gaps) if gaps else None
    # 分布の四分位（型プロファイル用に追加した出力。既存フィールドは不変）
    m["len_p25"] = pct(lengths, 0.25) if lengths else None
    m["len_p75"] = pct(lengths, 0.75) if lengths else None
    m["gap_p25"] = pct(gaps, 0.25) if gaps else None
    m["gap_p75"] = pct(gaps, 0.75) if gaps else None
    buckets = {"0.5〜0.8秒": 0, "0.8〜1.2秒": 0, "1.2〜1.8秒": 0,
               "1.8〜2.5秒": 0, "2.5秒以上": 0}
    for g in gaps:
        if g < 0.8:
            buckets["0.5〜0.8秒"] += 1
        elif g < 1.2:
            buckets["0.8〜1.2秒"] += 1
        elif g < 1.8:
            buckets["1.2〜1.8秒"] += 1
        elif g < 2.5:
            buckets["1.8〜2.5秒"] += 1
        else:
            buckets["2.5秒以上"] += 1
    m["gap_buckets"] = buckets
    m["long_gaps"] = [
        {"after_index": i, "time": sentences[i]["est_end"],
         "gap": sentences[i]["gap_after"],
         "before": sentences[i]["text"][-15:],
         "after": sentences[i + 1]["text"][:15]}
        for i in range(len(sentences) - 1)
        if sentences[i]["gap_after"] is not None
        and sentences[i]["gap_after"] >= LONG_GAP_SEC
    ]
    m["scene_gaps"] = [g for g in m["long_gaps"] if g["gap"] >= SCENE_GAP_SEC]
    m["small_gaps"] = small_gaps
    # 導入と締め
    if sentences:
        m["first_start"] = sentences[0]["start"]
        m["title_gap"] = sentences[0]["gap_after"]
        m["last_text"] = sentences[-1]["text"]
        m["trailing"] = duration - sentences[-1]["est_end"]
    # 文末分類
    endings = {}
    for s in sentences:
        c = classify_ending(s["text"])
        endings[c] = endings.get(c, 0) + 1
    m["endings"] = endings
    # 速度の推移（3分割、間込み）
    thirds = [0, 0, 0]
    for s in sentences:
        idx = min(2, int(s["start"] / duration * 3)) if duration else 0
        thirds[idx] += s["n_chars"]
    third_min = dur_min / 3 if dur_min else 0
    m["third_speeds"] = [c / third_min if third_min else None for c in thirds]
    return m


# ---------------------------------------------------------------------------
# F0（ピッチ）分析：parselmouth、強度ゲート、半音の相対統計
# ---------------------------------------------------------------------------

def hz_to_st(freq, ref):
    return 12.0 * math.log2(freq / ref)


def extract_f0_frames(wav_path):
    """音声 → {"times": [...], "f0s": [...], "duration": float}。
    強度ゲート（INTENSITY_GATE_DB 超）を通った有声フレームのみを返す。
    この有声フレーム列は、字幕に単語時刻がない場合の単語割り付け
    （発話区間の検出）にも使う。"""
    try:
        import parselmouth
    except ImportError:
        raise AnalyzeError(
            "praat-parselmouth が導入されていません。README.md の導入手順を"
            "参照してください", 6)

    snd = parselmouth.Sound(wav_path)
    if snd.n_channels > 1:
        snd = snd.extract_channel(1)
    duration = snd.duration
    pitch = snd.to_pitch(time_step=TIME_STEP, pitch_floor=PITCH_FLOOR,
                         pitch_ceiling=PITCH_CEILING)
    intensity = snd.to_intensity(time_step=TIME_STEP)

    import bisect
    n_frames = pitch.n_frames
    freqs = pitch.selected_array["frequency"]
    i_times = [intensity.get_time_from_frame_number(i)
               for i in range(1, intensity.n_frames + 1)]
    i_vals = intensity.values[0]
    times, f0s = [], []
    for i in range(1, n_frames + 1):
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
        raise AnalyzeError(
            f"声とみなせるフレームが不足しています（{len(f0s)}件）。"
            "BGMが大きい・無音が多い等の可能性があります", 6)
    return {"times": times, "f0s": f0s, "duration": duration}


def f0_stats(frames, sentences):
    """有声フレームと文リスト → F0統計（全体中央値からの半音で表す）。"""
    times = frames["times"]
    f0s = frames["f0s"]
    duration = frames["duration"]
    median_hz = statistics.median(f0s)
    sts = [hz_to_st(f, median_hz) for f in f0s]
    result = {
        "median_hz": median_hz,
        "n_voiced_frames": len(f0s),
        "range_p10_p90_st": (pct(sts, 0.90) - pct(sts, 0.10)),
        "duration": duration,
    }

    # 30秒窓ごとの中央値（地の文の安定度）
    windows = []
    t = 0.0
    while t < duration:
        vals = [f for tt, f in zip(times, f0s) if t <= tt < t + 30.0]
        if len(vals) >= 20:
            windows.append(hz_to_st(statistics.median(vals), median_hz))
        t += 30.0
    result["window_medians_st"] = windows
    result["window_stability_st"] = (max(windows) - min(windows)
                                     if len(windows) >= 2 else None)

    # 文ごとの統計
    def frames_in(t0, t1):
        return [f for tt, f in zip(times, f0s) if t0 <= tt <= t1]

    per_sentence = []
    for s in sentences:
        vals = frames_in(s["start"], s["est_end"])
        if len(vals) < 5:
            per_sentence.append(None)
            continue
        med_st = hz_to_st(statistics.median(vals), median_hz)
        v_st = [hz_to_st(f, median_hz) for f in vals]
        span = pct(v_st, 0.90) - pct(v_st, 0.10)
        # 文末の動き：冒頭0.3秒と末尾0.3秒の中央値差
        head = frames_in(s["start"], min(s["start"] + 0.3, s["est_end"]))
        tail = frames_in(max(s["est_end"] - 0.3, s["start"]), s["est_end"])
        head_st = (hz_to_st(statistics.median(head), median_hz)
                   if len(head) >= 2 else None)
        tail_st = (hz_to_st(statistics.median(tail), median_hz)
                   if len(tail) >= 2 else None)
        move = None
        if head_st is not None and tail_st is not None:
            move = tail_st - head_st
        per_sentence.append({"median_st": med_st, "span_st": span,
                             "final_move_st": move,
                             "start_st": head_st, "end_st": tail_st})

    valid = [p for p in per_sentence if p is not None]
    result["n_sentences_with_f0"] = len(valid)

    # 地の文（中央値付近）とセリフ（高音クラスタ）の対比
    narration = [p["median_st"] for p in valid if abs(p["median_st"]) <= 2.0]
    dialogue = [p["median_st"] for p in valid if p["median_st"] >= 4.0]
    result["narration_center_st"] = (statistics.median(narration)
                                     if narration else None)
    result["narration_n"] = len(narration)
    result["dialogue_center_st"] = (statistics.median(dialogue)
                                    if dialogue else None)
    result["dialogue_n"] = len(dialogue)
    if narration and dialogue:
        result["dialogue_diff_st"] = (statistics.median(dialogue)
                                      - statistics.median(narration))
    else:
        result["dialogue_diff_st"] = None
    spans_nar = [p["span_st"] for p in valid if abs(p["median_st"]) <= 2.0]
    spans_dia = [p["span_st"] for p in valid if p["median_st"] >= 4.0]
    result["span_narration"] = statistics.mean(spans_nar) if spans_nar else None
    result["span_dialogue"] = statistics.mean(spans_dia) if spans_dia else None
    result["span_all"] = (statistics.mean([p["span_st"] for p in valid])
                          if valid else None)

    # 文末の上がり下がり
    moves = [p["final_move_st"] for p in valid
             if p["final_move_st"] is not None]
    result["final_rise"] = sum(1 for x in moves if x >= 1.0)
    result["final_fall"] = sum(1 for x in moves if x <= -1.0)
    result["final_flat"] = sum(1 for x in moves if -1.0 < x < 1.0)
    # です・ます文の文末の動き（字幕文との対応づけ、参考値）
    dm_moves = [p["final_move_st"] for s, p in zip(sentences, per_sentence)
                if p is not None and p["final_move_st"] is not None
                and classify_ending(s["text"]) == "丁寧（です・ます）"]
    result["desumasu_final_median"] = (statistics.median(dm_moves)
                                       if dm_moves else None)
    result["desumasu_n"] = len(dm_moves)

    # 文単位の詳細（型プロファイル用に追加した出力。既存フィールドは不変）
    detail = []
    for i, p in enumerate(per_sentence):
        if p is None:
            detail.append(None)
            continue
        mv = p["final_move_st"]
        if mv is None:
            cls = None
        elif mv >= 1.0:
            cls = "rise"
        elif mv <= -1.0:
            cls = "fall"
        else:
            cls = "flat"
        detail.append({"index": i, "median_st": p["median_st"],
                       "span_st": p["span_st"], "start_st": p["start_st"],
                       "end_st": p["end_st"], "delta_st": mv,
                       "final_class": cls})
    result["per_sentence"] = detail
    deltas = [d["delta_st"] for d in detail
              if d is not None and d["delta_st"] is not None]
    result["delta_median_st"] = (statistics.median(deltas)
                                 if deltas else None)
    n_moves = (result["final_rise"] + result["final_fall"]
               + result["final_flat"])
    result["final_ratios"] = (
        {"rise": result["final_rise"] / n_moves,
         "fall": result["final_fall"] / n_moves,
         "flat": result["final_flat"] / n_moves} if n_moves else None)
    return result


# ---------------------------------------------------------------------------
# レポート生成
# ---------------------------------------------------------------------------

ALIGN_MODE_TEXT = {
    "caption": "単語時刻は字幕に含まれる実時刻を使用した。",
    "asr": "字幕に単語単位の時刻が含まれない形式だったため、音声を"
           "faster-whisper（モデル: {model}）で書き起こし、その単語"
           "タイムスタンプで文分割と間を測定した。文のテキストも書き起こしに"
           "基づくため、字幕とは細部が異なる場合がある。",
    "window": "字幕に単語時刻がなく音声認識も使わなかったため、字幕ウィンドウ"
              "内で単語を推定発話長で内分した（参考値）。",
    "asr_full": "日本語字幕が存在しなかったため、音声全体を faster-whisper"
                "（モデル: {model}）で書き起こし、その単語タイムスタンプで"
                "文分割と間を測定した。文面はすべて書き起こし由来であり、"
                "日本語字幕由来の数値とは出所が異なる（横比較の際は注意）。"
                "この経路ではVAD（音声区間検出）を無効化している"
                "（歌唱の持続発声がVADで落ちるため）。",
}


def build_report(info, video_id, sub_kind, sub_lang, cap, f0,
                 align_mode="asr", whisper_model="base"):
    title = info.get("title") or "(タイトル不明)"
    channel = info.get("channel") or info.get("uploader") or "—"
    views = info.get("view_count")
    duration = info.get("duration") or 0
    upload = info.get("upload_date") or ""
    if len(upload) == 8:
        upload = f"{upload[:4]}-{upload[4:6]}-{upload[6:]}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# 読み分析レポート：{title}")
    lines.append("")
    lines.append(f"- チャンネル：{channel}")
    lines.append(f"- URL：https://www.youtube.com/watch?v={video_id}")
    lines.append(f"- 再生数：{views:,}回（取得時点）" if views is not None
                 else "- 再生数：—")
    lines.append(f"- 長さ：{int(duration // 60)}分{int(duration % 60)}秒"
                 f"（{duration:.1f}秒）／投稿：{upload or '—'}")
    if align_mode == "asr_full":
        lines.append(f"- 分析日時：{now}／字幕：使用せず"
                     "（日本語字幕なし。音声をASRで書き起こし）")
    else:
        lines.append(f"- 分析日時：{now}／字幕：{sub_kind}（{sub_lang}）")
    lines.append("")
    lines.append("## 1. 方法と制約（正直な限界）")
    lines.append("")
    if align_mode == "asr_full":
        lines.append(
            "- 日本語字幕（公式・自動とも）が存在しない動画のため、"
            "テキストと時刻は音声認識の書き起こしに基づく。"
            f"{ALIGN_MODE_TEXT[align_mode].format(model=whisper_model)}"
            "文分割には単語の推定発話長モデル"
            f"（{WORD_CHAR_SEC}秒/字＋{WORD_BASE_SEC}秒）を当て、"
            f"推定無音{SENTENCE_SILENCE}秒以上を文の区切りとした。"
            "間の秒数には±0.2〜0.3秒程度の誤差がありうる。")
        lines.append(
            "- **字幕ではなく機械書き起こしが文面の土台**のため誤認識が"
            "ありうる。文末分類などの文面ベースの指標は参考値。"
            "とくに歌唱（歌詞）では書き起こし精度が大きく下がる。"
            "時刻・ピッチ系の指標は比較的影響を受けにくい。")
    else:
        lines.append(
            "- 字幕は yt-dlp で取得した日本語字幕（json3）。"
            f"単語時刻の扱い：{ALIGN_MODE_TEXT[align_mode].format(model=whisper_model)}"
            "文分割には単語の推定発話長モデル"
            f"（{WORD_CHAR_SEC}秒/字＋{WORD_BASE_SEC}秒）を当て、"
            f"推定無音{SENTENCE_SILENCE}秒以上を文の区切りとした。"
            "間の秒数には±0.2〜0.3秒程度の誤差がありうる。")
    if sub_kind == "自動字幕":
        lines.append(
            "- **公式字幕がなく自動字幕を使用。誤認識がある**ため、"
            "文末分類などの文面ベースの指標は参考値。"
            "時刻情報と全体傾向の分析には使えるが、文面の細部は鵜呑みにしないこと。")
    lines.append(
        f"- F0は音声から Praat（parselmouth）で抽出。強度{INTENSITY_GATE_DB:.0f}dB超の"
        "フレームだけを声としてゲートし、統計は中央値中心に取った。"
        "BGMが常時入っている動画では分布の裾の信頼性が下がる。")
    lines.append(
        "- 対象が人間の声の場合、Hzの絶対値はリアのTTSと声質が違うため模倣対象にせず、"
        "**半音単位の相対的な動き**だけを見る。")
    lines.append(
        "- セリフ／地の文の区別はピッチの高さクラスタによる推定"
        "（地の文＝文中央値が全体中央値の±2半音以内／セリフ＝+4半音以上の高音クラスタ）。"
        "台本との照合はしていない。")
    lines.append("")
    lines.append("## 2. 字幕ベースの測定" if align_mode != "asr_full"
                 else "## 2. 書き起こしベースの測定（ASR）")
    lines.append("")
    lines.append("### 2.1 文の長さと読みの速さ")
    lines.append("")
    lines.append(f"- 文数：{cap['n_sentences']}文／総文字数：{cap['total_chars']}字")
    lines.append(
        f"- 一文の長さ：中央値 **{fmt(cap['len_median'], 0)}字**、"
        f"平均{fmt(cap['len_mean'], 1)}字、最長{cap['len_max']}字。"
        f"10字以下 {fmt(cap['pct_le10'], 0)}％、20字以下 {fmt(cap['pct_le20'], 0)}％。")
    lines.append(
        f"- 読み速さ：全体（間込み）**約{fmt(cap['speed_overall'], 0)}字/分**、"
        f"発話内 **約{fmt(cap['speed_speech'], 0)}字/分**。")
    lines.append(
        f"- 発話時間の割合：約{fmt(cap['speech_ratio'], 0)}％"
        f"（残り約{fmt(100 - cap['speech_ratio'], 0) if cap['speech_ratio'] is not None else '—'}％が間と絵・映像を見せる時間）。")
    ts = cap["third_speeds"]
    lines.append(
        f"- 速度の推移（間込み・3分割）：前半 {fmt(ts[0], 0)} → 中盤 {fmt(ts[1], 0)}"
        f" → 後半 {fmt(ts[2], 0)} 字/分。")
    lines.append("")
    lines.append("### 2.2 文間の「間」")
    lines.append("")
    lines.append(
        f"- 文間の間：中央値 **{fmt(cap['gap_median'])}秒**、"
        f"平均{fmt(cap['gap_mean'])}秒。文中の小さな間"
        f"（{SMALL_GAP_MIN}〜{SENTENCE_SILENCE}秒推定）は{cap['small_gaps']}回。")
    lines.append("")
    lines.append("| 間の長さ | 回数 |")
    lines.append("|---|---|")
    for k, v in cap["gap_buckets"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append(
        f"- 長い間（{LONG_GAP_SEC}秒以上）は{len(cap['long_gaps'])}回、"
        f"うち場面級（{SCENE_GAP_SEC}秒以上）は{len(cap['scene_gaps'])}回。")
    for g in cap["long_gaps"]:
        lines.append(
            f"  - {g['time']:.1f}秒地点：{g['gap']:.2f}秒"
            f"（前「…{g['before']}」／後「{g['after']}…」）")
    lines.append("")
    lines.append("### 2.3 文末表現（自動分類・参考値）")
    lines.append("")
    for name, _ in ENDING_PATTERNS + [("その他", None)]:
        if name in cap["endings"]:
            lines.append(f"- {name}：{cap['endings'][name]}文")
    lines.append("")
    lines.append("### 2.4 導入と締め")
    lines.append("")
    lines.append(
        f"- 導入：開始{fmt(cap.get('first_start'))}秒で最初の発話"
        f"（タイトルコール想定）→ その後の間 **{fmt(cap.get('title_gap'))}秒** → 本文。")
    lines.append(
        f"- 締め：最後の文「{cap.get('last_text', '')[:30]}」の後、"
        f"動画終了まで **約{fmt(cap.get('trailing'), 1)}秒** の余韻。")
    lines.append("")
    lines.append("## 3. F0（ピッチ）分析 ※相対値（半音）で読むこと")
    lines.append("")
    lines.append(
        f"- 声フレーム数（ゲート後）：{f0['n_voiced_frames']}件。"
        f"F0中央値 {fmt(f0['median_hz'], 0)}Hz（絶対値は参考）。"
        f"分布の幅（p10〜p90）は約{fmt(f0['range_p10_p90_st'], 1)}半音。")
    if f0["window_stability_st"] is not None:
        lines.append(
            f"- 地の文の安定度：30秒窓ごとの中央値の振れ幅は"
            f"約{fmt(f0['window_stability_st'], 1)}半音。")
    lines.append(
        f"- 文単位でF0が取れた文：{f0['n_sentences_with_f0']}文"
        f"（地の文クラスタ {f0['narration_n']}文／セリフクラスタ {f0['dialogue_n']}文）。")
    if f0["dialogue_diff_st"] is not None:
        lines.append(
            f"- **セリフは地の文より中央値で約{fmt(f0['dialogue_diff_st'], 1)}半音高い**"
            f"（セリフ側 {fmt(f0['dialogue_center_st'], 1)}半音 vs "
            f"地の文 {fmt(f0['narration_center_st'], 1)}半音、いずれも全体中央値比）。")
    else:
        lines.append("- セリフ／地の文のクラスタが分離できず、ピッチ差は測定不能。")
    lines.append(
        f"- 文内のピッチの動き幅（p10〜p90平均）：全体 {fmt(f0['span_all'], 1)}半音"
        f"／地の文 {fmt(f0['span_narration'], 1)}半音"
        f"／セリフ {fmt(f0['span_dialogue'], 1)}半音。")
    lines.append(
        f"- 文末のピッチの動き：上昇 {f0['final_rise']}文・下降 {f0['final_fall']}文"
        f"・横ばい {f0['final_flat']}文（冒頭0.3秒と末尾0.3秒の中央値差±1半音で判定）。")
    if f0["desumasu_final_median"] is not None:
        sign = "+" if f0["desumasu_final_median"] >= 0 else ""
        lines.append(
            f"- 「です・ます」文（{f0['desumasu_n']}文）の文末の動きは中央値で"
            f"{sign}{fmt(f0['desumasu_final_median'], 1)}半音"
            "（字幕文との対応づけによる参考値）。")
    if f0.get("delta_median_st") is not None:
        sign = "+" if f0["delta_median_st"] >= 0 else ""
        lines.append(
            f"- 文ごとのピッチ変化（開始→終了）の中央値："
            f"{sign}{fmt(f0['delta_median_st'], 1)}半音"
            "（文単位の型。詳細は metrics.json の f0.per_sentence）。")
    if f0.get("final_ratios"):
        r = f0["final_ratios"]
        lines.append(
            f"- 文末パターンの比率：上昇 {r['rise'] * 100:.0f}％・"
            f"下降 {r['fall'] * 100:.0f}％・横ばい {r['flat'] * 100:.0f}％。")
    lines.append("")
    lines.append("## 4. サマリー")
    lines.append("")
    lines.append("| 項目 | 測定値 |")
    lines.append("|---|---|")
    lines.append(f"| 文の長さ（中央値） | {fmt(cap['len_median'], 0)}字 |")
    lines.append(f"| 20字以下の割合 | {fmt(cap['pct_le20'], 0)}％ |")
    lines.append(f"| 読み速さ（間込み） | 約{fmt(cap['speed_overall'], 0)}字/分 |")
    lines.append(f"| 読み速さ（発話内） | 約{fmt(cap['speed_speech'], 0)}字/分 |")
    lines.append(f"| 発話時間の割合 | 約{fmt(cap['speech_ratio'], 0)}％ |")
    lines.append(f"| 文間の間（中央値） | {fmt(cap['gap_median'])}秒 |")
    lines.append(f"| 場面級の間 | {len(cap['scene_gaps'])}回（{SCENE_GAP_SEC}秒以上） |")
    lines.append(f"| タイトルコール後の間 | {fmt(cap.get('title_gap'))}秒 |")
    lines.append(f"| 締めの余韻 | 約{fmt(cap.get('trailing'), 1)}秒 |")
    if f0["dialogue_diff_st"] is not None:
        lines.append(f"| セリフのピッチ差 | 約+{fmt(f0['dialogue_diff_st'], 1)}半音 |")
    lines.append("")
    lines.append("※著作権配慮：本文の全文転載はせず、統計値と分析に必要な最小限の断片のみを用いた。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

class _StaticDir:
    """--work-dir 指定時に使う、削除しない一時ディレクトリ互換の入れ物。"""

    def __init__(self, path):
        self._path = path

    def __enter__(self):
        return self._path

    def __exit__(self, *exc):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="YouTube動画の音声抽出＋読み分析（字幕＋F0）")
    ap.add_argument("target", help="YouTubeのURLまたは動画ID")
    ap.add_argument("--out", default=None,
                    help="レポート出力先ディレクトリ"
                         "（既定: ./analysis-output/<動画ID>/）")
    ap.add_argument("--keep-work", action="store_true",
                    help="字幕・音声の中間ファイルを残す")
    ap.add_argument("--ytdlp", default=None,
                    help="yt-dlp 実行ファイルのパス（既定: PATHと"
                         "~/.local/bin、Pythonモジュールの順に探す）")
    ap.add_argument("--player-client", default="android",
                    help="yt-dlp の YouTube プレイヤークライアント"
                         "（既定: android。空文字で指定なし）")
    ap.add_argument("--work-dir", default=None,
                    help="中間ファイル用ディレクトリ"
                         "（既定: 一時ディレクトリ。指定時は削除しない）")
    ap.add_argument("--whisper-model", default="base",
                    help="faster-whisper のモデル名またはローカルパス"
                         "（既定: base）")
    ap.add_argument("--no-asr", action="store_true",
                    help="音声認識を使わず、字幕ウィンドウ内分で近似する"
                         "（精度は落ちる。--asr never と同じ）")
    ap.add_argument("--asr", choices=["auto", "always", "never"],
                    default=None,
                    help="音声認識（faster-whisper）の使い方。"
                         "auto（既定）: 字幕に単語時刻が無い場合に時刻復元へ"
                         "使い、日本語字幕自体が無い場合は音声全体の"
                         "書き起こしにフォールバックする。"
                         "always: 字幕があっても常に書き起こしを土台にする。"
                         "never: 音声認識を使わない（--no-asr と同じ）")
    args = ap.parse_args(argv)

    global PLAYER_CLIENT_ARGS
    if args.player_client:
        PLAYER_CLIENT_ARGS = ["--extractor-args",
                              f"youtube:player_client={args.player_client}"]
    else:
        PLAYER_CLIENT_ARGS = []

    try:
        video_id = extract_video_id(args.target)
        ytdlp = [args.ytdlp] if args.ytdlp else find_ytdlp()
        info = fetch_metadata(ytdlp, video_id)
        duration = info.get("duration")
        if not duration:
            raise AnalyzeError("動画の長さが取得できませんでした", 3)
        kind, lang = pick_subtitle_track(info)
        # ASRモードの決定：--asr 未指定時は --no-asr があれば never、
        # なければ auto。auto では日本語字幕が無い場合に限り、音声全体の
        # 書き起こしへフォールバックする（字幕あり動画の挙動は従来通り）。
        asr_mode = args.asr or ("never" if args.no_asr else "auto")
        use_full_asr = asr_mode == "always" or (kind is None
                                                and asr_mode == "auto")
        if kind is None and not use_full_asr:
            raise AnalyzeError(
                "日本語字幕（公式・自動とも）が見つかりませんでした。"
                "--asr never のため音声認識への切り替えもしません／対処: "
                "日本語字幕のある動画を選ぶか、--asr auto（既定）で実行して"
                "音声認識による書き起こし分析を許可してください", 4)

        out_dir = args.out or os.path.join(
            os.getcwd(), "analysis-output", video_id)
        os.makedirs(out_dir, exist_ok=True)

        if args.work_dir:
            os.makedirs(args.work_dir, exist_ok=True)
            tmp_ctx = _StaticDir(args.work_dir)
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="video_analyzer_")
        with tmp_ctx as tmp:
            word_seq, windows = [], []
            subdir = os.path.join(tmp, "subs")
            os.makedirs(subdir)
            if not use_full_asr:
                sub_path = fetch_subtitles(ytdlp, video_id, kind, lang,
                                           subdir)
                word_seq, windows = parse_json3(sub_path)
                if len(word_seq) < 10:
                    raise AnalyzeError(
                        f"字幕の単語数が不足しています（{len(word_seq)}件）。"
                        "／対処: 字幕のある動画か確認してください", 4)

            auddir = os.path.join(tmp, "audio")
            os.makedirs(auddir)
            wav = fetch_audio(ytdlp, video_id, auddir)

            # 単語時刻と文面の決定：字幕の実時刻 > 音声認識（faster-whisper）
            # の単語タイムスタンプ > 字幕ウィンドウ内での内分（--asr never
            # 時）。日本語字幕が無い場合（または --asr always）は音声全体の
            # 書き起こしが文面と時刻の土台になる（align_mode="asr_full"）。
            n_timed = sum(1 for _, s in word_seq if s is not None)
            if use_full_asr:
                # 字幕なし経路では VAD を切る（歌唱の持続発声が VAD で
                # 大量に落ちるため。実測：本編の歌動画で単語数が桁違いに
                # 減ることを確認済み）。字幕あり経路の物差しは変えない。
                aligned = transcribe_words_asr(wav, args.whisper_model,
                                               vad_filter=False)
                align_mode = "asr_full"
                speech_override = sum(e - s for s, _, e in aligned)
            elif word_seq and n_timed >= 0.9 * len(word_seq):
                aligned = sorted([(s, t, None) for t, s in word_seq],
                                 key=lambda w: w[0])
                align_mode = "caption"
                speech_override = None
            elif asr_mode != "never":
                aligned = transcribe_words_asr(wav, args.whisper_model)
                align_mode = "asr"
                speech_override = sum(e - s for s, _, e in aligned)
            else:
                aligned = align_words_window(windows)
                align_mode = "window"
                speech_override = None

            sentences, speech_time, small_gaps = segment_sentences(
                aligned, speech_override)
            if len(sentences) < 3:
                raise AnalyzeError(
                    f"文分割の結果が不足しています（{len(sentences)}文）", 4)
            cap = caption_metrics(sentences, speech_time, small_gaps,
                                  float(duration))
            try:
                frames = extract_f0_frames(wav)
            except AnalyzeError:
                raise
            except Exception as e:  # parselmouth 内部エラー等
                raise AnalyzeError(f"F0解析に失敗しました: {e}", 6)
            f0 = f0_stats(frames, sentences)

            if args.keep_work:
                keep = os.path.join(out_dir, "work")
                os.makedirs(keep, exist_ok=True)
                for src_dir in (subdir, auddir):
                    for name in os.listdir(src_dir):
                        shutil.copy2(os.path.join(src_dir, name),
                                     os.path.join(keep, name))

        report = build_report(info, video_id, kind, lang, cap, f0,
                              align_mode, args.whisper_model)
        report_path = os.path.join(out_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        metrics = {"video_id": video_id, "title": info.get("title"),
                   "align_mode": align_mode,
                   "text_source": ("asr" if align_mode == "asr_full"
                                   else "captions"),
                   "asr_mode": asr_mode,
                   "asr_vad_filter": (False if align_mode == "asr_full"
                                      else True if align_mode == "asr"
                                      else None),
                   "subtitle_kind": kind, "subtitle_lang": lang,
                   "captions": cap,
                   "f0": {k: v for k, v in f0.items()},
                   # 文ごとの長さ・間の列（型プロファイル用に追加した
                   # 出力。既存キーは不変）
                   "sentences": [
                       {"index": i, "n_chars": s["n_chars"],
                        "gap_after": (round(s["gap_after"], 3)
                                      if s["gap_after"] is not None
                                      else None),
                        "start": round(s["start"], 2),
                        "est_end": round(s["est_end"], 2)}
                       for i, s in enumerate(sentences)]}
        with open(os.path.join(out_dir, "metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2,
                      default=str)
        print(f"OK: レポートを出力しました -> {report_path}")
        return 0
    except AnalyzeError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
