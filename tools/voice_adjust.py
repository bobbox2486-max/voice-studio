#!/usr/bin/env python3
"""voice_adjust.py — キャラクリ方式の音声調整ツール（生成側）。

ゲームのキャラクタークリエイト画面のように「トーンプリセット＋
ステータス（0〜100）」で声の調整を選び、キャラシート（JSON）に保存して
再利用できる音声生成CLIです。video_analyzer.py（測定側）と対になる
ツールで、あちらが測った「読みの型」（話速・間・ピッチ）を、こちらの
生成音声へ数値で反映します。

サブコマンド:
  presets     トーン／キャラプリセットとステータスの対応表を表示する
  make-sheet  video_analyzer.py の metrics.json からキャラシートを作る
  synthesize  キャラを選ぶ→少し調整→音声が出る（シート直指定も可）

製品原則: 「簡単な入口 × 細かいカスタム ＝『この声が欲しかった』が
売り」。入口は「キャラを選ぶ → 高さ・速さ・間・強弱で大まかに調整 →
音声が出る」の3ステップに絞り、奥には overrides 等の数値指定を
「上級調整」として正規に用意する二層構造にする。

終了コード: 0=成功 / 2=入力エラー / 3=API・ネットワークエラー /
            4=生成・音声処理エラー / 5=測定エラー
認証情報（APIキー）はサロゲート経由でのみ使い、値の出力・保存はしません。
"""

import argparse
import base64
import copy
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import voice_profile as vp
except ImportError:  # 型プロファイル機能なしでも従来動作は守る
    vp = None

SAMPLE_RATE = 24000  # 内部処理はすべて 24kHz/mono/16bit WAV に統一
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# 文字数基準は video_analyzer.py と同一（空白・句読点を除く）
PUNCT_RE = re.compile(r"[\s、。！？!?・…「」『』（）()]")


class AdjustError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# キャラクリ定義（プリセットとステータスのマッピング）
# ---------------------------------------------------------------------------

TONE_PRESETS = {
    "明るい": {
        "style": "明るく爽やかな声で、笑顔が伝わるように読み聞かせてください",
        "voice": "Leda", "pitch_bias_st": 1.0, "speed_bias": 1.00,
    },
    "柔らかい": {
        "style": "柔らかく優しい声で、包み込むようにゆったり読み聞かせてください",
        "voice": "Aoede", "pitch_bias_st": 0.0, "speed_bias": 0.95,
    },
    "元気": {
        "style": "元気いっぱいに、はつらつとした声で読み聞かせてください",
        "voice": "Zephyr", "pitch_bias_st": 0.5, "speed_bias": 1.05,
    },
    "落ち着いた": {
        "style": "落ち着いて穏やかな声で、ゆったりと読み聞かせてください",
        "voice": "Callirrhoe", "pitch_bias_st": -1.0, "speed_bias": 0.92,
    },
    "クール": {
        "style": "クールでさらっとした声で、感情は抑えめに読んでください",
        "voice": "Autonoe", "pitch_bias_st": -1.5, "speed_bias": 0.98,
    },
    "甘い": {
        "style": "甘く可愛らしい声で、親しみを込めて読み聞かせしてください",
        "voice": "Kore", "pitch_bias_st": 1.5, "speed_bias": 0.97,
    },
    "カスタム": {
        "style": "", "voice": "Leda", "pitch_bias_st": 0.0, "speed_bias": 1.00,
    },
}

EXPR_WORDS = {"弱": 20, "中": 50, "強": 80}


def expr_style_tier(value):
    """強弱（0〜100）→ スタイルプロンプトの抑揚指示文。"""
    if value <= 33:
        return "感情は控えめに、淡々と読んでください"
    if value <= 66:
        return "自然な抑揚で読んでください"
    return "感情豊かに、大きく抑揚をつけて読んでください"


def stat_pitch_st(v):
    return (v - 50) / 50.0 * 6.0


def stat_speed_cpm(v):
    return 180.0 + v * 3.0


def stat_gap_sec(v):
    return 0.2 + v / 100.0 * 1.3


def cpm_to_stat(cpm):
    return max(0, min(100, round((cpm - 180.0) / 3.0)))


def gap_to_stat(sec):
    return max(0, min(100, round((sec - 0.2) / 1.3 * 100.0)))


def norm_stat(value, field):
    """ステータス値の正規化（数値 or 弱/中/強 → 0〜100）。"""
    if isinstance(value, str) and value in EXPR_WORDS:
        return EXPR_WORDS[value]
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise AdjustError(
            f"stats.{field} の値が不正です: {value!r}。"
            "0〜100の数値か、弱/中/強 で指定してください。", 2)
    if not 0 <= v <= 100:
        raise AdjustError(
            f"stats.{field} は 0〜100 の範囲で指定してください（現在 {v}）。", 2)
    return v


# ---------------------------------------------------------------------------
# キャラプリセット（選ぶだけで使える完成キャラ）
# ---------------------------------------------------------------------------

# local エンジン（既存 tts CLI）で使う声。全キャラ共通で、この声に
# ピッチ・速さ・間の後処理でキャラ差をつける（数値担保はこちら）。
LOCAL_VOICE_ID = "avocado_v2:vd2_r82_rep500_0476_v068_28k_g5k"

CHARACTER_PRESETS = {
    "リア": {
        "slug": "ria",
        "description": "TikTokトーク用。明るく親しみやすい標準キャラ",
        "tone_preset": "明るい",
        "style_prompt": "",
        "stats": {"pitch": 58, "speed": 55, "gap": 25,
                  "expressiveness": 60},
        "gemini_voice": "Leda",
    },
    "絵本の読み手": {
        "slug": "ehon",
        "description": "りあキッズTV用。柔らかくゆっくり、間は長め",
        "tone_preset": "柔らかい",
        "style_prompt": "",
        "stats": {"pitch": 50, "speed": 30, "gap": 75,
                  "expressiveness": 55},
        "gemini_voice": "Aoede",
    },
    "占い師": {
        "slug": "uranai",
        "description": "リア占い用。落ち着いて神秘的、間を活かす",
        "tone_preset": "カスタム",
        "style_prompt": "落ち着いて神秘的な声で、言葉の間を大切に"
                        "しながら、ゆったりと読み聞かせてください",
        "stats": {"pitch": 42, "speed": 32, "gap": 80,
                  "expressiveness": 45},
        "gemini_voice": "Callirrhoe",
    },
    "ニュース読み": {
        "slug": "news",
        "description": "速報・紹介記事の朗読用。明瞭で速め、歯切れよく",
        "tone_preset": "カスタム",
        "style_prompt": "ニュース原稿を読むアナウンサーのように、"
                        "明瞭で歯切れよく読んでください",
        "stats": {"pitch": 50, "speed": 68, "gap": 30,
                  "expressiveness": 30},
        "gemini_voice": "Leda",
    },
    "元気な妹系": {
        "slug": "imouto",
        "description": "汎用。高め・速め・感情豊かな妹キャラ",
        "tone_preset": "元気",
        "style_prompt": "",
        "stats": {"pitch": 68, "speed": 62, "gap": 35,
                  "expressiveness": 80},
        "gemini_voice": "Zephyr",
    },
    "落ち着いたお姉さん系": {
        "slug": "oneesan",
        "description": "汎用。穏やかで安定感のあるお姉さんキャラ",
        "tone_preset": "落ち着いた",
        "style_prompt": "",
        "stats": {"pitch": 55, "speed": 45, "gap": 50,
                  "expressiveness": 50},
        "gemini_voice": "Callirrhoe",
    },
    "クール系": {
        "slug": "cool",
        "description": "汎用。低め・抑えめで、さらっとしたキャラ",
        "tone_preset": "クール",
        "style_prompt": "",
        "stats": {"pitch": 38, "speed": 48, "gap": 50,
                  "expressiveness": 25},
        "gemini_voice": "Autonoe",
    },
    "甘い系": {
        "slug": "amai",
        "description": "汎用。甘く可愛らしく、親しみを込めたキャラ",
        "tone_preset": "甘い",
        "style_prompt": "",
        "stats": {"pitch": 66, "speed": 45, "gap": 50,
                  "expressiveness": 65},
        "gemini_voice": "Kore",
    },
}


def character_sheet(name, engine=None):
    """キャラプリセット名 → キャラシート（dict）。"""
    if name not in CHARACTER_PRESETS:
        raise AdjustError(
            f"キャラプリセットが不正です: {name!r}。"
            f"使えるのは {', '.join(CHARACTER_PRESETS)} です"
            "（presets サブコマンドで一覧が見られます）。", 2)
    c = CHARACTER_PRESETS[name]
    eng = engine or "gemini"
    if eng not in ("gemini", "local"):
        raise AdjustError(
            f"engine が不正です: {eng!r}。gemini か local を指定して"
            "ください。", 2)
    voice = c["gemini_voice"] if eng == "gemini" else LOCAL_VOICE_ID
    return {
        "name": name,
        "tone_preset": c["tone_preset"],
        "style_prompt": c.get("style_prompt", ""),
        "stats": dict(c["stats"]),
        "overrides": {"target_speed_cpm": None, "target_gap_sec": None,
                      "pitch_shift_st": None},
        "engine": eng,
        "voice": voice,
        "match_reference_pitch": False,
        "allow_pitch_down": False,
        "reference_f0_median_hz": None,
        "source": {"type": "character_preset", "character": name},
    }


# ---------------------------------------------------------------------------
# キャラシート
# ---------------------------------------------------------------------------

def load_sheet(path):
    if not os.path.isfile(path):
        raise AdjustError(
            f"キャラシートが見つかりません: {path}\n"
            "対処: make-sheet で作るか、パスを確認してください。", 2)
    try:
        with open(path, encoding="utf-8") as f:
            sheet = json.load(f)
    except json.JSONDecodeError as e:
        raise AdjustError(
            f"キャラシートのJSONが壊れています: {path}（{e}）\n"
            "対処: JSONの構文（カンマや括弧）を直してください。", 2)
    if not isinstance(sheet, dict):
        raise AdjustError("キャラシートはJSONオブジェクトで書いてください。", 2)
    return sheet


def resolve_sheet(sheet):
    """キャラシート → 実効パラメータ（変換の内訳つき）。"""
    tone_name = sheet.get("tone_preset") or "カスタム"
    if tone_name not in TONE_PRESETS:
        raise AdjustError(
            f"tone_preset が不正です: {tone_name!r}。"
            f"使えるのは {', '.join(TONE_PRESETS)} です"
            "（presets サブコマンドで一覧が見られます）。", 2)
    tone = TONE_PRESETS[tone_name]
    stats = sheet.get("stats") or {}
    pitch_stat = norm_stat(stats.get("pitch", 50), "pitch")
    speed_stat = norm_stat(stats.get("speed", 50), "speed")
    gap_stat = norm_stat(stats.get("gap", 50), "gap")
    expr_stat = norm_stat(stats.get("expressiveness", 50), "expressiveness")

    overrides = sheet.get("overrides") or {}

    speed_cpm = stat_speed_cpm(speed_stat) * tone["speed_bias"]
    if overrides.get("target_speed_cpm") is not None:
        speed_cpm = float(overrides["target_speed_cpm"])
    gap_sec = stat_gap_sec(gap_stat)
    if overrides.get("target_gap_sec") is not None:
        gap_sec = float(overrides["target_gap_sec"])
    pitch_st = stat_pitch_st(pitch_stat) + tone["pitch_bias_st"]
    if overrides.get("pitch_shift_st") is not None:
        pitch_st = float(overrides["pitch_shift_st"])

    style = tone["style"]
    if tone_name == "カスタム":
        style = sheet.get("style_prompt") or ""
    expr_text = expr_style_tier(expr_stat)
    full_style = "。".join(s for s in [style, expr_text] if s)

    engine = sheet.get("engine") or "gemini"
    if engine not in ("gemini", "local"):
        raise AdjustError(
            f"engine が不正です: {engine!r}。gemini か local を指定してください。", 2)
    voice = sheet.get("voice") or tone["voice"]

    return {
        "name": sheet.get("name") or "（無名のシート）",
        "tone_preset": tone_name,
        "stats": {"pitch": pitch_stat, "speed": speed_stat,
                  "gap": gap_stat, "expressiveness": expr_stat},
        "engine": engine,
        "voice": voice,
        "style_prompt": full_style,
        "target_speed_cpm": speed_cpm,
        "target_gap_sec": gap_sec,
        "pitch_shift_st": pitch_st,
        "match_reference_pitch": bool(sheet.get("match_reference_pitch")),
        # 安全策（2026-11-18）: お手本寄せの下方向シフトは、シートで
        # allow_pitch_down を明示的に真にした場合だけ有効。その場合も
        # pitch_down_max_st（既定1.0半音）までしか下げない。大きな
        # 下げは高い声を「野太く」不自然にする（デモ実証の教訓）。
        "allow_pitch_down": bool(sheet.get("allow_pitch_down")),
        "pitch_down_max_st": (float(sheet["pitch_down_max_st"])
                              if sheet.get("pitch_down_max_st")
                              is not None else 1.0),
        "reference_f0_median_hz": sheet.get("reference_f0_median_hz"),
        "prosody": sheet.get("prosody"),
        "source": sheet.get("source"),
    }


# ---------------------------------------------------------------------------
# プリセット／シートへのカスタム調整（CLI引数 → シート上書き）
# ---------------------------------------------------------------------------

STAT_FIELDS = ("pitch", "speed", "gap", "expressiveness")


def _parse_stat_arg(spec, current, field):
    """調整値の解釈: '+10'/'-10' は相対指定、'60' は絶対値（0〜100）。"""
    s = str(spec).strip()
    try:
        if s.startswith(("+", "-")):
            new = float(current) + float(s)
        else:
            new = float(s)
    except ValueError:
        raise AdjustError(
            f"--{field} の値が不正です: {spec!r}。'60' のような絶対値か、"
            "'+10'/'-10' のような相対指定で書いてください。", 2)
    if new < 0 or new > 100:
        clamped = max(0.0, min(100.0, new))
        print(f"警告: {field} の調整結果 {new:.0f} が範囲（0〜100）外の"
              f"ため {clamped:.0f} に丸めました。", file=sys.stderr)
        new = clamped
    return new


def apply_adjustments(sheet, args):
    """キャラプリセット／読み込んだシートに CLI の調整を上書きする。

    入口の調整（高さ・速さ・間・強弱の4ステータス）と、上級調整
    （overrides の数値直接指定）の両方を受ける。
    """
    sheet = copy.deepcopy(sheet)
    stats = sheet.setdefault("stats", {})
    for field in STAT_FIELDS:
        spec = getattr(args, field, None)
        if spec is None:
            continue
        current = norm_stat(stats.get(field, 50), field)
        stats[field] = _parse_stat_arg(spec, current, field)
    overrides = sheet.setdefault("overrides", {})
    if getattr(args, "set_speed_cpm", None) is not None:
        overrides["target_speed_cpm"] = args.set_speed_cpm
    if getattr(args, "set_gap_sec", None) is not None:
        overrides["target_gap_sec"] = args.set_gap_sec
    if getattr(args, "set_pitch_st", None) is not None:
        overrides["pitch_shift_st"] = args.set_pitch_st
    # 文単位ピッチ整形は opt-in（安全策 2026-11-18）。既定はオフで、
    # 明示指定があったときだけシート側の apply_contour を立てる。
    if getattr(args, "apply_contour", False):
        prosody = sheet.get("prosody")
        if not prosody:
            raise AdjustError(
                "--apply-contour を指定しましたが、このシートには"
                "文単位の型（prosody）がありません。\n"
                "対処: metrics.json から make-sheet で作ったシートを"
                "使ってください。", 2)
        prosody["apply_contour"] = True
    return sheet


def save_sheet(sheet, path, name=None):
    """調整後のシートを「マイキャラ」として別名保存する。"""
    if name:
        sheet["name"] = name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# テキスト前処理
# ---------------------------------------------------------------------------

def count_chars(text):
    return len(PUNCT_RE.sub("", text))


def split_sentences(text):
    """空白を正規化し、文末記号と改行で文分割する（記号は文末に保持）。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t　]+", " ", line).strip()
        if not line:
            continue
        parts = re.findall(r"[^。！？!?\n]+[。！？!?]?", line)
        for p in parts:
            p = p.strip()
            if p and count_chars(p) > 0:
                sentences.append(p)
    return sentences


# ---------------------------------------------------------------------------
# 音声生成エンジン
# ---------------------------------------------------------------------------

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def gemini_tts(sentence, eff, out_wav):
    """Gemini TTS で1文を生成し、24kHz WAV で保存する。"""
    from dynamic_credentials import add_surrogate_to_request
    text = sentence
    if eff["style_prompt"]:
        text = f"{eff['style_prompt']}：{sentence}"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": eff["voice"]}},
                "languageCode": "ja-JP",
            },
        },
    }
    url = f"{GEMINI_BASE}/models/{GEMINI_TTS_MODEL}:generateContent"
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(2):
        request = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"})
        add_surrogate_to_request(
            request, "custom.gemini-api",
            allowed_hosts=["generativelanguage.googleapis.com"])
        try:
            with urllib.request.build_opener(NoRedirect()).open(
                    request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code == 429:
                advice = ("対処: 無料枠の利用上限（クォータ）に達しています。"
                          "時間をおく（日次で回復）か、engine を local に"
                          "切り替えてください。声や文の問題ではありません。")
            elif e.code == 400:
                advice = ("対処: 声の名前（voice）がモデルの対応表にあるか、"
                          "文に記号だけの部分がないか確認してください。")
            else:
                advice = "対処: 時間をおいて再試行してください。"
            raise AdjustError(
                f"Gemini TTS がHTTP {e.code} を返しました: {body}\n{advice}", 3)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt == 1:
                raise AdjustError(
                    f"Gemini API へ接続できませんでした: {e}\n"
                    "対処: ネットワークを確認し、時間をおいて再試行して"
                    "ください（この環境はGoogleへの接続が遅いことがあります）。", 3)
    for cand in result.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                pcm = base64.b64decode(inline["data"])
                m = re.search(r"rate=(\d+)", inline.get("mimeType", ""))
                rate = int(m.group(1)) if m else 24000
                raw = out_wav + ".raw.wav"
                with wave.open(raw, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(pcm)
                normalize_wav(raw, out_wav)
                os.remove(raw)
                return
    raise AdjustError(
        "Gemini TTS の応答に音声データが含まれていませんでした。\n"
        "対処: 文が短すぎる/記号だけの場合は文を分け直し、再試行して"
        "ください。", 4)


def local_tts(sentence, eff, out_wav):
    """既存 tts CLI で1文を生成し、24kHz WAV へ変換する。"""
    tts = shutil.which("tts") or "/opt/hatch/bin/tts"
    if not os.path.exists(tts):
        raise AdjustError(
            "既存TTS（tts CLI）が見つかりません。\n"
            "対処: engine を gemini にするか、tts CLI の導入を確認して"
            "ください。", 4)
    mp3 = out_wav + ".mp3"
    cmd = [tts, "speak", "--text", sentence, "--language", "ja",
           "--output", mp3]
    if eff["voice"]:
        cmd += ["--voice", eff["voice"]]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not os.path.isfile(mp3):
        raise AdjustError(
            f"既存TTSでの生成に失敗しました: {(proc.stderr or proc.stdout)[-300:]}\n"
            "対処: voice のIDが正しいか確認してください"
            "（tts の既定voiceは男性声です。リア用はキャラシートの voice に"
            "指定してください）。", 4)
    normalize_wav(mp3, out_wav)
    os.remove(mp3)


# ---------------------------------------------------------------------------
# 音声処理・測定
# ---------------------------------------------------------------------------

def run_ffmpeg(args, what):
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error"] + args,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise AdjustError(
            f"ffmpeg での{what}に失敗しました: {proc.stderr[-300:]}\n"
            "対処: 入力音声が壊れていないか確認してください。", 4)


def normalize_wav(src, dst):
    run_ffmpeg(["-i", src, "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-c:a", "pcm_s16le", dst], "音声の正規化")


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def atempo_chain(factor):
    """atempo は1段の範囲が限られるため、必要なだけ連ねる。"""
    filters = []
    f = factor
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f /= 0.5
    filters.append(f"atempo={f:.4f}")
    return filters


def process_segment(src, dst, tempo_factor, pitch_st):
    filters = []
    if abs(pitch_st) > 0.01:
        ratio = 2 ** (pitch_st / 12.0)
        filters.append(f"rubberband=pitch={ratio:.5f}")
    if abs(tempo_factor - 1.0) > 0.005:
        filters += atempo_chain(tempo_factor)
    if not filters:
        shutil.copy2(src, dst)
        return
    run_ffmpeg(["-i", src, "-af", ",".join(filters),
                "-ar", str(SAMPLE_RATE), "-ac", "1", dst], "速度/ピッチ調整")


def _measure_segment_shape(wav_path):
    """1文セグメントのピッチ形状を測る（voice_profile の物差し）。

    戻り値: {"chunk_st": [冒頭/中盤/末尾 各1/3の中央値（文の中央値比・
    半音）], "head_st"/"tail_st": 冒頭0.3秒／末尾0.3秒の中央値
    （測定側の物差しそのもの）, "delta_st": tail_st - head_st,
    "span_st": p10〜p90の幅}。測れなければ None。
    """
    if vp is None:
        return None
    try:
        frames = vp.extract_f0_frames(wav_path)
    except Exception:
        return None
    times, f0s, dur = frames["times"], frames["f0s"], frames["duration"]
    if dur < 0.5 or len(f0s) < 30:
        return None
    med = statistics.median(f0s)
    sts = [vp.hz_to_st(f, med) for f in f0s]

    def med_in(t0, t1):
        vals = [s for t, s in zip(times, sts) if t0 <= t <= t1]
        return statistics.median(vals) if len(vals) >= 2 else None

    chunks = []
    for k in range(3):
        t0, t1 = dur * k / 3.0, dur * (k + 1) / 3.0
        if k < 2:
            vals = [s for t, s in zip(times, sts) if t0 <= t < t1]
        else:
            vals = [s for t, s in zip(times, sts) if t0 <= t <= t1]
        if len(vals) < 5:
            return None
        chunks.append(statistics.median(vals))
    head_st = med_in(0.0, vp.HEAD_TAIL_SEC)
    tail_st = med_in(dur - vp.HEAD_TAIL_SEC, dur)
    if head_st is None or tail_st is None:
        # 端の窓が測れない場合は1/3分割の端同士で代用する
        head_st, tail_st = chunks[0], chunks[2]
    return {"chunk_st": chunks, "head_st": head_st, "tail_st": tail_st,
            "delta_st": tail_st - head_st,
            "span_st": vp.pct(sts, 0.90) - vp.pct(sts, 0.10)}


def _rms_of(path):
    import array
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = array.array("h")
    samples.frombytes(raw)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _match_rms(src, dst):
    """整形後の音量を元セグメントに合わせる。

    rubberband は文末などの微小音量部をさらに弱めることがあり、
    そのままだと測定側の強度ゲートで文末フレームが落ちて
    「型が測れない」事態になるため、RMS比で吸収する（0.7〜1.5倍）。
    """
    import array
    s_rms, d_rms = _rms_of(src), _rms_of(dst)
    if s_rms <= 0 or d_rms <= 0:
        return
    gain = max(0.7, min(1.5, s_rms / d_rms))
    if abs(gain - 1.0) < 0.02:
        return
    with wave.open(dst, "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = array.array("h")
    samples.frombytes(raw)
    for i, s in enumerate(samples):
        v = int(s * gain)
        samples[i] = max(-32768, min(32767, v))
    with wave.open(dst, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())


def shape_segment_contour(src, dst, pitch_st, prosody):
    """文単位のピッチ整形。文を3分割し、各部のピッチをずらして
    お手本の「文ごとのピッチ変化（delta）」と「抑揚の幅（span）」の
    中央値に寄せる。全体シフト pitch_st も同時に適用する。

    delta の補正量は、測定側の物差し（冒頭0.3秒→末尾0.3秒の中央値差）
    で測った現在値から計算する（1/3分割の中央値で計算すると、端の
    窓で測る評価値と符号まで食い違うことを実測で確認したため）。
    補正の傾斜は ±6半音に制限する（TTSの文末急降下など、極端な
    素の輪郭を全量は打ち消さない）。さらに各部のシフトは ±2半音に
    制限する：文末のきしみ声は rubberband のシフト量が大きいと
    F0追跡ごと壊れて「型が測れない」状態になることを実測したため
    （+3半音で文末の有声フレームが 8→2 に減少）。届く範囲で寄せる。
    整形できたら True（測れない短文などは False。呼び出し側で
    通常の一律シフトに倒す）。
    """
    shape = _measure_segment_shape(src)
    if shape is None:
        return False
    target_delta = prosody.get("delta_median_st")
    target_span = prosody.get("span_median_st")
    scale = 1.0
    if (target_span is not None and shape["span_st"]
            and shape["span_st"] > 0.5):
        scale = max(0.6, min(1.6, target_span / shape["span_st"]))
    shifts = [c * (scale - 1.0) for c in shape["chunk_st"]]
    if target_delta is not None:
        scaled_delta = shape["delta_st"] * scale
        corr = max(-6.0, min(6.0, target_delta - scaled_delta))
        shifts[0] -= corr / 2.0
        shifts[2] += corr / 2.0
    shifts = [max(-2.0, min(2.0, s)) + pitch_st for s in shifts]
    with wave.open(src, "rb") as w:
        n = w.getnframes()
        data = w.readframes(n)
    bounds = [0, n // 3, 2 * n // 3, n]
    tmpdir = os.path.dirname(dst)
    parts = []
    try:
        for k in range(3):
            raw_p = os.path.join(tmpdir, f"_chunk{k}.raw.wav")
            out_p = os.path.join(tmpdir, f"_chunk{k}.wav")
            with wave.open(raw_p, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(data[bounds[k] * 2:bounds[k + 1] * 2])
            if abs(shifts[k]) > 0.01:
                ratio = 2 ** (shifts[k] / 12.0)
                run_ffmpeg(["-i", raw_p, "-af",
                            f"rubberband=pitch={ratio:.5f}",
                            "-ar", str(SAMPLE_RATE), "-ac", "1", out_p],
                           "文単位のピッチ整形")
            else:
                shutil.copy2(raw_p, out_p)
            parts.append(out_p)
            os.remove(raw_p)
        concat_wavs(parts, dst)
        _match_rms(src, dst)
    finally:
        for p in parts:
            if os.path.exists(p):
                os.remove(p)
    return True


def make_silence(sec, path):
    n = int(SAMPLE_RATE * sec)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * n)


def concat_wavs(paths, dst):
    with wave.open(dst, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        for p in paths:
            with wave.open(p, "rb") as w:
                if (w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1
                        or w.getsampwidth() != 2):
                    raise AdjustError(
                        f"内部形式が揃っていない音声があります: {p}", 4)
                out.writeframes(w.readframes(w.getnframes()))


def vad_frames(wav_path):
    """10ms RMS走査 → (有声フラグ列, フレーム長秒)。"""
    with wave.open(wav_path, "rb") as w:
        raw = w.readframes(w.getnframes())
    import array
    samples = array.array("h")
    samples.frombytes(raw)
    frame = SAMPLE_RATE // 100  # 10ms
    n_frames = len(samples) // frame
    flags = []
    rms = []
    for i in range(n_frames):
        chunk = samples[i * frame:(i + 1) * frame]
        if not chunk:
            break
        rms.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)))
    if rms:
        threshold = max(max(rms) * 0.04, 60.0)
        flags = [r > threshold for r in rms]
    return flags, 0.01


def speech_seconds(wav_path):
    """VAD基準の発話時間（無音を除く秒数）。"""
    flags, flen = vad_frames(wav_path)
    return sum(flags) * flen


def trim_silence(src, dst, pad_sec=0.03):
    """前後の無音を切り落とす（発話の前後に pad_sec だけ残す）。"""
    flags, flen = vad_frames(src)
    voiced_idx = [i for i, v in enumerate(flags) if v]
    if not voiced_idx:
        shutil.copy2(src, dst)
        return
    frame = SAMPLE_RATE // 100
    pad = int(pad_sec / flen)
    start = max(0, (voiced_idx[0] - pad) * frame)
    end = min(len(flags) * frame, (voiced_idx[-1] + 1 + pad) * frame)
    with wave.open(src, "rb") as w:
        w.setpos(start)
        data = w.readframes(end - start)
    with wave.open(dst, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data)


def measure_boundary_gaps(wav_path, centers_sec):
    """組み立て位置（挿入した間の中心）ごとに、実際の無音の長さを測る。

    文中の小さいポーズと混ざる「内部の無音ラン中央値」方式は、テンポを
    遅くすると文中ポーズまで数えてしまうため、文間の間の測定には
    こちらを使う。
    """
    flags, flen = vad_frames(wav_path)
    values = []
    for c in centers_sec:
        i = int(c / flen)
        if i < 0 or i >= len(flags):
            continue
        if flags[i]:
            # 中心が有声だった場合は前後0.3秒以内で最も近い無音ランを探す
            found = None
            for d in range(1, int(0.3 / flen)):
                for j in (i - d, i + d):
                    if 0 <= j < len(flags) and not flags[j]:
                        found = j
                        break
                if found is not None:
                    break
            if found is None:
                continue
            i = found
        lo = i
        while lo > 0 and not flags[lo - 1]:
            lo -= 1
        hi = i
        while hi < len(flags) - 1 and not flags[hi + 1]:
            hi += 1
        values.append((hi - lo + 1) * flen)
    return values


def measure_f0_median(wav_path):
    """parselmouth で有声フレームのF0中央値（Hz）。測れなければ None。"""
    try:
        import parselmouth
    except ImportError:
        return None
    try:
        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75,
                             pitch_ceiling=600)
        freqs = [float(f) for f in pitch.selected_array["frequency"]
                 if f > 0]
        if len(freqs) < 30:
            return None
        return statistics.median(freqs)
    except Exception:
        return None


def measure_vad(wav_path):
    """10ms RMS走査で 発話時間・内部の間の中央値 を測る。"""
    voiced, flen = vad_frames(wav_path)
    if not voiced:
        raise AdjustError("完成音声の測定ができませんでした（無音）。", 5)
    speech_sec = sum(voiced) * flen
    duration = wav_duration(wav_path)
    # 内部の無音ラン（先頭・末尾を除く）
    gaps = []
    run = 0
    started = False
    for v in voiced:
        if v:
            if started and run > 0:
                gaps.append(run * 0.01)
            started = True
            run = 0
        elif started:
            run += 1
    internal = [g for g in gaps if g >= 0.08]
    gap_median = statistics.median(internal) if internal else None
    return {"speech_sec": speech_sec, "gap_median": gap_median,
            "n_gaps": len(internal), "duration": duration}


# ---------------------------------------------------------------------------
# サブコマンド: presets
# ---------------------------------------------------------------------------

def cmd_presets(args):
    print("■ トーンプリセット")
    for name, t in TONE_PRESETS.items():
        print(f"- {name}: 推奨voice={t['voice']}"
              f" / ピッチ寄せ={t['pitch_bias_st']:+.1f}半音"
              f" / 速さ寄せ=×{t['speed_bias']:.2f}")
        print(f"  スタイル: {t['style'] or '（style_prompt を自分で書く）'}")
    print("\n■ ステータス（0〜100）")
    print("- pitch（高さ）: 50が基準。0で−6半音、100で+6半音（(値-50)/50×6）")
    print("- speed（速さ）: 目標話速 = 180 + 値×3 chars/min（180〜480）")
    print("- gap（間）: 文間の無音 = 0.2 + 値/100×1.3 秒（0.2〜1.5）")
    print("- expressiveness（強弱）: 0-33=控えめ / 34-66=自然 / 67-100=感情豊か"
          "（弱/中/強 の指定も可。スタイルプロンプトへ反映）")
    print("\n■ キャラプリセット（選ぶだけで使える完成キャラ）")
    for name, c in CHARACTER_PRESETS.items():
        eff = resolve_sheet(character_sheet(name))
        st = c["stats"]
        print(f"- {name}: {c['description']}")
        print(f"  トーン={c['tone_preset']}"
              f" / 高さ{st['pitch']}・速さ{st['speed']}・間{st['gap']}"
              f"・強弱{st['expressiveness']}"
              f" → 目標 話速{eff['target_speed_cpm']:.0f}cpm"
              f"・間{eff['target_gap_sec']:.2f}秒"
              f"・ピッチ{eff['pitch_shift_st']:+.1f}半音"
              f" / 推奨voice={c['gemini_voice']}（gemini時）")
    print("  ※local エンジンでは全キャラ共通の声（リア用 voice ID）に、"
          "ピッチ・速さ・間の後処理でキャラ差をつけます。")
    print("  ※キャラは土台です。synthesize の --speed +10 などで"
          "少し調整し、--save-sheet でマイキャラとして保存できます。")
    return 0


# ---------------------------------------------------------------------------
# サブコマンド: make-sheet
# ---------------------------------------------------------------------------

def cmd_make_sheet(args):
    if not os.path.isfile(args.from_metrics):
        raise AdjustError(
            f"metrics.json が見つかりません: {args.from_metrics}\n"
            "対処: video_analyzer.py の出力フォルダにある metrics.json の"
            "パスを指定してください。", 2)
    try:
        with open(args.from_metrics, encoding="utf-8") as f:
            metrics = json.load(f)
    except json.JSONDecodeError as e:
        raise AdjustError(f"metrics.json のJSONが壊れています: {e}", 2)
    cap = metrics.get("captions") or {}
    speed = cap.get("speed_speech") or cap.get("speed_overall")
    gap = cap.get("gap_median")
    if speed is None or gap is None:
        raise AdjustError(
            "metrics.json に話速（speed_speech）または間（gap_median）が"
            "ありません。\n対処: video_analyzer.py の完全な分析結果を"
            "使ってください。", 2)
    f0 = metrics.get("f0") or {}
    sheet = {
        "name": args.name or f"お手本: {metrics.get('title') or metrics.get('video_id')}",
        "tone_preset": args.tone,
        "style_prompt": "",
        "stats": {
            "pitch": 50,
            "speed": cpm_to_stat(float(speed)),
            "gap": gap_to_stat(float(gap)),
            "expressiveness": 50,
        },
        "overrides": {
            "target_speed_cpm": round(float(speed), 1),
            "target_gap_sec": round(float(gap), 2),
            "pitch_shift_st": None,
        },
        "engine": args.engine,
        "voice": args.voice or "",
        "match_reference_pitch": False,
        "allow_pitch_down": False,
        "reference_f0_median_hz": (round(float(f0["median_hz"]), 1)
                                   if f0.get("median_hz") else None),
        "source": {"type": "metrics",
                   "video_id": metrics.get("video_id"),
                   "title": metrics.get("title")},
    }
    # 文単位の型（文末パターン比率・文ごとのピッチ変化と抑揚の幅）も
    # シートに持たせる。ただし整形（prosody.apply_contour）は安全策で
    # 既定オフ——型データは記録するだけに留め、有効化は opt-in
    # （シートで明示 or synthesize の --apply-contour）。
    if vp is not None:
        try:
            sheet["prosody"] = vp.prosody_for_sheet(
                vp.profile_from_metrics(metrics))
        except Exception:
            sheet["prosody"] = None
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)
    print(f"OK: キャラシートを作成しました -> {args.out}")
    print(f"  話速（お手本）: {speed:.1f} chars/min / 間: {gap:.2f}秒"
          f" / お手本F0中央値: {sheet['reference_f0_median_hz']} Hz")
    if sheet.get("prosody"):
        pr = sheet["prosody"]
        print(f"  文単位の型: ピッチ変化の中央値 {pr['delta_median_st']}半音"
              f" / 抑揚の幅の中央値 {pr['span_median_st']}半音"
              f" / 文単位整形 {'あり' if pr['apply_contour'] else 'なし'}")
    print("  ※話速と間は overrides に数値で入っています"
          "（ステータスは逆換算した表示値）。")
    return 0


# ---------------------------------------------------------------------------
# サブコマンド: synthesize
# ---------------------------------------------------------------------------

def cmd_synthesize(args):
    # 入口: キャラプリセット（おすすめ）か、キャラシート直指定のどちらか
    if getattr(args, "preset", None):
        sheet = character_sheet(args.preset,
                                engine=getattr(args, "engine", None))
    elif getattr(args, "profile", None):
        sheet = load_sheet(args.profile)
        if getattr(args, "engine", None):
            sheet["engine"] = args.engine
    else:
        raise AdjustError(
            "音声の元が指定されていません。\n"
            "対処: --preset キャラ名（おすすめ）か、--profile "
            "キャラシート.json を指定してください。", 2)
    # 調整: 入口の4ステータス＋上級調整（overrides）を上書き
    sheet = apply_adjustments(sheet, args)
    if getattr(args, "save_sheet", None):
        save_name = getattr(args, "sheet_name", None)
        if not save_name and getattr(args, "preset", None):
            save_name = f"{args.preset}・カスタム"
        save_sheet(sheet, args.save_sheet, name=save_name)
        print(f"OK: 調整後のシートをマイキャラとして保存しました"
              f" -> {args.save_sheet}")
    eff = resolve_sheet(sheet)

    if args.text_file:
        if not os.path.isfile(args.text_file):
            raise AdjustError(
                f"テキストファイルが見つかりません: {args.text_file}", 2)
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text or ""
    sentences = split_sentences(text)
    if not sentences:
        raise AdjustError(
            "読み上げる文がありません。\n"
            "対処: --text か --text-file で本文を指定してください"
            "（記号だけの文は数えません）。", 2)
    total_chars = sum(count_chars(s) for s in sentences)

    os.makedirs(args.out_dir, exist_ok=True)
    if getattr(args, "preset", None):
        # プリセット使用時は、調整後のシート控えを出力先にも残す
        with open(os.path.join(args.out_dir, "sheet-used.json"), "w",
                  encoding="utf-8") as f:
            json.dump(sheet, f, ensure_ascii=False, indent=2)
    workdir = tempfile.mkdtemp(prefix="voice_adjust_")
    try:
        # 1) 文ごとに生成 → 前後の無音をトリム（間の実測を挿入値に一致
        #    させるため）。話速は生成側・完成側ともVAD基準で測る。
        raw_segs = []
        gen = gemini_tts if eff["engine"] == "gemini" else local_tts
        for i, s in enumerate(sentences):
            seg = os.path.join(workdir, f"seg{i:03d}_gen.wav")
            gen(s, eff, seg)
            trimmed = os.path.join(workdir, f"seg{i:03d}_raw.wav")
            trim_silence(seg, trimmed)
            raw_segs.append(trimmed)
        raw_speech_sec = sum(speech_seconds(p) for p in raw_segs)
        raw_cpm = (total_chars / (raw_speech_sec / 60.0)
                   if raw_speech_sec > 0 else 0.0)

        # 2) 生成直後のF0を測る（ピッチ寄せの材料＋レポート用）
        pitch_st = eff["pitch_shift_st"]
        match_delta = 0.0
        match_note = ""
        speech_only = os.path.join(workdir, "speech_only.wav")
        concat_wavs(raw_segs, speech_only)
        base_f0 = measure_f0_median(speech_only)
        if eff["match_reference_pitch"]:
            ref = eff["reference_f0_median_hz"]
            if ref and base_f0:
                raw_delta = 12.0 * math.log2(float(ref) / base_f0)
                raw_delta = max(-6.0, min(6.0, raw_delta))
                if raw_delta < 0:
                    # 安全策（2026-11-18）: 下方向の寄せは既定で無効。
                    # 高い声を大きく下げると「野太く」不自然になる
                    # （デモ実証でボブ評価「怖い・野太い」）。シートで
                    # allow_pitch_down を明示した場合のみ、上限つきで
                    # 適用する。
                    if eff["allow_pitch_down"]:
                        match_delta = max(raw_delta,
                                          -abs(eff["pitch_down_max_st"]))
                        if match_delta != raw_delta:
                            match_note = (
                                f"下方向の寄せ {raw_delta:+.2f}半音は上限 "
                                f"{eff['pitch_down_max_st']:.1f}半音で制限")
                    else:
                        match_note = (
                            f"下方向の寄せ {raw_delta:+.2f}半音は安全策で"
                            "無効（シートの allow_pitch_down が未指定）")
                        print(f"注意: {match_note}。", file=sys.stderr)
                else:
                    match_delta = raw_delta
                pitch_st += match_delta
            else:
                print("警告: お手本F0へ寄せる指定ですが、F0を測れなかった"
                      "ためピッチ寄せをスキップしました。", file=sys.stderr)

        # 3) テンポ係数
        tempo = 1.0
        if eff["target_speed_cpm"] and raw_cpm > 0:
            # atempo の係数は「再生速度の倍率」。速すぎる生成を目標まで
            # 落とすには 目標/素 の係数で遅くする。
            tempo = eff["target_speed_cpm"] / raw_cpm
            if tempo > 2.0 or tempo < 0.5:
                print(f"警告: テンポ係数 {tempo:.2f} が調整範囲（0.5〜2.0）を"
                      "超えたため丸めました。目標話速が生成エンジンの素の"
                      "速さと大きく離れています。", file=sys.stderr)
                tempo = max(0.5, min(2.0, tempo))

        # 4) 後処理＋間挿入で組み立て
        #    シートに文単位の型（prosody）があり apply_contour が真なら、
        #    文ごとにピッチ形状を整形してからテンポを合わせる。
        gap_wav = os.path.join(workdir, "gap.wav")
        make_silence(eff["target_gap_sec"], gap_wav)
        pieces = []
        adj_segs = []
        prosody = eff.get("prosody") or {}
        use_contour = bool(prosody.get("apply_contour")) and vp is not None
        shaped_n = 0
        for i, raw in enumerate(raw_segs):
            adj = os.path.join(workdir, f"seg{i:03d}_adj.wav")
            shaped = os.path.join(workdir, f"seg{i:03d}_shaped.wav")
            if use_contour and shape_segment_contour(
                    raw, shaped, pitch_st, prosody):
                process_segment(shaped, adj, tempo, 0.0)
                shaped_n += 1
            else:
                process_segment(raw, adj, tempo, pitch_st)
            adj_segs.append(adj)
            pieces.append(adj)
            if i < len(raw_segs) - 1:
                pieces.append(gap_wav)
        out_wav = os.path.join(args.out_dir, f"{args.out_name}.wav")
        concat_wavs(pieces, out_wav)
        out_mp3 = os.path.join(args.out_dir, f"{args.out_name}.mp3")
        run_ffmpeg(["-i", out_wav, "-codec:a", "libmp3lame", "-q:a", "3",
                    out_mp3], "MP3変換")

        # 5) 完成品の測定（間は組み立て位置で実測する）
        vad = measure_vad(out_wav)
        centers = []
        timings = []
        t = 0.0
        for i, adj in enumerate(adj_segs):
            d = wav_duration(adj)
            timings.append({"start": t, "est_end": t + d})
            t += d
            if i < len(adj_segs) - 1:
                centers.append(t + eff["target_gap_sec"] / 2.0)
                t += eff["target_gap_sec"]
        gap_values = measure_boundary_gaps(out_wav, centers)
        vad["gap_median"] = (statistics.median(gap_values)
                             if gap_values else None)
        vad["n_gaps"] = len(gap_values)
        meas_cpm = (total_chars / (vad["speech_sec"] / 60.0)
                    if vad["speech_sec"] > 0 else 0.0)
        final_f0 = measure_f0_median(out_wav)

        # 5b) 文単位の型の実測（お手本との比較用）
        meas_prosody = None
        if vp is not None:
            try:
                frames = vp.extract_f0_frames(out_wav)
                feats = vp.sentence_features(frames, timings)
                prof = vp.aggregate_profile(feats)
                meas_prosody = {
                    "delta_median_st": prof["delta_median_st"],
                    "span_median_st": prof["span_median_st"],
                    "final_ratios": prof["final_ratios"],
                    "shaped_sentences": shaped_n,
                }
            except Exception:
                meas_prosody = None

        # 6) レポート
        report = build_report(eff, sentences, total_chars, raw_cpm,
                              tempo, pitch_st, match_delta, base_f0,
                              vad, meas_cpm, final_f0,
                              out_wav, out_mp3,
                              meas_prosody=meas_prosody,
                              match_note=match_note)
        report_path = os.path.join(args.out_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        with open(os.path.join(args.out_dir, "profile-used.json"), "w",
                  encoding="utf-8") as f:
            json.dump(eff, f, ensure_ascii=False, indent=2)
        print(f"OK: 音声を生成しました -> {out_wav}")
        print(f"    MP3: {out_mp3}")
        print(f"    レポート: {report_path}")
        print(f"    実測: 話速 {meas_cpm:.0f} chars/min（目標 "
              f"{eff['target_speed_cpm']:.0f}）/ 間中央値 "
              f"{vad['gap_median']}秒（目標 {eff['target_gap_sec']:.2f}）")
        return 0
    finally:
        if args.keep_work:
            kept = os.path.join(args.out_dir, "work")
            os.makedirs(kept, exist_ok=True)
            for name in os.listdir(workdir):
                shutil.copy2(os.path.join(workdir, name),
                             os.path.join(kept, name))
        shutil.rmtree(workdir, ignore_errors=True)


def fmt_or_dash(x, digits=1):
    return f"{x:.{digits}f}" if x is not None else "—"


def build_report(eff, sentences, total_chars, raw_cpm, tempo, pitch_st,
                 match_delta, base_f0, vad, meas_cpm, final_f0,
                 out_wav, out_mp3, meas_prosody=None, match_note=""):
    st = eff["stats"]
    lines = []
    lines.append(f"# 音声調整レポート — {eff['name']}")
    lines.append("")
    lines.append("## キャラシート（適用内容）")
    lines.append("")
    lines.append(f"- トーンプリセット: {eff['tone_preset']}")
    lines.append(f"- ステータス: 高さ {st['pitch']:.0f} / 速さ {st['speed']:.0f}"
                 f" / 間 {st['gap']:.0f} / 強弱 {st['expressiveness']:.0f}")
    lines.append(f"- エンジン: {eff['engine']} / 声: {eff['voice']}")
    lines.append(f"- スタイルプロンプト: {eff['style_prompt'] or '（なし）'}")
    if eff["source"]:
        lines.append(f"- お手本: {eff['source'].get('title') or ''}"
                     f"（{eff['source'].get('video_id') or ''}）")
    lines.append("")
    lines.append("## 目標 vs 実測")
    lines.append("")
    lines.append("| 項目 | 目標 | 実測 | 備考 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 話速（発話内 chars/min） | {eff['target_speed_cpm']:.0f} "
                 f"| {meas_cpm:.0f} | 素の生成 {raw_cpm:.0f} に atempo "
                 f"×{tempo:.3f} |")
    gap_m = vad["gap_median"]
    lines.append(f"| 文間の間（秒・中央値） | {eff['target_gap_sec']:.2f} "
                 f"| {fmt_or_dash(gap_m, 2)} | 組み立て位置 "
                 f"{vad['n_gaps']}箇所での実測 |")
    shift_note = f"指定 {eff['pitch_shift_st']:+.2f}半音"
    if match_delta:
        shift_note += f" ＋ お手本寄せ {match_delta:+.2f}半音"
    pitch_remark = "rubberband で適用"
    if match_note:
        pitch_remark += f"／{match_note}"
    lines.append(f"| ピッチ | {shift_note} | F0中央値 "
                 f"{fmt_or_dash(final_f0)} Hz（生成直後 "
                 f"{fmt_or_dash(base_f0)} Hz） | {pitch_remark} |")
    lines.append(f"| 強弱（抑揚） | ステータス {st['expressiveness']:.0f} "
                 "| 数値担保なし | スタイルプロンプトへの反映のみ |")
    prosody = eff.get("prosody") or {}
    if prosody.get("apply_contour"):
        mp = meas_prosody or {}
        td = prosody.get("delta_median_st")
        ts = prosody.get("span_median_st")
        lines.append(
            f"| 文のピッチ変化・中央値（半音） | {fmt_or_dash(td)} "
            f"| {fmt_or_dash(mp.get('delta_median_st'))} "
            f"| 文単位の型。整形した文 {mp.get('shaped_sentences', 0)}"
            f"/{len(sentences)} |")
        lines.append(
            f"| 文内の抑揚の幅・中央値（半音） | {fmt_or_dash(ts)} "
            f"| {fmt_or_dash(mp.get('span_median_st'))} "
            "| 文単位の型（3分割整形の近似） |")
        tr = prosody.get("final_ratios") or {}
        mr = mp.get("final_ratios") or {}
        if tr and mr:
            lines.append(
                "| 文末パターン比率（上/下/横） | "
                f"{tr.get('rise', 0) * 100:.0f}/{tr.get('fall', 0) * 100:.0f}"
                f"/{tr.get('flat', 0) * 100:.0f}％ "
                f"| {mr.get('rise', 0) * 100:.0f}/{mr.get('fall', 0) * 100:.0f}"
                f"/{mr.get('flat', 0) * 100:.0f}％ "
                "| 文単位の型 |")
    lines.append("")
    lines.append("## 内訳")
    lines.append("")
    lines.append(f"- 文の数: {len(sentences)} / 総文字数（句読点除く）: "
                 f"{total_chars}")
    lines.append(f"- 完成音声の長さ: {vad['duration']:.2f}秒"
                 f"（うち発話 {vad['speech_sec']:.2f}秒）")
    if eff["reference_f0_median_hz"]:
        lines.append(f"- お手本F0中央値: {eff['reference_f0_median_hz']} Hz")
    lines.append("")
    lines.append("## 文の一覧")
    lines.append("")
    for i, s in enumerate(sentences, 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("## 出力ファイル")
    lines.append("")
    lines.append(f"- WAV: {out_wav}")
    lines.append(f"- MP3: {out_mp3}")
    lines.append("")
    lines.append("※強弱（抑揚）とトーンはスタイルプロンプト経由のため、"
                 "数値での再現保証はありません。数値で担保しているのは"
                 "話速・間・ピッチ中央値と、文単位の型（ピッチ変化と"
                 "抑揚の幅の中央値。ただし文を3分割して整形する近似で、"
                 "文内の細かい抑揚カーブそのものの移植ではありません）。")
    lines.append("※安全策（2026-11-18）: ピッチのお手本寄せは下方向が"
                 "既定で無効（シートの allow_pitch_down を明示した"
                 "場合のみ上限つきで適用）。文単位のピッチ整形は"
                 "既定オフ（--apply-contour で opt-in）。類似度や"
                 "数値の一致は自然さの保証ではないため、提出前は"
                 "必ず試聴チェックを通すこと。")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="キャラクリ方式の音声調整ツール（生成側）")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("presets",
                   help="プリセット（トーン／キャラ）とステータスの対応表を表示")

    ms = sub.add_parser("make-sheet",
                        help="metrics.json からキャラシートを作成")
    ms.add_argument("--from-metrics", required=True,
                    help="video_analyzer.py の metrics.json")
    ms.add_argument("--out", required=True, help="出力するキャラシートJSON")
    ms.add_argument("--name", help="シート名")
    ms.add_argument("--tone", default="カスタム",
                    choices=list(TONE_PRESETS), help="トーンプリセット")
    ms.add_argument("--engine", default="gemini",
                    choices=["gemini", "local"])
    ms.add_argument("--voice", help="声（空ならプリセット推奨）")

    sy = sub.add_parser("synthesize",
                        help="キャラを選ぶ→少し調整→音声が出る")
    sy.add_argument("--text", help="読み上げるテキスト")
    sy.add_argument("--text-file", help="テキストファイル")
    sy.add_argument("--preset", choices=list(CHARACTER_PRESETS),
                    help="キャラプリセット名（おすすめの入口）")
    sy.add_argument("--profile", help="キャラシートJSON（--preset の代わり）")
    sy.add_argument("--engine", choices=["gemini", "local"],
                    help="エンジンの上書き")
    # 入口の調整（4つだけ）: 絶対値 '60' または相対指定 '+10'/'-10'
    sy.add_argument("--pitch", help="高さの調整（0〜100、または +10 の相対指定）")
    sy.add_argument("--speed", help="速さの調整（同上）")
    sy.add_argument("--gap", help="間の調整（同上）")
    sy.add_argument("--expr", dest="expressiveness",
                    help="強弱の調整（同上）")
    # 上級調整: 「欲しかった声」まで追い込むための数値直接指定
    sy.add_argument("--set-speed-cpm", type=float,
                    help="上級調整: 目標話速（chars/min）を数値で直接指定")
    sy.add_argument("--set-gap-sec", type=float,
                    help="上級調整: 文間の間（秒）を数値で直接指定")
    sy.add_argument("--set-pitch-st", type=float,
                    help="上級調整: ピッチシフト（半音）を数値で直接指定")
    sy.add_argument("--apply-contour", action="store_true",
                    help="opt-in: 文単位のピッチ整形を有効化する"
                         "（既定オフ。local土台では不自然になり得るため"
                         "試聴確認の上で使うこと）")
    sy.add_argument("--save-sheet",
                    help="調整後のシートをマイキャラとして保存するJSONパス")
    sy.add_argument("--sheet-name", help="--save-sheet で保存するシートの名前")
    sy.add_argument("--out-dir", required=True, help="出力フォルダ")
    sy.add_argument("--out-name", default="voice", help="出力ファイル名")
    sy.add_argument("--keep-work", action="store_true",
                    help="中間ファイルを out-dir/work に残す")

    args = parser.parse_args()
    try:
        if args.command == "presets":
            return cmd_presets(args)
        if args.command == "make-sheet":
            return cmd_make_sheet(args)
        if args.command == "synthesize":
            if not args.text and not args.text_file:
                parser.error("synthesize には --text か --text-file が必要です")
            return cmd_synthesize(args)
    except AdjustError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
