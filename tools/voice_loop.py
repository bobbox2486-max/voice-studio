#!/usr/bin/env python3
"""voice_loop.py — 「抽出 → 近付け → 測る → 直す」を回す自動ループ。

お手本（video_analyzer.py の metrics.json）から型プロファイルを作り、
キャラシート化 → 生成（voice_adjust.py）→ 生成音声の型を測定
（voice_profile.py）→ お手本との類似度スコア算出 → ズレの大きい
項目をシート側で補正して再生成、を1コマンドで反復します。

狙いは「反復で類似度が上がること」の実証と、補正の効きの記録です。
各ラウンドの音声・レポートは out-dir/round<N>/ に残り、全体の
経過は out-dir/loop-report.md と loop-result.json にまとまります。

使い方:
  python3 voice_loop.py --metrics test-output/gLv9_04OUyI/metrics.json \
      --text-file script.txt --engine local --rounds 3 \
      --out-dir test-output/voice-loop/bench

終了コード: 0=成功 / 2=入力エラー / 3=生成・測定の失敗
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import voice_adjust as va
import voice_profile as vp


class LoopError(Exception):
    def __init__(self, message, code=3):
        super().__init__(message)
        self.code = code


def _ns(**kw):
    """voice_adjust のサブコマンド関数へ渡す引数オブジェクト。"""
    base = dict(preset=None, profile=None, engine=None, text=None,
                text_file=None, pitch=None, speed=None, gap=None,
                expressiveness=None, set_speed_cpm=None, set_gap_sec=None,
                set_pitch_st=None, save_sheet=None, sheet_name=None,
                out_dir=None, out_name="voice", keep_work=False,
                from_metrics=None, out=None, name=None, tone="カスタム",
                voice=None)
    base.update(kw)
    return argparse.Namespace(**base)


def build_sheet(metrics_path, engine, voice, out_path):
    """metrics.json → キャラシート（voice_adjust.make-sheet を再利用）。"""
    rc = va.cmd_make_sheet(_ns(from_metrics=metrics_path, out=out_path,
                               engine=engine, voice=voice))
    if rc != 0:
        raise LoopError("キャラシートの作成に失敗しました", 3)
    return va.load_sheet(out_path)


def synthesize(sheet_path, text_file, out_dir, out_name="voice"):
    rc = va.cmd_synthesize(_ns(profile=sheet_path, text_file=text_file,
                               out_dir=out_dir, out_name=out_name))
    if rc != 0:
        raise LoopError(f"音声生成に失敗しました（{out_dir}）", 3)
    return os.path.join(out_dir, f"{out_name}.wav")


def adjust_sheet(sheet, eff, gen_profile, ref_profile):
    """測定結果とお手本のズレから、次ラウンドのシート補正を作る。

    補正の基準は「生成の実測がお手本に合うこと」であって、シートの
    目標値そのものではない（測定系の系統誤差 +0.09秒などを、目標を
    ずらして吸収するため。実測がお手本に一致しているのにシート目標
    との差を見て補正すると過補正になる——実測で確認した教訓）。
    補正できるのは話速・間・ピッチ中央値。文単位の型（delta/span）は
    整形側が直接目標に寄せるため、ここでは補正しない。
    戻り値: (新シート, 補正内容の文字列リスト)
    """
    import copy
    sheet = copy.deepcopy(sheet)
    overrides = sheet.setdefault("overrides", {})
    notes = []
    gen_speed = gen_profile.get("speed_speech")
    ref_speed = ref_profile.get("speed_speech")
    target_speed = eff["target_speed_cpm"]
    if gen_speed and ref_speed and target_speed:
        factor = ref_speed / gen_speed
        if abs(factor - 1.0) > 0.03:
            factor = max(0.75, min(1.33, factor))
            new_target = round(target_speed * factor, 1)
            overrides["target_speed_cpm"] = new_target
            notes.append(
                f"話速の目標を {target_speed:.0f} → {new_target:.0f} cpm"
                f"（実測 {gen_speed:.0f} vs お手本 {ref_speed:.0f}）")
    gen_gap = gen_profile.get("gap_median")
    ref_gap = ref_profile.get("gap_median")
    target_gap = eff["target_gap_sec"]
    if gen_gap is not None and ref_gap is not None and target_gap:
        err = gen_gap - ref_gap
        if abs(err) > 0.03:
            new_gap = round(target_gap - err, 2)
            new_gap = max(0.15, min(2.0, new_gap))
            if abs(new_gap - target_gap) > 0.01:
                overrides["target_gap_sec"] = new_gap
                notes.append(
                    f"間の目標を {target_gap:.2f} → {new_gap:.2f} 秒"
                    f"（実測 {gen_gap:.2f} vs お手本 {ref_gap:.2f}）")
    gen_f0 = gen_profile.get("f0_median_hz")
    ref_f0 = sheet.get("reference_f0_median_hz")
    if gen_f0 and ref_f0 and not sheet.get("match_reference_pitch"):
        diff_st = abs(vp.hz_to_st(gen_f0, float(ref_f0)))
        if diff_st > 2.0:
            sheet["match_reference_pitch"] = True
            notes.append(
                f"ピッチ中央値のお手本寄せを有効化"
                f"（実測との差 {diff_st:.1f} 半音）")
    return sheet, notes


def fmt(x, digits=2):
    return f"{x:.{digits}f}" if x is not None else "—"


def run_loop(args):
    if not os.path.isfile(args.metrics):
        raise LoopError(f"metrics.json が見つかりません: {args.metrics}", 2)
    if not os.path.isfile(args.text_file):
        raise LoopError(
            f"テキストファイルが見つかりません: {args.text_file}", 2)
    with open(args.metrics, encoding="utf-8") as f:
        metrics = json.load(f)
    with open(args.text_file, encoding="utf-8") as f:
        text = f.read()
    script = va.split_sentences(text)
    if not script:
        raise LoopError("台本から文を取り出せませんでした", 2)

    ref_profile = vp.profile_from_metrics(metrics)
    os.makedirs(args.out_dir, exist_ok=True)

    voice = args.voice
    if args.engine == "local" and not voice:
        voice = va.LOCAL_VOICE_ID
    sheet_path = os.path.join(args.out_dir, "sheet-round0.json")
    sheet = build_sheet(args.metrics, args.engine, voice, sheet_path)

    history = []
    for r in range(1, args.rounds + 1):
        round_dir = os.path.join(args.out_dir, f"round{r}")
        os.makedirs(round_dir, exist_ok=True)
        eff = va.resolve_sheet(sheet)
        wav = synthesize(sheet_path, args.text_file, round_dir)
        gen_profile = vp.profile_from_generated(wav, script)
        sim = vp.similarity(ref_profile, gen_profile)
        entry = {"round": r, "wav": wav, "score": sim["score"],
                 "dimensions": sim["dimensions"],
                 "missing": sim["missing"],
                 "gen_profile": gen_profile, "adjustments": []}
        history.append(entry)
        print(f"[round {r}] 類似度スコア: "
              f"{sim['score']:.1f} / 100" if sim["score"] is not None
              else f"[round {r}] スコア算出不能（測定不足）")
        for key, d in sim["dimensions"].items():
            print(f"    {d['label']}: お手本 {d['ref']} / 生成 {d['gen']}"
                  f" → {d['score']:.0f}点")
        if sim["score"] is not None and sim["score"] >= args.target_score:
            print(f"目標スコア {args.target_score} に到達しました。")
            entry["adjustments"] = ["（目標到達のため終了）"]
            break
        if r < args.rounds:
            sheet, notes = adjust_sheet(sheet, eff, gen_profile,
                                        ref_profile)
            sheet_path = os.path.join(args.out_dir, f"sheet-round{r}.json")
            va.save_sheet(sheet, sheet_path)
            entry["adjustments"] = notes
            for n in notes:
                print(f"    補正: {n}")

    fitted_path = os.path.join(args.out_dir, "fitted-sheet.json")
    va.save_sheet(sheet, fitted_path)

    result = {"metrics": args.metrics, "text_file": args.text_file,
              "engine": args.engine, "ref_profile": ref_profile,
              "rounds": history, "fitted_sheet": fitted_path}
    with open(os.path.join(args.out_dir, "loop-result.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    lines = []
    lines.append("# 音声ループ レポート（抽出→近付け→測定→補正）")
    lines.append("")
    lines.append(f"- お手本: {metrics.get('title') or args.metrics}")
    lines.append(f"- 台本: {args.text_file}（{len(script)}文）")
    lines.append(f"- エンジン: {args.engine}")
    lines.append("")
    lines.append("## お手本の型プロファイル")
    lines.append("")
    lines.append(f"- 話速 {fmt(ref_profile.get('speed_speech'), 0)} cpm"
                 f" / 間 {fmt(ref_profile.get('gap_median'))} 秒"
                 f" / F0中央値 {fmt(ref_profile.get('f0_median_hz'), 0)} Hz")
    lines.append(f"- 文のピッチ変化・中央値 {fmt(ref_profile.get('delta_median_st'))} 半音"
                 f"（出所: {ref_profile.get('delta_source')}）"
                 f" / 抑揚の幅・中央値 {fmt(ref_profile.get('span_median_st'))} 半音")
    lines.append(f"- 文末比率: {ref_profile.get('final_ratios')}")
    lines.append("")
    lines.append("## ラウンド経過")
    lines.append("")
    lines.append("| ラウンド | 類似度 | 話速 | 間 | ピッチ変化 | 抑揚の幅 | F0 |")
    lines.append("|---|---|---|---|---|---|---|")
    for h in history:
        d = h["dimensions"]
        def sc(key):
            return (f"{d[key]['score']:.0f}" if key in d else "—")
        lines.append(
            f"| {h['round']} | **{fmt(h['score'], 1)}** | {sc('speed')} "
            f"| {sc('gap')} | {sc('delta')} | {sc('span')} "
            f"| {sc('f0_median')} |")
    lines.append("")
    for h in history:
        if h["adjustments"]:
            lines.append(f"- ラウンド{h['round']}後の補正: "
                         + "／".join(h["adjustments"]))
    lines.append("")
    lines.append(f"最終シート: {fitted_path}")
    lines.append("")
    lines.append("※類似度は「型」の数値的な近さの物差しであり、"
                 "声質そのものの似せ度ではありません。文単位の型は"
                 "3分割整形による近似です。")
    report_path = os.path.join(args.out_dir, "loop-report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"OK: ループレポート -> {report_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="抽出→シート化→生成→類似度→補正の自動ループ")
    ap.add_argument("--metrics", required=True,
                    help="お手本の metrics.json（video_analyzer.py の出力）")
    ap.add_argument("--text-file", required=True, help="読み上げ台本")
    ap.add_argument("--engine", default="local",
                    choices=["gemini", "local"])
    ap.add_argument("--voice", help="声（local時はリア用voice IDが既定）")
    ap.add_argument("--rounds", type=int, default=3,
                    help="最大ラウンド数（既定3）")
    ap.add_argument("--target-score", type=float, default=85.0,
                    help="到達したら止める類似度スコア（既定85）")
    ap.add_argument("--out-dir", required=True, help="出力フォルダ")
    args = ap.parse_args()
    try:
        return run_loop(args)
    except LoopError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return e.code
    except va.AdjustError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
