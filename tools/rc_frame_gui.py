#!/usr/bin/env python3
"""Local browser GUI for the RealityCapture frame extractor."""

from __future__ import annotations

import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

import rc_frame_extractor as extractor


SMART_PRESETS = {
    "標準（おすすめ）": {
        "sample_fps": 3.0,
        "min_sharpness": 18.0,
        "min_difference": 0.018,
        "max_gap": 1.5,
        "caption": "歩く程度の速度で撮影した動画向け。連続性と枚数のバランスを取ります。",
    },
    "多めに残す": {
        "sample_fps": 5.0,
        "min_sharpness": 14.0,
        "min_difference": 0.012,
        "max_gap": 1.0,
        "caption": "速めの移動や細部重視向け。処理時間と画像枚数は増えます。",
    },
    "少なめに絞る": {
        "sample_fps": 2.0,
        "min_sharpness": 22.0,
        "min_difference": 0.030,
        "max_gap": 2.0,
        "caption": "ゆっくり撮影した長い動画向け。似た画像を強めに間引きます。",
    },
}


st.set_page_config(
    page_title="RC Frame Studio",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #f4f1eb; color: #1e2927; }
      .block-container { max-width: 1120px; padding-top: 2.4rem; padding-bottom: 4rem; }
      [data-testid="stHeader"] { background: transparent; }
      .hero {
        padding: 2.2rem 2.4rem; border-radius: 24px; color: #f8faf6;
        background: linear-gradient(125deg, #173d36 0%, #1f5a4e 58%, #bd7544 145%);
        box-shadow: 0 18px 50px rgba(23, 61, 54, .16); margin-bottom: 1.5rem;
      }
      .hero .eyebrow { letter-spacing: .14em; font-size: .75rem; opacity: .72; font-weight: 700; }
      .hero h1 { font-size: clamp(2rem, 5vw, 3.5rem); margin: .35rem 0 .5rem; line-height: 1.05; }
      .hero p { max-width: 700px; opacity: .84; font-size: 1.05rem; margin: 0; }
      [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.64); border: 1px solid #d9d3c8;
        border-radius: 18px; padding: .75rem;
      }
      [data-testid="stMetric"] {
        background: rgba(255,255,255,.72); border: 1px solid #ded8ce;
        border-radius: 16px; padding: 1rem 1.1rem;
      }
      .stButton > button[kind="primary"] {
        border-radius: 999px; background: #c86f3e; border-color: #c86f3e;
        min-height: 3rem; font-weight: 700;
      }
      .stButton > button[kind="primary"]:hover { background: #a9552e; border-color: #a9552e; }
      .small-note { color: #697571; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">REALITYCAPTURE PREP</div>
      <h1>RC Frame Studio</h1>
      <p>実績のある固定fps抽出を標準に、必要なときだけ自動選別を追加できます。</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if "output_root" not in st.session_state:
    st.session_state.output_root = str(Path.home() / "Desktop")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    with st.form("source_form"):
        st.subheader("1 · 動画を選ぶ")
        uploaded = st.file_uploader(
            "ここへMOVをドロップ",
            type=["mov", "mp4", "m4v"],
            key="video_upload",
            help="MOV、MP4、M4Vに対応します。",
        )
        st.caption("または、大きな動画はファイルパスを直接指定")
        source_path_text = st.text_input(
            "MOVのパス",
            placeholder="/Users/yourname/Movies/scan.mov",
            key="source_path",
            help="Finderで動画を選び、⌥⌘Cでパスをコピーできます。",
        )

        st.subheader("2 · 保存先")
        output_root = st.text_input(
            "保存先フォルダ",
            key="output_root",
            help="この中に日時付きの新しいフォルダを作成します。既存画像とは混ざりません。",
        )
        st.markdown(
            '<p class="small-note">例：デスクトップに <b>RC_frames_20260717_183000</b> を作成</p>',
            unsafe_allow_html=True,
        )
        run_clicked = st.form_submit_button(
            "フレーム抽出を開始",
            type="primary",
            width="stretch",
        )

with right:
    st.subheader("3 · 抽出設定")
    processing_mode = st.radio(
        "処理方式",
        ["固定fpsで直接抽出（推奨）", "ブレ・重複を自動選別"],
        index=0,
    )
    if processing_mode == "固定fpsで直接抽出（推奨）":
        st.info("以前の `-vf fps=10 -q:v 1` と同じ一回抽出方式です。")
        direct_fps = st.select_slider(
            "1秒あたりの画像枚数",
            options=[1, 2, 3, 5, 8, 10],
            value=10,
            format_func=lambda value: f"{value} fps",
        )
        jpeg_quality = st.select_slider(
            "JPEG品質（小さいほど高画質）",
            options=[1, 2, 3, 4, 5],
            value=1,
            format_func=lambda value: f"q:v {value}",
        )
        allow_large_job = st.checkbox(
            "30,000枚を超える長時間処理を許可",
            help="必要な場合だけ有効にしてください。10fpsでは約50分が30,000枚です。",
        )
    else:
        st.warning("自動選別は二回デコードします。まず固定fps抽出を試すことを推奨します。")
        preset_name = st.radio("選別プリセット", list(SMART_PRESETS), index=0)
        preset = SMART_PRESETS[preset_name]
        st.caption(preset["caption"])
        with st.expander("詳細設定"):
            sample_fps = st.slider(
                "候補フレーム／秒", 1.0, 8.0, preset["sample_fps"], 0.5,
                help="カメラ移動が速い場合は増やします。",
            )
            min_sharpness = st.slider(
                "鮮明度の下限", 5.0, 50.0, preset["min_sharpness"], 1.0,
                help="高くすると、ブレた画像を強めに除外します。",
            )
            min_difference = st.slider(
                "似た画像の間引き", 0.005, 0.060, preset["min_difference"], 0.001,
                format="%.3f",
                help="高くすると、似た画像を強めに間引きます。",
            )
            max_gap = st.slider(
                "最大画像間隔（秒）", 0.5, 3.0, preset["max_gap"], 0.25,
                help="画像のつながりが途切れないための救済間隔です。",
            )


def process_direct(source: Path, output: Path) -> tuple[int, int]:
    ffmpeg = extractor.command_path("ffmpeg")
    ffprobe = extractor.command_path("ffprobe")
    duration = extractor.video_duration(source, ffprobe)
    estimated_frames = math.ceil(duration * direct_fps)
    if estimated_frames > extractor.MAX_DIRECT_FRAMES and not allow_large_job:
        raise RuntimeError(
            f"約{estimated_frames:,}枚になるため処理前に停止しました。"
            "fpsを下げるか、動画を分割してください。必要なら大量処理の許可も選べます。"
        )
    with st.status("固定fpsで画像を書き出しています…", expanded=True) as status:
        st.write(f"動画時間：{duration:.1f}秒")
        st.write(f"設定：{direct_fps} fps / q:v {jpeg_quality}")
        st.write(f"予定枚数：約{estimated_frames:,}枚")
        frames = extractor.extract_frames_direct(
            source, output, direct_fps, ffmpeg, jpeg_quality
        )
        extractor.write_direct_report(
            source, output, direct_fps, jpeg_quality, len(frames)
        )
        status.update(label="直接抽出が完了しました", state="complete", expanded=False)
    return len(frames), estimated_frames


def process_smart(source: Path, output: Path) -> tuple[int, int]:
    ffmpeg = extractor.command_path("ffmpeg")
    ffprobe = extractor.command_path("ffprobe")
    duration = extractor.video_duration(source, ffprobe)
    estimated_candidates = math.ceil(duration * sample_fps)
    if estimated_candidates > extractor.MAX_CANDIDATES:
        raise RuntimeError(
            f"候補が約{estimated_candidates:,}枚になるため、安全上停止しました。"
            "「少なめに絞る」を選ぶか、動画を分割してください。"
        )
    with st.status("動画を解析しています…", expanded=True) as status:
        st.write(f"動画時間：{duration:.1f} 秒")
        with tempfile.TemporaryDirectory(prefix="rc-gui-") as temporary:
            st.write(f"低解像度の候補を約{estimated_candidates:,}枚デコード中…")
            candidates = extractor.extract_candidates(source, Path(temporary), sample_fps, ffmpeg)
            if not candidates:
                raise RuntimeError("動画からフレームを読み取れませんでした。")
            progress = st.progress(0, text="鮮明度を計測中…")

            def update(done: int, total: int) -> None:
                progress.progress(done / total, text=f"鮮明度を計測中… {done}/{total}")

            metrics = extractor.analyze(candidates, sample_fps, update)
            kept, decisions = extractor.choose_frames(
                metrics, min_sharpness, min_difference, max_gap
            )
            if not kept:
                raise RuntimeError("採用できる画像がありません。鮮明度の下限を下げてください。")
            st.write(f"採用した{len(kept):,}枚だけを元解像度で書き出し中…")
            extractor.output_frames_from_video(
                source, kept, output, sample_fps, ffmpeg
            )
            extractor.write_report(decisions, output)
        status.update(label="抽出が完了しました", state="complete", expanded=False)
    return len(kept), len(decisions)


if run_clicked:
    if not output_root.strip():
        st.error("保存先フォルダを指定してください。")
    elif not source_path_text.strip() and uploaded is None:
        st.error("MOVファイルを選択してください。")
    elif not (output_base := Path(output_root).expanduser()).exists() or not output_base.is_dir():
        st.error("保存先フォルダが見つかりません。既存のフォルダを指定してください。")
    else:
        destination = output_base / f"RC_frames_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            if source_path_text.strip():
                local_source = Path(source_path_text.strip()).expanduser()
                if not local_source.is_file():
                    raise FileNotFoundError(f"動画が見つかりません：{local_source}")
                if processing_mode == "固定fpsで直接抽出（推奨）":
                    kept_count, candidate_count = process_direct(local_source, destination)
                else:
                    kept_count, candidate_count = process_smart(local_source, destination)
            else:
                suffix = Path(uploaded.name).suffix.lower() or ".mov"
                with tempfile.TemporaryDirectory(prefix="rc-upload-") as upload_directory:
                    local_source = Path(upload_directory) / f"input{suffix}"
                    with local_source.open("wb") as file:
                        shutil.copyfileobj(uploaded, file)
                    if processing_mode == "固定fpsで直接抽出（推奨）":
                        kept_count, candidate_count = process_direct(local_source, destination)
                    else:
                        kept_count, candidate_count = process_smart(local_source, destination)
            st.session_state.last_result = {
                "output": str(destination),
                "kept": kept_count,
                "candidates": candidate_count,
                "mode": processing_mode,
            }
        except Exception as error:
            st.error(f"処理を完了できませんでした：{error}")


result = st.session_state.get("last_result")
if result:
    st.divider()
    st.subheader("抽出結果")
    if result.get("mode") == "固定fpsで直接抽出（推奨）":
        metric_a, metric_b = st.columns(2)
        metric_a.metric("出力画像", f"{result['kept']:,} 枚")
        metric_b.metric("処理方式", "固定fps・直接抽出")
    else:
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("採用画像", f"{result['kept']:,} 枚")
        metric_b.metric("候補画像", f"{result['candidates']:,} 枚")
        rate = result["kept"] / result["candidates"] * 100 if result["candidates"] else 0
        metric_c.metric("採用率", f"{rate:.0f}%")
    st.success(f"保存先：{result['output']}")

    preview_files = sorted(Path(result["output"]).glob("*.jpg"))[:12]
    if preview_files:
        st.caption("先頭から最大12枚をプレビュー")
        columns = st.columns(4)
        for index, image_path in enumerate(preview_files):
            columns[index % 4].image(str(image_path), caption=image_path.stem, width="stretch")

st.divider()
st.caption(
    "最終確認：RealityCaptureでAlignment後、カメラ位置が連続しているか、複数コンポーネントに分かれていないか確認してください。"
)
