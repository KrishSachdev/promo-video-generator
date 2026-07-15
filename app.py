"""
AUTOMATED PROMO VIDEO GENERATOR (CNN version)
=============================================
Generates short promotional videos from long-form content using a lightweight CNN
(EfficientNetB0) + optional audio features. Replaces PCA/z-score scoring with
a semantic visual score from a pretrained model.

INSTALL (CPU is fine):
---------------------
pip install moviepy opencv-python-headless librosa numpy scipy scikit-learn pillow imageio-ffmpeg tensorflow
# for scene-aware cuts:
pip install scenedetect[opencv]

Notes:
- Keep fps_sample small (2–4) to avoid slow inference.
- This uses transfer learning inference only (no training).
"""

from __future__ import annotations
import os
import cv2
import argparse
import warnings
import tempfile
from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np
import librosa
from moviepy.editor import VideoFileClip, concatenate_videoclips
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import sys

# TF quiet + imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# PySceneDetect (optional)
_SCENEDETECT_OK = True
try:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector
except Exception:  # pragma: no cover
    _SCENEDETECT_OK = False

warnings.filterwarnings("ignore")


# ----------------------- small utilities (less noise) -----------------------

def set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def log(*msg) -> None:
    print(*msg, flush=True)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) == 0:
        return x
    window = max(1, min(window, len(x)))
    if window == 1:
        return x.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _remove_overlaps(segs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(segs) <= 1:
        return segs
    segs.sort(key=lambda s: s[0])
    merged = [segs[0]]
    for a, b in segs[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def save_artifacts(
    out_dir: Path, run_tag: str, base_stem: str, scores: np.ndarray | None, segs: List[Tuple[float, float]] | None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if scores is not None:
        np.save(out_dir / f"scores_{run_tag}_{base_stem}.npy", scores)
        log(f"Saved per-sample scores to: {out_dir / f'scores_{run_tag}_{base_stem}.npy'}")
    if segs is not None:
        import csv
        path = out_dir / f"segments_{run_tag}_{base_stem}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_sec", "end_sec"])
            for s, e in segs:
                w.writerow([round(float(s), 3), round(float(e), 3)])
        log(f"Saved selected segments to: {path}")


# ------------------------------- CNN scorer -------------------------------

class PromoCNN:
    """
    EfficientNet-based feature extractor with memory-safe batched scoring.
    The constructor uses EfficientNetB0 with global pooling to produce a compact
    embedding for each frame. The scores(...) method processes frames in batches
    and converts embeddings into a 0-1 score via min-max normalization of the
    embedding L2 norms. This keeps behavior simple and avoids adding extra trainable layers.
    """

    def __init__(self, img_size: int = 224):
        # EfficientNetB0 without top, with global average pooling to get compact embeddings
        base = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg", input_shape=(img_size, img_size, 3))
        self.model = base  # model returns (batch, features)
        self.img_size = img_size

    def _prep_single(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        arr = rgb.astype("float32")
        arr = preprocess_input(arr)  # EfficientNet preprocessing
        return arr

    def scores(self, frames: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """
        Compute a per-frame scalar score for each frame in `frames`.
        Processing is done in batches to avoid large memory spikes.
        Returns a 1D numpy array of float32 scores in range [0,1].
        """
        if not frames:
            return np.array([], dtype=np.float32)

        # Prepare frames incrementally and run inference in batches
        n = len(frames)
        norms = []
        i = 0
        while i < n:
            end = min(n, i + batch_size)
            batch_frames = [self._prep_single(f) for f in frames[i:end]]
            batch = np.stack(batch_frames, axis=0)
            # predict embeddings
            emb = self.model.predict(batch, verbose=0)
            # compute L2 norm per embedding as a simple scalar signal
            batch_norms = np.linalg.norm(emb, axis=1)
            norms.extend(batch_norms.tolist())
            i = end

        norms = np.array(norms, dtype=np.float32)
        if norms.size == 0:
            return np.zeros(0, dtype=np.float32)

        # Normalize to 0-1 to get a pseudo-score
        minv = float(norms.min())
        maxv = float(norms.max())
        rng = maxv - minv
        if rng <= 1e-6:
            scores = np.clip((norms - minv), 0.0, 1.0)
        else:
            scores = (norms - minv) / rng
        return scores.astype(np.float32)


# ----------------------------- core generator -----------------------------

class PromoVideoGenerator:
    """
    - Visual scoring: EfficientNetB0 embeddings -> L2 norm -> normalized score
    - Audio: RMS + onset (optional fusion)
    - Selection: peaks on smoothed CNN scores
    - Optional scene snapping via PySceneDetect
    """

    def __init__(
        self,
        target_duration: int = 30,
        fps_sample: int = 2,
        scene_snap: bool = True,
        save_scores: bool = False,
        run_tag: str = "run",
        out_dir: str | Path = "eval_artifacts",
    ):
        self.target_duration = int(target_duration)
        self.fps_sample = max(1, int(fps_sample))
        self.scene_snap = bool(scene_snap)
        self.scaler = StandardScaler()
        self.cnn = PromoCNN(img_size=224)
        self.save_scores = bool(save_scores)
        self.run_tag = str(run_tag)
        self.out_dir = Path(out_dir)

    # ------------------------------ features ------------------------------

    def _extract_visual_scores(self, video_path: Path, duration: float, progress_cb=None) -> np.ndarray:
        """
        Memory-safe visual extraction with frame skipping and batched CNN inference.
        progress_cb, if provided, will be called with values in [0,1].
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log("Could not open video for visual CNN analysis; using flat scores.")
            return np.ones(int(max(1, duration * self.fps_sample)), dtype=np.float32)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else int(max(1, fps * duration))

        # sample_every controls sampling rate relative to video fps and desired fps_sample
        sample_every = max(1, int(round(fps / float(self.fps_sample)))) if self.fps_sample > 0 else max(1, int(round(fps)))
        # additional frame skipping factor to reduce memory on constrained hosts
        # tune this if you need fewer frames; default keeps some sampling density
        frame_skip = max(1, sample_every * 5)

        frames_buffer: List[np.ndarray] = []
        idx = 0
        extracted = 0
        expected = max(1, total_frames // frame_skip)

        ok, frame = cap.read()
        while ok:
            if idx % frame_skip == 0:
                # keep original BGR frame for later preprocessing in PromoCNN
                frames_buffer.append(frame)
                extracted += 1
                if progress_cb is not None:
                    try:
                        progress_cb(min(0.5, extracted / expected * 0.5))  # up to 0.5 for frame collection
                    except Exception:
                        pass
            idx += 1
            ok, frame = cap.read()

        cap.release()

        if not frames_buffer:
            return np.ones(int(max(1, duration * self.fps_sample)), dtype=np.float32)

        # Compute scores in batches using PromoCNN.scores
        # We will call scores in batches via the PromoCNN method which itself batches.
        # Provide a small progress update before and after model inference.
        if progress_cb is not None:
            try:
                progress_cb(0.5)
            except Exception:
                pass

        scores = self.cnn.scores(frames_buffer, batch_size=8)

        # indicate visual stage nearly finished
        if progress_cb is not None:
            try:
                progress_cb(0.65)
            except Exception:
                pass

        if scores.size == 0:
            return np.ones(len(frames_buffer), dtype=np.float32)
        return scores

    def _extract_audio_features(self, clip: VideoFileClip, duration: float) -> np.ndarray:
        if clip.audio is None:
            return np.zeros((int(max(1, duration * self.fps_sample)), 2), dtype=np.float32)

        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = Path(tmp.name)
            tmp.close()
            clip.audio.write_audiofile(
                str(wav_path), fps=22050, nbytes=2, codec="pcm_s16le", verbose=False, logger=None
            )
            y, _ = librosa.load(str(wav_path), sr=22050, mono=True)
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception:
            return np.zeros((int(max(1, duration * self.fps_sample)), 2), dtype=np.float32)

        frame_length = max(1, int(22050 / self.fps_sample))
        hop = frame_length
        try:
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
            onset = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
            m = min(len(rms), len(onset))
            if m == 0:
                return np.zeros((int(max(1, duration * self.fps_sample)), 2), dtype=np.float32)
            return np.column_stack([rms[:m], onset[:m]]).astype(np.float32)
        except Exception:
            return np.zeros((int(max(1, duration * self.fps_sample)), 2), dtype=np.float32)

    def extract_features(self, video_path: Path, progress_cb=None) -> Tuple[np.ndarray, float]:
        log("Extracting features with CNN + Audio (no PCA)...")
        # Open once and reuse duration/audio
        with VideoFileClip(str(video_path)) as probe:
            duration = float(probe.duration)
            audio_feats = self._extract_audio_features(probe, duration)

        vis_scores = self._extract_visual_scores(video_path, duration, progress_cb=progress_cb)

        # if a progress callback exists, indicate audio+fusion will happen next
        if progress_cb is not None:
            try:
                progress_cb(0.75)
            except Exception:
                pass

        m = min(len(vis_scores), len(audio_feats))
        if m == 0:
            log("Not enough features; using flat scores.")
            return np.ones((max(len(vis_scores), len(audio_feats), 1), 1), dtype=np.float32), duration

        fused = np.column_stack([vis_scores[:m], audio_feats[:m]])  # [cnn, rms, onset]
        fused_norm = self.scaler.fit_transform(fused)

        # signal near completion of feature extraction
        if progress_cb is not None:
            try:
                progress_cb(0.9)
            except Exception:
                pass

        log("Time steps:", fused_norm.shape[0])
        log("Features per step: cnn_score + audio(2) =", fused_norm.shape[1])
        return fused_norm.astype(np.float32), duration

    # ----------------------------- scoring + scenes -----------------------------

    def score_signal(self, feats: np.ndarray, window_s: float = 2.0) -> np.ndarray:
        if feats.shape[0] < 3:
            return np.ones(feats.shape[0], dtype=np.float32)
        base = feats[:, 0].astype(np.float32)
        window = max(3, int(self.fps_sample * window_s))
        return _smooth(base, window)

    def detect_scene_bounds(self, video_path: Path, threshold: float = 27.0) -> List[Tuple[float, float]]:
        if not self.scene_snap or not _SCENEDETECT_OK:
            return []
        try:
            vid = open_video(str(video_path))
            sm = SceneManager()
            sm.add_detector(ContentDetector(threshold=threshold))
            sm.detect_scenes(vid)
            scenes = sm.get_scene_list()
            return [(s.get_seconds(), e.get_seconds()) for (s, e) in scenes]
        except Exception:
            return []

    def _snap_to_scenes(
        self, segs: List[Tuple[float, float]], bounds: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not self.scene_snap or not bounds:
            return segs
        snapped: List[Tuple[float, float]] = []
        for a, b in segs:
            s_start, s_end = a, b
            for x, y in bounds:
                if x <= a <= y:  # start scene
                    s_start, s_end = x, y
                    break
            if b > s_end:  # extend into next scene end if needed
                for x, y in bounds:
                    if x < b <= y:
                        s_end = y
                        break
            if s_end - s_start >= 0.8:
                snapped.append((max(0.0, s_start), s_end))
        return _remove_overlaps(snapped) if snapped else segs

    # ------------------------------ selection ------------------------------

    def select_segments(self, scores: np.ndarray, video_duration: float) -> List[Tuple[float, float]]:
        log("Selecting optimal segments...")
        if video_duration <= self.target_duration:
            log("Video shorter than target; using full video")
            return [(0.0, float(video_duration))]

        prom = max(0.15, float(scores.std()) * 0.5)
        dist = max(3, int(self.fps_sample * 2))
        peaks, _ = find_peaks(scores, prominence=prom, distance=dist)

        if peaks.size == 0:
            log("No peaks found; using top values directly")
            k = max(3, int(self.target_duration / 5))
            peaks = np.argsort(scores)[-k:]
            peaks.sort()

        t_per = float(video_duration) / float(len(scores))
        peak_times = peaks.astype(np.float32) * t_per
        peak_scores = scores[peaks]

        num_clips = min(len(peaks), max(3, int(self.target_duration / 3)))
        sel_times = np.sort(peak_times[np.argsort(peak_scores)[-num_clips:]])

        segments: List[Tuple[float, float]] = []
        clip_len = min(video_duration / 3.0, (self.target_duration / max(1, len(sel_times))) * 1.2)

        for t in sel_times:
            a = max(0.0, float(t) - clip_len / 2.0)
            b = min(float(video_duration), float(t) + clip_len / 2.0)
            if b - a < 1.0:
                a = max(0.0, float(t) - 0.5)
                b = min(float(video_duration), float(t) + 0.5)
            segments.append((a, b))

        return _remove_overlaps(segments)

    # ------------------------------- assembly -------------------------------

    def create_promo(
        self,
        video_path: str | Path,
        output_path: str | Path,
        add_effects: bool = True,
        scene_threshold: float = 27.0,
        progress_cb=None,
    ) -> None:
        video_path = Path(video_path)
        output_path = Path(output_path)
        base_stem = video_path.stem

        log("\nStarting promo generation (CNN)…")
        log("=" * 60)

        feats, duration = self.extract_features(video_path, progress_cb=progress_cb)

        # indicate scoring stage start
        if progress_cb is not None:
            try:
                progress_cb(0.95)
            except Exception:
                pass

        scores = self.score_signal(feats)
        if self.save_scores:
            save_artifacts(self.out_dir, self.run_tag, base_stem, scores, None)

        segs = self.select_segments(scores, duration)
        bounds = self.detect_scene_bounds(video_path, threshold=scene_threshold)
        if bounds:
            segs = self._snap_to_scenes(segs, bounds)
            log(f"Snapped to {len(bounds)} detected scenes.")
        if self.save_scores:
            save_artifacts(self.out_dir, self.run_tag, base_stem, None, segs)

        log("\nAssembling clips…")
        clips = []
        with VideoFileClip(str(video_path)) as vid:
            for i, (a, b) in enumerate(segs):
                try:
                    clip = vid.subclip(a, b)
                    if add_effects and clip.duration > 1.0 and (i % 2 == 0):
                        clip = clip.speedx(1.1)
                    if clip.duration > 0.6:
                        clip = clip.crossfadein(0.3).crossfadeout(0.3)
                    clips.append(clip)
                except Exception as e:
                    log(f"Skipping segment {i+1}: {e}")

            if not clips:
                raise ValueError("No valid clips could be extracted")

            log(f"Concatenating {len(clips)} clips…")
            final = concatenate_videoclips(clips, method="compose")
            if final.duration > self.target_duration:
                final = final.subclip(0, self.target_duration)

            log("\nRendering final promo (secs):", round(final.duration, 1))
            final.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                fps=24,
                preset="medium",
                threads=4,
                verbose=False,
                logger=None,
            )

        # final progress update
        if progress_cb is not None:
            try:
                progress_cb(1.0)
            except Exception:
                pass

        log("\n" + "=" * 60)
        log("SUCCESS! Promo saved to:", str(output_path))
        log("=" * 60)
        try:
            size_mb = output_path.stat().st_size / (1024 * 1024.0)
        except Exception:
            size_mb = -1.0
        log("Stats")
        log(" - Original duration (s):", round(duration, 1))
        log(" - Promo duration (s):   ", round(min(self.target_duration, duration), 1))
        log(" - Compression ratio:    ", round(duration / max(1e-6, min(self.target_duration, duration)), 1))
        log(" - Segments used:        ", len(segs))
        log(" - File size (MB):       ", round(size_mb, 2))
        log("=" * 60)


# ----------------------------------- CLI -----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate promotional videos using CNN-powered analysis")
    p.add_argument("--input", "-i", required=True, help="Input video file path")
    p.add_argument("--output", "-o", default="promo_output.mp4", help="Output promo file path")
    p.add_argument("--duration", "-d", type=int, default=30, help="Target duration in seconds")
    p.add_argument("--fps", type=int, default=2, help="Sampling fps for analysis")
    p.add_argument("--no-effects", action="store_true", help="Disable subtle speed/fade effects")
    p.add_argument("--no-scene-snap", action="store_true", help="Disable snapping to scene boundaries")
    p.add_argument("--scene-threshold", type=float, default=27.0, help="PySceneDetect content threshold")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--save-scores", action="store_true", help="Persist scores/segments for evaluation")
    p.add_argument("--run-tag", type=str, default="run", help="Tag for saved artifacts")
    p.add_argument("--out-dir", type=str, default="eval_artifacts", help="Directory for artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        log("Error: Input file not found:", str(in_path))
        return

    set_global_seeds(args.seed)

    log("\n" + "=" * 60)
    log("AUTOMATED PROMO VIDEO GENERATOR (CNN)")
    log("=" * 60)
    log("Input:", str(in_path))
    log("Output:", args.output)
    log("Target Duration (s):", args.duration)
    log("Analysis FPS:", args.fps)
    log("Scene Snapping:", "ON" if not args.no_scene_snap else "OFF", "(PySceneDetect)" if _SCENEDETECT_OK else "(Unavailable)")
    log("Scene Threshold:", args.scene_threshold)
    log("Seed:", args.seed)
    log("Save Artifacts:", "ON" if args.save_scores else "OFF", "->", args.out_dir)
    log("=" * 60 + "\n")

    gen = PromoVideoGenerator(
        target_duration=args.duration,
        fps_sample=args.fps,
        scene_snap=(not args.no_scene_snap),
        save_scores=args.save_scores,
        run_tag=args.run_tag,
        out_dir=args.out_dir,
    )
    gen.create_promo(
        video_path=in_path,
        output_path=Path(args.output),
        add_effects=(not args.no-effects if hasattr(args, "no-effects") else not args.no_effects),
        scene_threshold=args.scene_threshold,
    )


if __name__ == "__main__":
    # Avoid triggering CLI when running under Streamlit
    if 'streamlit' not in sys.modules:
        main()

# ------------------------------ Streamlit UI ------------------------------

def _render_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Promo Video Generator", page_icon="🎬", layout="centered")
    st.title("Automated Promo Video Generator (CNN)")
    st.caption("Upload a source video, configure options, and generate a short promo.")

    with st.sidebar:
        st.header("Options")
        target_duration = st.slider("Target duration (seconds)", min_value=10, max_value=120, value=30, step=5)
        fps_sample = st.slider("Analysis FPS (samples/sec)", min_value=1, max_value=8, value=2, step=1)
        scene_snap = st.checkbox("Snap to scene boundaries", value=True)
        scene_threshold = st.slider("Scene threshold", min_value=5.0, max_value=40.0, value=27.0, step=1.0)
        add_effects = st.checkbox("Subtle effects (speed/fades)", value=True)
        save_scores = st.checkbox("Save artifacts (scores/segments)", value=False)
        run_tag = st.text_input("Run tag", value="run")
        out_dir = st.text_input("Artifacts output dir", value="eval_artifacts")
        seed = st.number_input("Seed", min_value=0, value=123, step=1)

    uploaded = st.file_uploader("Upload input video", type=["mp4", "mov", "mkv", "avi"])

    if uploaded is not None:
        st.video(uploaded)

    col1, col2 = st.columns(2)
    generate_clicked = col1.button("Generate Promo", type="primary", disabled=(uploaded is None))
    reset_clicked = col2.button("Reset")

    if reset_clicked:
        st.experimental_rerun()

    if generate_clicked and uploaded is not None:
        try:
            set_global_seeds(int(seed))

            with st.status("Preparing input…", expanded=False) as status:
                status.update(label="Saving uploaded video")
                src_tmp = tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False)
                src_tmp.write(uploaded.read())
                src_tmp.flush()
                src_path = Path(src_tmp.name)
                src_tmp.close()

                status.update(label="Setting up generator")
                gen = PromoVideoGenerator(
                    target_duration=int(target_duration),
                    fps_sample=int(fps_sample),
                    scene_snap=bool(scene_snap),
                    save_scores=bool(save_scores),
                    run_tag=str(run_tag),
                    out_dir=str(out_dir),
                )

            st.write("Generating promo — this can take a little time…")
            progress = st.progress(0)

            out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            out_path = Path(out_tmp.name)
            out_tmp.close()

            # Run generation with progress callback
            gen.create_promo(
                video_path=src_path,
                output_path=out_path,
                add_effects=bool(add_effects),
                scene_threshold=float(scene_threshold),
                progress_cb=lambda p: progress.progress(int(max(0, min(100, p * 100))))
            )

            st.success("Promo generated!")
            with out_path.open("rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(
                label="Download Promo MP4",
                data=video_bytes,
                file_name="promo_output.mp4",
                mime="video/mp4",
            )

            # Cleanup temp source file to save space
            try:
                src_path.unlink(missing_ok=True)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Failed to generate promo: {e}")


# Render UI when executed by Streamlit
try:
    import streamlit as _st  # noqa: F401
    _render_streamlit_app()
except Exception:
    # Not running under Streamlit or import failed; ignore UI path.
    pass
