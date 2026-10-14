# -*- coding: utf-8 -*-
"""
크로마 특징 추출 — MIDI(ASAP)와 실제 오디오(week4) 양쪽.

원본:
  - load_notes / notes_to_chroma_stream: scripts/p1_verification.py
  - audio_chroma: scripts/week4_gt_pipeline.py
  - build_score_chroma_v2: scripts/week4_score_chroma_v2.py
    (2026-09-07 진단으로 확인된 최선의 week4 스코어 표현 — 교과서 triad(0.50) 대신
     chord_hints 실제 보이싱(0.57) + chord_timeline의 검증된 구간 경계)
계산 로직은 원본과 동일, config 기반 재실행을 위해 이동만 했다.
"""
import json

import numpy as np
import pretty_midi

PITCH_CLASSES = 12


def load_notes(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch))
    notes.sort(key=lambda x: x[0])
    return notes


def notes_to_chroma_stream(notes, duration, hop):
    """고정 hop으로 프레임별 크로마(피치클래스 에너지, 지속시간 가중) 생성.
    각 프레임을 L2 정규화한다."""
    n_frames = max(2, int(np.ceil(duration / hop)))
    chroma = np.zeros((n_frames, PITCH_CLASSES))
    for (ns, ne, pitch) in notes:
        f0 = max(0, int(ns / hop))
        f1 = min(n_frames - 1, int(np.ceil(ne / hop)))
        for f in range(f0, f1 + 1):
            fs, fe = f * hop, (f + 1) * hop
            overlap = min(ne, fe) - max(ns, fs)
            if overlap > 0:
                chroma[f, pitch % PITCH_CLASSES] += overlap
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return chroma / norms


def audio_chroma(path, hop=0.1, sr=22050):
    """실제 오디오(week4 기타 녹음)에서 librosa chroma_cqt로 크로마 추출."""
    import librosa

    y, sr = librosa.load(path, sr=sr, mono=True)
    hop_length = int(sr * hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = chroma.T  # (frames, 12)
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return chroma / norms, len(y) / sr


def build_score_chroma_v2(song_package_path, hop_sec=0.1, max_sec=90.0, recording_len=None):
    """
    chord_timeline의 구간 경계(타이밍, onset-detection으로 검증됨, 평균오차 85ms) +
    chord_hints의 실제 프렛/현 MIDI 노트(교과서 triad 대신 실제 연주 보이싱)를 합쳐
    week4용 스코어 크로마를 만든다.

    각 chord_timeline 구간에 대해, 그 구간(±50ms 여유)에 걸치는 chord_hints의 모든
    노트 이벤트의 pitch class 합집합을 그 구간의 크로마로 사용한다(균일 가중).
    """
    with open(song_package_path, encoding="utf-8") as f:
        pkg = json.load(f)

    timeline = [e for e in pkg["chord_timeline"] if e["chord"] != "N"]
    chord_hints = pkg["tab_tracking"]["chord_hints"]

    tl_starts = sorted(e["start_sec"] for e in timeline if e["start_sec"] < max_sec)

    all_notes = []  # (절대시각, pitch_class)
    for hint in chord_hints:
        base = hint["start_sec"]
        for ev in hint.get("events", []):
            t = base + ev["offset_sec"]
            for n in ev.get("notes", []):
                all_notes.append((t, n["midi"] % 12))
    all_notes.sort(key=lambda x: x[0])

    n_frames = int(np.ceil(max_sec / hop_sec))
    chroma = np.zeros((n_frames, PITCH_CLASSES))

    for i, t0 in enumerate(tl_starts):
        t1 = tl_starts[i + 1] if i + 1 < len(tl_starts) else max_sec
        t1 = min(t1, max_sec)
        if recording_len is not None and t0 >= recording_len:
            continue
        pcs = set(pc for t, pc in all_notes if t0 - 0.05 <= t < t1 + 0.05)
        if not pcs:
            continue
        f0 = int(t0 / hop_sec)
        f1 = min(n_frames - 1, int(np.ceil(t1 / hop_sec)))
        for f in range(f0, f1 + 1):
            for pc in pcs:
                chroma[f, pc] = 1.0

    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return chroma / norms
