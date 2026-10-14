import numpy as np

from score_following.features import notes_to_chroma_stream


def test_simple_progression_maps_to_expected_pitch_classes():
    """C major triad (C4=60,E4=64,G4=68) 0~1초 -> 프레임의 피치클래스가
    {0(C), 4(E), 7(G)}에만 에너지를 가져야 한다."""
    notes = [(0.0, 1.0, 60), (0.0, 1.0, 64), (0.0, 1.0, 67)]
    chroma = notes_to_chroma_stream(notes, duration=1.0, hop=0.5)

    assert chroma.shape == (2, 12)
    for frame in chroma:
        nonzero_pcs = set(np.nonzero(frame)[0].tolist())
        assert nonzero_pcs == {0, 4, 7}
        assert np.isclose(np.linalg.norm(frame), 1.0)


def test_octave_collapses_to_same_pitch_class():
    notes = [(0.0, 0.5, 60), (0.0, 0.5, 72)]  # 두 옥타브 차이 C, 둘 다 pitch class 0
    chroma = notes_to_chroma_stream(notes, duration=0.5, hop=0.5)
    nonzero_pcs = set(np.nonzero(chroma[0])[0].tolist())
    assert nonzero_pcs == {0}
