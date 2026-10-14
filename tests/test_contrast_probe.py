import numpy as np
from score_following.contrast_probe import piano_roll, normalize, weights, choose, window
from score_following.features import notes_to_chroma_stream


def test_pitch_fold_matches_original_overlap_and_octave_collision():
    notes = [(0.,.31,60),(.12,.44,72),(.25,.39,67)]
    roll = piano_roll(notes,1.)
    folded = np.zeros((len(roll),12))
    for pitch in range(128):
        folded[:,pitch%12] += roll[:,pitch]
    np.testing.assert_allclose(normalize(folded),notes_to_chroma_stream(notes,1.,.1))
    assert not np.array_equal(roll[:,60],roll[:,72])


def test_contrast_is_gt_free_permutation_equivariant_and_null_safe():
    rng = np.random.default_rng(19)
    templates = rng.random((4,20,128))
    observed = templates[2].copy()
    w = weights(templates)
    assert choose(templates,observed,w)==2
    perm = rng.permutation(128)
    np.testing.assert_allclose(weights(templates[:,:,perm]),w[perm])
    assert choose(templates[:,:,perm],observed[:,perm],w[perm])==2
    assert choose(np.zeros((3,20,128)),np.zeros((20,128)),weights(np.zeros((3,20,128))))==0
    assert window(np.zeros((30,128)),25,25).shape==(20,128)
