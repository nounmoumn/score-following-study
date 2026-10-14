from scripts import p3_skypiano_two_axis_split as split


def synthetic_trials():
    rows = []
    for participant in range(12):
        for piece in range(12):
            rows.append(
                {
                    "trial_id": f"t-{participant}-{piece}",
                    "participant_id": f"p{participant}",
                    "expertise": "professional" if participant < 6 else "amateur",
                    "piece_id": f"w{piece}",
                    "piece_stratum": "graded" if piece < 6 else "exercise",
                    "duration_sec": 60.0,
                }
            )
    return rows


def test_two_axis_split_is_deterministic_and_disjoint():
    first = split.build_split(synthetic_trials())
    second = split.build_split(list(reversed(synthetic_trials())))
    assert first["participant_assignment"] == second["participant_assignment"]
    assert first["piece_assignment"] == second["piece_assignment"]
    development = [row for row in first["trials"] if row["cell"] == "development"]
    lockbox = [row for row in first["trials"] if row["cell"] == "primary_lockbox"]
    assert {row["participant_id"] for row in development}.isdisjoint({row["participant_id"] for row in lockbox})
    assert {row["piece_id"] for row in development}.isdisjoint({row["piece_id"] for row in lockbox})
    assert first["pre_outcome_split_gate_passed"]


def test_public_samples_and_short_trials_are_excluded():
    rows = synthetic_trials()
    rows[0]["trial_id"] = next(iter(split.PUBLIC_SAMPLE_IDS))
    rows[1]["duration_sec"] = 44.9
    result = split.build_split(rows)
    assert len(result["trials"]) == len(rows) - 2
    assert result["gates"]["sample_trials_excluded"]
