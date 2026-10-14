from scripts import p3_skypiano_full_release_gate as gate


def test_five_samples_without_archive_is_not_full_release():
    explorer = {"samples": [{"trialId": str(i)} for i in range(5)], "datasets": []}
    result = gate.assess("<html></html>", explorer, [{"name": "other", "html_url": "https://example.test"}])
    assert not result["full_release_ready"]
    assert not result["resolvable_release_pointer"]


def test_enumerated_trials_and_public_repository_pass_pointer_gate():
    explorer = {"samples": [{"trialId": str(i)} for i in range(6)], "datasets": []}
    repos = [{"name": "SKY-Piano", "html_url": "https://github.com/example/SKY-Piano"}]
    result = gate.assess("", explorer, repos)
    assert result["enumerated_full_trials"]
    assert result["full_release_ready"]


def test_project_archive_is_detected_but_does_not_replace_trial_manifest():
    explorer = {"samples": [{"trialId": str(i)} for i in range(5)], "datasets": []}
    result = gate.assess('<a href="https://zenodo.org/records/1">data</a>', explorer, [])
    assert result["resolvable_release_pointer"]
    assert not result["full_release_ready"]
