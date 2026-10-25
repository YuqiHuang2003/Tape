import re
from dataclasses import asdict

import numpy as np
import pytest

from dsai6707.io import write_json
from scripts import reproduce_tape as subject
from scripts import run_uniform_tape as uniform


def test_released_settings_are_not_silently_changed():
    spec = {
        "shape": [24, 32],
        "mode": "joint",
        "retained_per_layer_reference": 16,
        "search": asdict(uniform.config_for(24, 16)),
    }
    assert subject.search_config(spec).evaluation_budget == 256
    spec["search"]["generations"] = 14
    with pytest.raises(RuntimeError, match="settings differ"):
        subject.search_config(spec)


def test_released_methods_select_their_own_masks(tmp_path):
    anchors = np.zeros((3, 2, 4), dtype=bool)
    anchors[0, :, :2] = True
    anchors[1, :, 1:3] = True
    anchors[2, :, 2:] = True
    np.save(tmp_path / "anchors.npy", anchors, allow_pickle=False)
    np.save(tmp_path / "winner_mask.npy", anchors[2], allow_pickle=False)
    spec = {"shape": [2, 4], "anchor_names": ["reap", "frequency", "hybrid"]}
    np.testing.assert_array_equal(subject.evaluation_mask(tmp_path, spec, "tape"), anchors[2])
    np.testing.assert_array_equal(subject.evaluation_mask(tmp_path, spec, "reap"), anchors[0])
    assert subject.evaluation_mask(tmp_path, spec, "original").all()
    with pytest.raises(ValueError, match="no released mask"):
        subject.evaluation_mask(tmp_path, spec, "unknown")


def test_cpu_verification_rejects_modified_members(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "ARTIFACTS", tmp_path / "artifacts")
    (tmp_path / "code.py").write_text("modified")
    write_json(tmp_path / "RELEASE_MANIFEST.json", {"files": {"code.py": {"sha256": "bad"}}})
    with pytest.raises(RuntimeError, match="modified"):
        subject.verify_release()


def test_cpu_verification_rejects_parent_directory_members(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    write_json(tmp_path / "RELEASE_MANIFEST.json", {"files": {"../outside": {"sha256": "bad"}}})
    with pytest.raises(RuntimeError, match="unsafe"):
        subject.verify_release()


def test_published_text_is_english():
    manifest = subject.ROOT / "RELEASE_MANIFEST.json"
    if not manifest.exists():
        pytest.skip("release language check runs in the generated repository")
    from dsai6707.io import read_json

    for name in read_json(manifest)["files"]:
        path = subject.ROOT / name
        if path.suffix in {".md", ".py", ".tex", ".json", ".yml", ".toml", ".txt"}:
            assert re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", path.read_text()) is None, name


@pytest.mark.parametrize(
    "case",
    [
        "granite_24",
        "granite_16",
        "ernie_48",
        "ernie_32",
        "ernie_48_identity_only",
        "ernie_48_allocation_only",
    ],
)
def test_packaged_cpu_replay_matches_all_recorded_candidates(case):
    if not (subject.ARTIFACTS / "cases.json").exists():
        pytest.skip("release artifact test: run again inside the generated GitHub tree")
    result = subject.replay(case)
    assert result["candidates"] == 256
    assert result["selection_matches"] is True
