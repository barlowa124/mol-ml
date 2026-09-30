import hashlib
import json

from dti_fusion.provenance import git_commit, sha256_file, write_manifest


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"payload")
    assert sha256_file(p) == hashlib.sha256(b"payload").hexdigest()


def test_write_manifest_records_inputs_and_config(tmp_path):
    inp = tmp_path / "data.csv"
    inp.write_text("a,b\n1,2\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("k: v\n")
    out = tmp_path / "sub" / "manifest.json"

    m = write_manifest(out, inputs=[inp], config_path=str(cfg))

    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == m
    assert m["config_sha256"] == hashlib.sha256(b"k: v\n").hexdigest()
    assert m["inputs"][0]["sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert set(m["versions"]) == {"numpy", "pandas", "torch"}
    assert "created_utc" in m and "git" in m


def test_write_manifest_skips_missing_inputs_and_config(tmp_path):
    out = tmp_path / "manifest.json"
    m = write_manifest(
        out,
        inputs=[tmp_path / "does_not_exist.parquet"],
        config_path=str(tmp_path / "no_config.yaml"),
    )
    assert m["inputs"] == []
    assert m["config_sha256"] is None


def test_git_commit_returns_string():
    # inside the repo: "<sha>" or "<sha>+dirty"; outside: "unknown"
    assert isinstance(git_commit(), str) and git_commit()
