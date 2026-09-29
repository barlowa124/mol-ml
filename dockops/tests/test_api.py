from fastapi.testclient import TestClient

from dockops.api import app

client = TestClient(app)


def test_targets_endpoint():
    resp = client.get("/targets")
    assert resp.status_code == 200
    assert "demo_target" in resp.json()


def test_submit_and_retrieve_job():
    resp = client.post(
        "/jobs",
        json={"smiles": "CC(=O)Oc1ccccc1C(=O)O", "target": "demo_target"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["engine"] == "mock"
    assert body["status"] == "ok"
    assert body["provenance"]["engine"] == "mock"

    fetched = client.get(f"/jobs/{body['job_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["job_id"] == body["job_id"]


def test_unknown_target_404():
    resp = client.post("/jobs", json={"smiles": "CCO", "target": "nope"})
    assert resp.status_code == 404


def test_unknown_job_404():
    assert client.get("/jobs/does-not-exist").status_code == 404
