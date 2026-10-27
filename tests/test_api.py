import io

from fastapi.testclient import TestClient
from PIL import Image

from conftest import wait
from edit_studio.app import create_app

H = {"X-Edit-Studio": "1"}


def client(services):
    return TestClient(create_app(services), base_url="http://127.0.0.1:8765")


def png_bytes(color=(200, 50, 50), size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_security_guards(services):
    c = client(services)
    assert c.get("/api/system").status_code == 200
    assert c.post("/api/jobs", json={"task": "models.scan"}).status_code == 403  # no custom header
    evil = TestClient(create_app(services), base_url="http://evil.example")
    assert evil.get("/api/system").status_code == 400  # DNS rebinding
    assert c.get("/files/../settings.json").status_code == 404
    assert c.get("/files/settings.json").status_code == 404  # only outputs/inputs are served


def test_upload_gallery_and_files(services):
    c = client(services)
    r = c.post("/api/uploads", files={"file": ("photo.png", png_bytes(), "image/png")}, headers=H)
    assert r.status_code == 200, r.text
    item = r.json()
    assert item["id"].startswith("u-") and item["width"] == 64 and item["height"] == 48
    assert c.get(item["url"]).status_code == 200
    # same file again -> same item
    assert c.post("/api/uploads", files={"file": ("again.png", png_bytes(), "image/png")}, headers=H).json()["id"] == item["id"]
    # hidden uploads (masks) stay out of the library
    hidden = c.post("/api/uploads", files={"file": ("mask.png", png_bytes((255, 255, 255)), "image/png")},
                    data={"hidden": "true"}, headers=H).json()
    ups = c.get("/api/gallery", params={"uploads": True}).json()
    ids = [i["id"] for i in ups["items"]]
    assert item["id"] in ids and hidden["id"] not in ids
    bad = c.post("/api/uploads", files={"file": ("x.exe", b"MZ", "application/octet-stream")}, headers=H)
    assert bad.status_code == 400


def test_settings_validation(services):
    c = client(services)
    r = c.put("/api/settings", json={"memory_mode": "split", "model_folders": ["C:/ComfyUI"]}, headers=H)
    assert r.status_code == 200 and r.json()["memory_mode"] == "split"
    assert c.put("/api/settings", json={"memory_mode": "turbo"}, headers=H).status_code == 400
    r = c.put("/api/settings", json={"hf_token": "hf_secret"}, headers=H).json()
    assert r["hf_token_set"] is True and "hf_token" not in r


def test_models_listing_and_invalid_job(services):
    c = client(services)
    m = c.get("/api/models").json()
    ids = {x["id"] for x in m["catalog"]}
    assert {"z-image-turbo", "wan2.2-ti2v-5b", "qwen-image-edit-2511"} <= ids
    assert all(x["plan"] for x in m["catalog"])
    r = c.post("/api/jobs", json={"task": "image.generate", "params": {"model": "z-image-turbo", "prompt": "x", "count": 50}}, headers=H)
    assert r.status_code == 422
    assert c.post("/api/jobs", json={"task": "nope", "params": {}}, headers=H).status_code == 400


def test_job_with_missing_model_waits_for_download(services, monkeypatch):
    c = client(services)

    def fake_download(repo, variant, progress, cancel, skip=()):
        raise ConnectionError("offline")

    monkeypatch.setattr(services.hub, "download", fake_download)
    r = c.post("/api/jobs", json={"task": "image.generate", "params": {"model": "z-image-turbo", "prompt": "a cat"}}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["downloads"] and body["job"]["depends_on"] == body["downloads"][0]["id"]
    job = wait(services.jobs.get(body["job"]["id"]))
    dl = services.jobs.get(body["downloads"][0]["id"])
    assert dl.status == "failed" and dl.error_key == "network"
    assert job.status == "failed" and job.error_key == "dependency_failed"
