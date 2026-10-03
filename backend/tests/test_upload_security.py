"""Upload hardening: type sniffing, corrupt/oversized/bomb images, size limits before buffering, safe filenames."""
import io
import time

import pytest
from PIL import Image

from app.core.config import get_settings
from tests.conftest import register_and_verify

V = "/api/v1"


def img(fmt="PNG", size=(120, 90), color=(40, 140, 60), **save):
    b = io.BytesIO(); Image.new("RGB", size, color).save(b, fmt, **save); return b.getvalue()


def post(client, headers, data, name="leaf.png", ctype="image/png", **form):
    return client.post(f"{V}/analyses", headers=headers, files={"file": (name, io.BytesIO(data), ctype)}, data=form)


def stored_files(settings):
    return [p for p in settings.upload_path.rglob("*") if p.is_file()]


# ------------------------------------------------------------------ allowed types (decided by content, not by the label)
@pytest.mark.parametrize("fmt,mime", [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")])
def test_real_jpeg_png_webp_are_accepted_whatever_the_client_calls_them(client, user_auth, fmt, mime):
    r = post(client, user_auth, img(fmt), name="photo.bin", ctype="application/octet-stream")
    assert r.status_code == 201 and r.json()["content_type"] == mime


@pytest.mark.parametrize("name,data", [
    ("a.gif", b"GIF89a" + b"\x00" * 200), ("a.bmp", b"BM" + b"\x00" * 200), ("a.svg", b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>"),
    ("a.html", b"<html><script>alert(1)</script></html>"), ("a.pdf", b"%PDF-1.4 " + b"\x00" * 200), ("a.exe", b"MZ" + b"\x00" * 200),
    ("a.php", b"<?php system($_GET['c']); ?>"), ("a.png", b"#!/bin/sh\nrm -rf /\n"), ("a.tiff", b"II*\x00" + b"\x00" * 200),
])
def test_non_images_are_rejected_even_when_named_like_images(client, user_auth, settings, name, data):
    r = post(client, user_auth, data, name=name, ctype="image/png")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media" and stored_files(settings) == []


def test_image_with_a_script_appended_is_inert(client, user_auth):
    polyglot = img("PNG") + b"<?php system($_GET['c']); ?><script>alert(1)</script>"
    r = post(client, user_auth, polyglot, name="shell.php.png")
    assert r.status_code == 201
    out = client.get(f"{V}/analyses/{r.json()['id']}/image", headers=user_auth)
    assert out.headers["content-type"] == "image/png" and out.headers["x-content-type-options"] == "nosniff"      # never served as HTML/script


# ------------------------------------------------------------------ corrupt / tiny / oversized
@pytest.mark.parametrize("label,data", [
    ("truncated jpeg", img("JPEG", size=(800, 800))[:1500]), ("truncated png", img("PNG", size=(500, 500))[:300]),
    ("garbage after jpeg magic", b"\xff\xd8\xff" + b"garbage" * 80), ("garbage after png magic", b"\x89PNG\r\n\x1a\n" + b"garbage" * 80),
    ("fake webp", b"RIFF\x24\x00\x00\x00WEBP" + b"junk" * 40),
])
def test_corrupt_images_are_rejected_and_nothing_is_stored(client, user_auth, settings, label, data):
    r = post(client, user_auth, data, ctype="image/jpeg")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image", label
    assert stored_files(settings) == [] and client.get(f"{V}/analyses", headers=user_auth).json()["total"] == 0


@pytest.mark.parametrize("size,ok", [((47, 300), False), ((300, 47), False), ((20, 20), False), ((48, 48), True), ((64, 64), True)])
def test_minimum_dimensions(client, user_auth, size, ok):
    r = post(client, user_auth, img(size=size))
    assert (r.status_code == 201) if ok else (r.status_code == 422 and r.json()["error"]["code"] == "image_too_small")


def test_normal_phone_photo_sizes_are_accepted(client, user_auth):
    for size in ((4000, 3000), (3024, 4032), (1920, 1080)):                                    # 12 MP, 12 MP portrait, full HD
        assert post(client, user_auth, img("JPEG", size=size, quality=80), name="p.jpg", ctype="image/jpeg").status_code == 201, size


def test_decompression_bomb_is_refused_from_the_header_without_decoding(client, user_auth, settings):
    bomb = io.BytesIO(); Image.new("1", (9000, 9000)).save(bomb, "PNG")                       # ~81 MP "claimed", tiny file
    assert len(bomb.getvalue()) < 200_000 and 9000 * 9000 > settings.ml_max_pixels
    t = time.perf_counter()
    r = post(client, user_auth, bomb.getvalue())
    assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large" and stored_files(settings) == []
    assert time.perf_counter() - t < 3


def test_oversized_file_is_rejected_and_cleaned_up(client, user_auth, admin_auth, settings):
    client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {"max_upload_mb": 1}})
    r = post(client, user_auth, img("PNG", size=(120, 90)) + b"\0" * (1024 * 1024 + 10))
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large" and stored_files(settings) == []


def test_admin_cannot_raise_the_limit_above_the_server_cap(client, admin_auth, settings, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 3)
    client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {"max_upload_mb": 20}})
    from app.db.session import SessionLocal
    from app.services import analysis_service, settings_service
    with SessionLocal() as db:
        assert settings_service.effective_max_upload_bytes(db) == 3 * 1024 * 1024


# ------------------------------------------------------------------ rejected BEFORE buffering the whole body
def test_declared_huge_body_gets_413_immediately(client, user_auth):
    t = time.perf_counter()
    r = client.post(f"{V}/analyses", headers={**user_auth, "Content-Length": str(2 * 1024 * 1024 * 1024)}, content=b"x")
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large" and time.perf_counter() - t < 2


def test_chunked_body_without_content_length_is_cut_off(client, user_auth, settings):
    def chunks():
        yield b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\nContent-Type: image/png\r\n\r\n'      # a valid multipart start
        for _ in range(14):
            yield b"\0" * (1024 * 1024)                                                          # 14 MB > (10 MB limit + 1 MB framing)
    r = client.post(f"{V}/analyses", headers={**user_auth, "Content-Type": "multipart/form-data; boundary=x"}, content=chunks())
    assert r.status_code == 413 and stored_files(settings) == []


def test_uploads_within_the_limit_still_work_through_the_middleware(client, user_auth):
    assert post(client, user_auth, img("JPEG", size=(2000, 1500), quality=90), ctype="image/jpeg").status_code == 201


# ------------------------------------------------------------------ names and paths
@pytest.mark.parametrize("evil,expected", [
    ("../../etc/passwd.png", "passwd.png"), ("..\\..\\windows\\system32\\cmd.png", "cmd.png"), ("/abs/path/leaf.png", "leaf.png"),
    ("fake‮gnp.exe", "fakegnp.exe"), ("   spaced   name.png ", "spaced name.png"),
])
def test_filenames_are_sanitised_for_display_and_never_used_on_disk(client, user_auth, settings, evil, expected):
    r = post(client, user_auth, img(), name=evil)
    assert r.status_code == 201 and r.json()["original_filename"] == expected
    for p in stored_files(settings):
        assert settings.upload_path in p.parents and p.suffix == ".png" and len(p.stem) == 32 and all(c in "0123456789abcdef" for c in p.stem)


def test_control_characters_are_stripped_from_stored_names():
    from app.services.analysis_service import safe_filename
    assert safe_filename("le\x00af\x07\x1b.png") == "leaf.png" and safe_filename("\x00\x01") is None and safe_filename(None) is None and safe_filename("a/b\\c.png") == "c.png"


def test_very_long_and_unicode_filenames(client, user_auth):
    assert len(post(client, user_auth, img(), name="x" * 1000 + ".png").json()["original_filename"]) <= 255
    assert post(client, user_auth, img(), name="टमाटर-पत्ता-🍅.png").json()["original_filename"] == "टमाटर-पत्ता-🍅.png"


def test_files_are_only_reachable_through_the_authenticated_api(client, user_auth, settings):
    a = post(client, user_auth, img()).json()
    rel = stored_files(settings)[0].relative_to(settings.upload_path)
    for path in (f"/uploads/{rel}", f"/static/{rel}", f"/{rel}", f"{V}/uploads/{rel}"):
        assert client.get(path).status_code in (404, 405)
    assert client.get(f"{V}/analyses/{a['id']}/image").status_code == 401
    other = register_and_verify(client, email="o@example.com")
    assert client.get(f"{V}/analyses/{a['id']}/image", headers={"Authorization": f"Bearer {other['access_token']}"}).status_code == 404


def test_upload_requires_authentication_and_a_file(client, user_auth):
    assert client.post(f"{V}/analyses", files={"file": ("a.png", img(), "image/png")}).status_code == 401
    assert client.post(f"{V}/analyses", headers=user_auth).status_code == 422
