import io

from tests.conftest import PNG, register_and_verify

V = "/api/v1"


# ---- profile / settings
def test_profile_update_and_clear(client, user_auth):
    r = client.patch(f"{V}/users/me", headers=user_auth, json={"full_name": "  New Name ", "farm_name": "Green Acres", "region": "Punjab", "phone": "+91 98765-43210"})
    assert r.status_code == 200
    u = r.json()
    assert u["full_name"] == "New Name" and u["farm_name"] == "Green Acres" and u["phone"] == "+91 98765-43210"
    r = client.patch(f"{V}/users/me", headers=user_auth, json={"farm_name": ""})
    assert r.json()["farm_name"] is None and r.json()["full_name"] == "New Name"


def test_profile_cannot_escalate_role_or_change_email(client, user_auth):
    r = client.patch(f"{V}/users/me", headers=user_auth, json={"role": "admin", "email": "x@y.co", "is_verified": False})
    assert r.status_code == 200
    assert r.json()["role"] == "user" and r.json()["email"] == "farmer@example.com"


def test_profile_validation(client, user_auth):
    assert client.patch(f"{V}/users/me", headers=user_auth, json={"phone": "abc"}).status_code == 422
    assert client.patch(f"{V}/users/me", headers=user_auth, json={"full_name": "A"}).status_code == 422


def test_preferences(client, user_auth):
    r = client.patch(f"{V}/users/me/preferences", headers=user_auth, json={"units": "imperial", "weekly_summary": True})
    assert r.json()["preferences"]["units"] == "imperial" and r.json()["preferences"]["language"] == "en"
    assert client.patch(f"{V}/users/me/preferences", headers=user_auth, json={"units": "bogus"}).status_code == 422


def test_delete_account_requires_password(client, user_auth):
    assert client.post(f"{V}/users/me/delete", headers=user_auth, json={"password": "Wrong12345"}).status_code == 400
    assert client.post(f"{V}/users/me/delete", headers=user_auth, json={"password": "Farmer123"}).status_code == 200
    assert client.get(f"{V}/users/me", headers=user_auth).status_code == 401
    assert client.post(f"{V}/auth/login", json={"email": "farmer@example.com", "password": "Farmer123"}).status_code == 401


# ---- admin authorization
def test_admin_endpoints_forbidden_for_regular_user(client, user_auth):
    for method, path in [("get", "/admin/stats"), ("get", "/admin/users"), ("get", "/admin/settings"), ("get", "/admin/audit-log")]:
        r = getattr(client, method)(f"{V}{path}", headers=user_auth)
        assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden", path
    assert client.put(f"{V}/admin/settings", headers=user_auth, json={"values": {"maintenance_mode": True}}).status_code == 403
    assert client.patch(f"{V}/admin/users/00000000-0000-0000-0000-000000000000", headers=user_auth, json={"role": "admin"}).status_code == 403


def test_admin_stats_and_user_list(client, admin_auth):
    register_and_verify(client, email="a@example.com")
    register_and_verify(client, email="b@example.com", name="Bee Keeper")
    s = client.get(f"{V}/admin/stats", headers=admin_auth).json()
    assert s["users_total"] == 3 and s["admins"] == 1 and s["users_verified"] == 3
    r = client.get(f"{V}/admin/users?q=bee", headers=admin_auth).json()
    assert r["total"] == 1 and r["items"][0]["email"] == "b@example.com"
    assert client.get(f"{V}/admin/users?role=admin", headers=admin_auth).json()["total"] == 1
    assert client.get(f"{V}/admin/users?page=0", headers=admin_auth).status_code == 422
    assert client.get(f"{V}/admin/users?page_size=1000", headers=admin_auth).status_code == 422


def test_admin_can_promote_and_deactivate(client, admin_auth):
    d = register_and_verify(client, email="t@example.com")
    uid, h = d["user"]["id"], {"Authorization": f"Bearer {d['access_token']}"}
    assert client.get(f"{V}/admin/stats", headers=h).status_code == 403
    assert client.patch(f"{V}/admin/users/{uid}", headers=admin_auth, json={"role": "admin"}).json()["role"] == "admin"
    # role change revokes sessions: must re-login to get an admin token
    assert client.get(f"{V}/admin/stats", headers=h).status_code == 401
    client.patch(f"{V}/admin/users/{uid}", headers=admin_auth, json={"is_active": False})
    r = client.post(f"{V}/auth/login", json={"email": "t@example.com", "password": "Farmer123"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "account_disabled"
    log = client.get(f"{V}/admin/audit-log", headers=admin_auth).json()
    assert [e["action"] for e in log].count("user.update") == 2


def test_admin_guards(client, admin_auth):
    me = client.get(f"{V}/users/me", headers=admin_auth).json()
    r = client.patch(f"{V}/admin/users/{me['id']}", headers=admin_auth, json={"role": "user"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "self_lockout"
    r = client.patch(f"{V}/admin/users/{me['id']}", headers=admin_auth, json={"is_active": False})
    assert r.status_code == 400
    assert client.patch(f"{V}/admin/users/00000000-0000-0000-0000-000000000000", headers=admin_auth, json={"role": "user"}).status_code == 404
    assert client.patch(f"{V}/admin/users/not-a-uuid", headers=admin_auth, json={"role": "user"}).status_code == 422
    assert client.patch(f"{V}/admin/users/{me['id']}", headers=admin_auth, json={"role": "superuser"}).status_code == 422


def test_admin_settings_roundtrip_and_validation(client, admin_auth):
    r = client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {"announcement": "Harvest season!", "max_upload_mb": 5}})
    assert r.status_code == 200 and r.json()["announcement"] == "Harvest season!"
    assert client.get(f"{V}/config/public").json()["max_upload_mb"] == 5
    bad = client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {"max_upload_mb": 999, "nope": 1, "maintenance_mode": "yes"}})
    assert bad.status_code == 422 and set(bad.json()["error"]["fields"]) == {"max_upload_mb", "nope", "maintenance_mode"}
    assert client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {}}).status_code == 422


# ---- analyses
def upload(client, headers, data=PNG, name="leaf.png", **form):
    return client.post(f"{V}/analyses", headers=headers, files={"file": (name, io.BytesIO(data), "image/png")}, data=form)


def test_upload_list_get_image_delete(client, user_auth):
    r = upload(client, user_auth, crop_type="Tomato", source="camera", notes="yellow spots")
    assert r.status_code == 201, r.text
    a = r.json()
    assert a["status"] == "uploaded" and a["result"] is None and a["content_type"] == "image/png" and a["source"] == "camera"
    assert client.get(f"{V}/analyses", headers=user_auth).json()["total"] == 1
    assert client.get(f"{V}/analyses?q=tomato", headers=user_auth).json()["total"] == 1
    assert client.get(f"{V}/analyses?q=potato", headers=user_auth).json()["total"] == 0
    assert client.get(f"{V}/analyses?status=completed", headers=user_auth).json()["total"] == 0
    assert client.get(f"{V}/analyses?status=bogus", headers=user_auth).status_code == 422
    img = client.get(f"{V}/analyses/{a['id']}/image", headers=user_auth)
    assert img.status_code == 200 and img.content == PNG and img.headers["content-type"] == "image/png"
    assert client.get(f"{V}/analyses/summary", headers=user_auth).json()["total"] == 1
    assert client.delete(f"{V}/analyses/{a['id']}", headers=user_auth).status_code == 200
    assert client.get(f"{V}/analyses/{a['id']}", headers=user_auth).status_code == 404


def test_analyses_are_private_between_users(client, user_auth):
    a = upload(client, user_auth).json()
    other = register_and_verify(client, email="other@example.com")
    oh = {"Authorization": f"Bearer {other['access_token']}"}
    for path in (f"/analyses/{a['id']}", f"/analyses/{a['id']}/image"):
        assert client.get(f"{V}{path}", headers=oh).status_code == 404
    assert client.delete(f"{V}/analyses/{a['id']}", headers=oh).status_code == 404
    assert client.get(f"{V}/analyses", headers=oh).json()["total"] == 0


def test_upload_rejects_non_images_even_with_image_content_type(client, user_auth):
    r = upload(client, user_auth, data=b"<html>not an image</html>", name="evil.png")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media"
    r = upload(client, user_auth, data=b"GIF89a....", name="x.gif")
    assert r.status_code == 415


def test_upload_empty_and_missing_file(client, user_auth):
    assert upload(client, user_auth, data=b"").status_code == 422
    assert client.post(f"{V}/analyses", headers=user_auth).status_code == 422


def test_upload_too_large_cleans_up(client, user_auth, admin_auth, settings):
    client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {"max_upload_mb": 1}})
    big = PNG + b"\0" * (1024 * 1024 + 10)
    r = upload(client, user_auth, data=big)
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"
    leftovers = [p for p in settings.upload_path.rglob("*") if p.is_file()]
    assert leftovers == []


def test_upload_validation_and_maintenance(client, user_auth, admin_auth):
    assert upload(client, user_auth, crop_type="Unobtainium").status_code == 422
    assert upload(client, user_auth, source="drone").status_code == 422
    client.put(f"{V}/admin/settings", headers=admin_auth, json={"values": {"maintenance_mode": True}})
    r = upload(client, user_auth)
    assert r.status_code == 503 and r.json()["error"]["code"] == "maintenance"


def test_upload_requires_auth(client):
    assert client.post(f"{V}/analyses", files={"file": ("a.png", PNG, "image/png")}).status_code == 401


def test_path_traversal_filename_is_harmless(client, user_auth, settings):
    r = upload(client, user_auth, name="../../etc/passwd.png")
    assert r.status_code == 201
    stored = [p for p in settings.upload_path.rglob("*") if p.is_file()]
    assert all(settings.upload_path in p.parents for p in stored)
