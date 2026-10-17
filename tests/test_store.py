import pytest

import db


@pytest.fixture(params=["json", "sql"])
def store(request, tmp_path):
    if request.param == "json":
        return db.JsonFileStore(tmp_path, db.LEGACY_FILENAMES)
    return db.build_store(f"sqlite:///{tmp_path}/s.db")  # QP_DB_CREATE_ALL=1 via conftest


def test_roundtrip(store):
    assert store.load("pro_tokens") == {}
    store.save("pro_tokens", {"PRO-1": 123.0, "PRO-2": 456.0})
    assert store.load("pro_tokens") == {"PRO-1": 123.0, "PRO-2": 456.0}


def test_save_replaces_namespace(store):
    store.save("tv_sessions", {"a": {"alerts": []}})
    store.save("tv_sessions", {"b": {"alerts": [{"x": 1}]}})
    assert list(store.load("tv_sessions")) == ["b"]


def test_namespaces_are_isolated(store):
    store.save("report_tokens", {"RPT-1": True})
    assert store.load("pro_tokens") == {}


def test_json_store_reads_legacy_list_format(tmp_path):
    (tmp_path / "qp_report_tokens.json").write_text('["RPT-a", "LITE-b"]')
    s = db.JsonFileStore(tmp_path, db.LEGACY_FILENAMES)
    assert set(s.load("report_tokens")) == {"RPT-a", "LITE-b"}


def test_json_store_tolerates_corrupt_file(tmp_path):
    (tmp_path / "qp_pro_tokens.json").write_text("{not json")
    assert db.JsonFileStore(tmp_path, db.LEGACY_FILENAMES).load("pro_tokens") == {}


def test_build_store_defaults_to_json(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("QP_STATE_DIR", str(tmp_path))
    assert isinstance(db.build_store(), db.JsonFileStore)


def test_token_persistence_survives_reload(monkeypatch):
    from services import tokens

    tokens.REPORT_ACCESS_TOKENS.add("RPT-persist")
    tokens._save_tokens()
    tokens.REPORT_ACCESS_TOKENS.clear()
    tokens._load_tokens()
    assert "RPT-persist" in tokens.REPORT_ACCESS_TOKENS
