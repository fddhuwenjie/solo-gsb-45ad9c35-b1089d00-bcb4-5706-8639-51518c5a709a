"""家庭应收台账 API 行为测试。"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage import configure_store
from tests.test_vouchers import (
    RECORDS,
    _ch,
    _create_complete as _create_raw_session,
    _issue,
    _rev,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    configure_store(str(tmp_path / "receivables.db"))
    yield
    configure_store(None)


def _post(account_id, items):
    return client.post(
        f"/accounts/{account_id}/receivables",
        json={"vouchers": items},
    )


def _get(account_id):
    return client.get(f"/accounts/{account_id}/receivables")


def _item(sid, voucher_no):
    return {"session_id": sid, "voucher_no": voucher_no}


def _voucher_ids(sid):
    rows = client.get(f"/dispatch/sessions/{sid}/vouchers").json()["vouchers"]
    return {row["voucher_no"]: row["voucher_id"] for row in rows}


def _create_complete():
    sid = _create_raw_session()
    assert _issue(sid, 0).status_code == 200
    return sid


def test_post_first_voucher_creates_account_and_full_receivable():
    sid = _create_complete()
    r = _post("family-a", [_item(sid, 1)])
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["entries_posted"] == 1
    assert data["batch_net_change_yuan"] == 0.1
    assert data["net_receivable_yuan"] == 0.1
    entry = data["entries"][0]
    assert entry["entry_no"] == 1
    assert entry["session_id"] == sid
    assert entry["voucher_id"]
    assert entry["voucher_no"] == 1
    assert entry["revision_no"] == 0
    assert entry["billing_type"] == "full"
    assert entry["billing_amount_yuan"] == 0.1
    assert entry["amount_yuan"] == 0.1
    assert entry["posted_at"].endswith("Z")


def test_supplement_refund_and_zero_adjustment_update_balance():
    sid = _create_complete()
    assert _post("family-a", [_item(sid, 1)]).json()["net_receivable_yuan"] == 0.1

    new1 = dict(RECORDS[1])
    new1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid, 0, [_ch(1, new1)]).status_code == 200
    assert _issue(sid, 1).status_code == 200
    second = _post("family-a", [_item(sid, 2)])
    assert second.json()["batch_net_change_yuan"] == 0.01
    assert second.json()["net_receivable_yuan"] == 0.11

    new1.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    assert _rev(sid, 1, [_ch(1, new1)]).status_code == 200
    assert _issue(sid, 2).status_code == 200
    third = _post("family-a", [_item(sid, 3)])
    assert third.json()["batch_net_change_yuan"] == -0.02
    assert third.json()["net_receivable_yuan"] == 0.09

    new3 = dict(RECORDS[3])
    new3.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.11)
    assert _rev(sid, 2, [_ch(3, new3)]).status_code == 200
    assert _issue(sid, 3).status_code == 200
    fourth = _post("family-a", [_item(sid, 4)])
    assert fourth.json()["batch_net_change_yuan"] == 0.0
    assert fourth.json()["net_receivable_yuan"] == 0.09


def test_same_batch_can_fill_contiguous_prefix_and_preserves_request_order():
    sid1 = _create_complete()
    sid2 = _create_complete()
    r = _post("family-a", [_item(sid1, 1), _item(sid2, 1), _item(sid1, 2)])
    # sid1 voucher 2 does not exist yet, so this first request proves validation order
    # but cannot pass; issue it and resubmit an interleaved contiguous batch.
    assert r.status_code == 422
    new1 = dict(RECORDS[1])
    new1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid1, 0, [_ch(1, new1)]).status_code == 200
    assert _issue(sid1, 1).status_code == 200

    r = _post("family-a", [_item(sid1, 1), _item(sid2, 1), _item(sid1, 2)])
    assert r.status_code == 200, r.text
    entries = r.json()["entries"]
    assert [(e["session_id"], e["voucher_no"]) for e in entries] == [
        (sid1, 1),
        (sid2, 1),
        (sid1, 2),
    ]
    assert [e["entry_no"] for e in entries] == [1, 2, 3]
    assert r.json()["batch_net_change_yuan"] == 0.21


def test_nonexistent_session_or_voucher_rejects_entire_batch():
    sid = _create_complete()
    r = _post("family-a", [_item(sid, 1), _item("missing-session", 1)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_not_found"
    assert _get("family-a").status_code == 404

    r = _post("family-a", [_item(sid, 2)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_not_found"
    assert _get("family-a").status_code == 404


def test_voucher_sequence_gap_rejects_entire_batch_without_partial_entries():
    sid1 = _create_complete()
    sid2 = _create_complete()
    assert _post("family-a", [_item(sid2, 1)]).status_code == 200

    new1 = dict(RECORDS[1])
    new1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid1, 0, [_ch(1, new1)]).status_code == 200
    assert _issue(sid1, 1).status_code == 200

    r = _post("family-a", [_item(sid1, 2), _item(sid2, 1)])
    assert r.status_code == 409
    assert r.json()["error"] == "ledger_duplicate_voucher"
    # sid2 already existed in the account and remains exactly one entry; no sid1 entry.
    ledger = _get("family-a").json()
    assert ledger["entry_count"] == 1
    assert ledger["entries"][0]["session_id"] == sid2
    assert ledger["net_receivable_yuan"] == 0.1

    r = _post("family-a", [_item(sid1, 2)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_sequence_gap"
    assert r.json()["expected_voucher_no"] == 1
    assert _get("family-a").json()["entry_count"] == 1


def test_first_post_must_start_from_voucher_one():
    sid = _create_complete()
    new1 = dict(RECORDS[1])
    new1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid, 0, [_ch(1, new1)]).status_code == 200
    assert _issue(sid, 1).status_code == 200

    r = _post("family-a", [_item(sid, 2)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_sequence_gap"
    assert r.json()["expected_voucher_no"] == 1
    assert _get("family-a").status_code == 404


def test_duplicate_in_same_batch_rejects():
    sid = _create_complete()
    r = _post("family-a", [_item(sid, 1), _item(sid, 1)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_duplicate_voucher"
    assert _get("family-a").status_code == 404


def test_duplicate_existing_voucher_rejects_even_alongside_new_voucher():
    sid = _create_complete()
    assert _post("family-a", [_item(sid, 1)]).status_code == 200
    new1 = dict(RECORDS[1])
    new1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid, 0, [_ch(1, new1)]).status_code == 200
    assert _issue(sid, 1).status_code == 200

    r = _post("family-a", [_item(sid, 1), _item(sid, 2)])
    assert r.status_code == 409
    assert r.json()["error"] == "ledger_duplicate_voucher"
    ledger = _get("family-a").json()
    assert ledger["entry_count"] == 1
    assert ledger["net_receivable_yuan"] == 0.1


def test_cross_account_voucher_rejects_and_keeps_first_account_assignment():
    sid = _create_complete()
    assert _post("family-a", [_item(sid, 1)]).status_code == 200
    r = _post("family-b", [_item(sid, 1)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_cross_account_voucher"
    assert r.json()["account_id"] == "family-a"
    assert _get("family-b").status_code == 404
    assert _get("family-a").json()["entry_count"] == 1


def test_get_ledger_returns_entries_in_account_posting_order():
    sid1 = _create_complete()
    sid2 = _create_complete()
    assert _post("family-a", [_item(sid1, 1), _item(sid2, 1)]).status_code == 200

    new1 = dict(RECORDS[1])
    new1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid1, 0, [_ch(1, new1)]).status_code == 200
    assert _issue(sid1, 1).status_code == 200
    assert _post("family-a", [_item(sid1, 2)]).status_code == 200

    ledger = _get("family-a").json()
    assert ledger["entry_count"] == 3
    assert ledger["net_receivable_yuan"] == 0.21
    assert [(e["entry_no"], e["session_id"], e["voucher_no"]) for e in ledger["entries"]] == [
        (1, sid1, 1),
        (2, sid2, 1),
        (3, sid1, 2),
    ]
    assert ledger["entries"][0]["voucher_id"] == _voucher_ids(sid1)[1]


def test_concurrent_posts_do_not_double_charge():
    sid = _create_complete()
    results = []

    def worker():
        results.append(_post("family-a", [_item(sid, 1)]).status_code)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [200] + [409] * 7
    ledger = _get("family-a").json()
    assert ledger["entry_count"] == 1
    assert ledger["net_receivable_yuan"] == 0.1


def test_ledger_persists_after_store_restart(tmp_path):
    db_path = str(tmp_path / "persistent.db")
    configure_store(db_path)
    sid = _create_complete()
    assert _post("family-a", [_item(sid, 1)]).status_code == 200

    configure_store(db_path)  # 同一路径重建存储，模拟进程重启
    ledger = _get("family-a").json()
    assert ledger["entry_count"] == 1
    assert ledger["net_receivable_yuan"] == 0.1
    assert ledger["entries"][0]["session_id"] == sid


def test_old_voucher_never_changes_but_new_voucher_can_continue_posting():
    sid = _create_complete()
    first_post = _post("family-a", [_item(sid, 1)])
    first_entry = first_post.json()["entries"][0]

    new1 = dict(RECORDS[1])
    new1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid, 0, [_ch(1, new1)]).status_code == 200
    voucher1 = _issue(sid, 1).json()
    second_post = _post("family-a", [_item(sid, 2)]).json()

    ledger = _get("family-a").json()
    assert ledger["entries"][0] == first_entry
    assert ledger["entries"][1]["amount_yuan"] == 0.01
    assert ledger["entries"][1]["voucher_id"] == voucher1["voucher_id"]
    assert ledger["net_receivable_yuan"] == 0.11


def test_request_validation_rejects_empty_and_nonpositive_voucher_no():
    r = client.post("/accounts/family-a/receivables", json={"vouchers": []})
    assert r.status_code == 422
    r = _post("family-a", [_item("sid", 0)])
    assert r.status_code == 422


def test_request_accepts_items_alias_for_voucher_list():
    sid = _create_complete()
    r = client.post(
        "/accounts/family-a/receivables",
        json={"items": [_item(sid, 1)]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["net_receivable_yuan"] == 0.1


def test_balances_are_independent_between_accounts():
    sid1 = _create_complete()
    sid2 = _create_complete()
    assert _post("family-a", [_item(sid1, 1)]).status_code == 200
    assert _post("family-b", [_item(sid2, 1)]).status_code == 200
    assert _get("family-a").json()["net_receivable_yuan"] == 0.1
    assert _get("family-b").json()["net_receivable_yuan"] == 0.1
    assert _get("family-a").json()["entries"][0]["session_id"] == sid1
    assert _get("family-b").json()["entries"][0]["session_id"] == sid2
