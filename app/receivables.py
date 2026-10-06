"""家庭应收台账：把多张已冻结结算凭证按账户汇总入账。

入账只读取 session_settlement_vouchers 中落盘且不可变的计费栏：
- full / supplement：按计费栏金额增加净应收；
- refund：按计费栏金额减少净应收（净变动为负数）；
- no_adjustment：金额与净变动均为 0。

一个会话必须从首张凭证（voucher_no=1）起连续记入同一个家庭账户；一次请求
可以按列表次序补齐前序，例如已记 1 后，同批提交 [2, 3]。凭证不存在、出单
序号跳跃、凭证已记入其他账户或重复提交（批内或与已记明细重复）均整批拒绝，
余额和明细不变。

「校验所有凭证与已记明细 — 写入本批明细 — 更新余额」在同一个 SQLite
BEGIN IMMEDIATE 事务和进程锁内完成；receivable_ledger_entries 的
(session_id, voucher_no) 唯一约束也会兜底，保证并发入账不会重复计费。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from app.settlement import _round6
from app.storage import SessionStore, get_store


class ReceivableLedgerError(Exception):
    """应收台账操作失败：code 决定错误体，status_code 决定 HTTP 状态码。"""

    def __init__(
        self,
        code: str,
        reason: str,
        status_code: int = 422,
        extra: Optional[dict] = None,
    ):
        super().__init__(reason)
        self.code = code
        self.status_code = status_code
        self.extra = extra or {}


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _signed_amount(billing_type: str, amount: float) -> float:
    if billing_type in ("full", "supplement"):
        return _round6(amount)
    if billing_type == "refund":
        return _round6(-amount)
    return 0.0


def _entry_from_row(row) -> dict:
    billing_amount = float(row["billing_amount_yuan"])
    return {
        "entry_no": int(row["entry_no"]),
        "session_id": row["session_id"],
        "voucher_id": row["voucher_id"],
        "voucher_no": int(row["voucher_no"]),
        "revision_no": int(row["revision_no"]),
        "billing_type": row["billing_type"],
        "billing_amount_yuan": billing_amount,
        "amount_yuan": float(row["amount_yuan"]),
        "posted_at": row["posted_at"],
    }


def post_receivables_batch(
    account_id: str,
    items: List[dict],
    store: Optional[SessionStore] = None,
) -> dict:
    """校验并原子写入一批凭证明细，返回本批新增明细、净变动和最新余额。"""
    store = store or get_store()

    # 批内重复立即拒绝；随后在事务中再与持久化明细核对。
    seen_in_request = set()
    for position, item in enumerate(items):
        key = (item["session_id"], int(item["voucher_no"]))
        if key in seen_in_request:
            raise ReceivableLedgerError(
                "ledger_duplicate_voucher",
                f"本批第 {position + 1} 项重复提交会话 {key[0]} 的出单序号 "
                f"{key[1]}；整批拒绝，不入账任何明细",
                extra={"session_id": key[0], "voucher_no": key[1]},
            )
        seen_in_request.add(key)

    posted_at = _utc_now_iso()
    with store.write_transaction() as conn:
        account_row = store.get_receivable_account(conn, account_id)
        if account_row is None:
            current_balance = 0.0
            created_at = posted_at
        else:
            current_balance = float(account_row["net_receivable_yuan"])
            created_at = account_row["created_at"]

        # session_id -> 本次涉及的 voucher_no，保留请求中的入账次序。
        requested_by_session: dict[str, list[int]] = {}
        validated_rows = []

        for position, item in enumerate(items):
            session_id = item["session_id"]
            voucher_no = int(item["voucher_no"])
            requested_by_session.setdefault(session_id, []).append(voucher_no)

            voucher_row = store.get_voucher_by_no(conn, session_id, voucher_no)
            if voucher_row is None:
                raise ReceivableLedgerError(
                    "ledger_voucher_not_found",
                    f"本批第 {position + 1} 项：会话 {session_id} 不存在出单序号 "
                    f"{voucher_no} 的已出凭证；整批拒绝，不入账任何明细",
                    extra={"session_id": session_id, "voucher_no": voucher_no},
                )

            existing = store.get_receivable_entry_any_account(
                conn, session_id, voucher_no
            )
            if existing is not None:
                existing_account = existing["account_id"]
                if existing_account == account_id:
                    code = "ledger_duplicate_voucher"
                    detail = "该凭证已记入本账户，不能重复计费"
                else:
                    code = "ledger_cross_account_voucher"
                    detail = f"该凭证已记入账户 {existing_account}，一个会话只能连续归属同一账户"
                raise ReceivableLedgerError(
                    code,
                    f"本批第 {position + 1} 项：会话 {session_id} 的出单序号 "
                    f"{voucher_no} {detail}；整批拒绝，余额与已记明细不变",
                    409 if code == "ledger_duplicate_voucher" else 422,
                    extra={
                        "session_id": session_id,
                        "voucher_no": voucher_no,
                        "account_id": existing_account,
                    },
                )

            validated_rows.append((item, voucher_row))

        # 对每个会话检查：已记最大序号 + 本批序号必须组成从 1 起、无跳号的区间。
        for session_id, requested_numbers in requested_by_session.items():
            # 已记明细本身在唯一约束和历史批次校验下应连续；仍查询最大序号作为前序边界。
            last_row = conn.execute(
                "SELECT MAX(voucher_no) AS max_no FROM receivable_ledger_entries"
                " WHERE account_id = ? AND session_id = ?",
                (account_id, session_id),
            ).fetchone()
            max_existing = last_row["max_no"]
            expected_start = 1 if max_existing is None else int(max_existing) + 1
            requested_set = set(requested_numbers)
            requested_min = min(requested_numbers)
            requested_max = max(requested_numbers)
            expected_numbers = set(range(expected_start, requested_max + 1))

            if requested_min < expected_start:
                # 正常情况下小于前序边界已由重复/跨账户检查拦截；保留不变量错误。
                raise ReceivableLedgerError(
                    "ledger_voucher_sequence_gap",
                    f"会话 {session_id} 已记至出单序号 {expected_start - 1}，"
                    f"本批却从 {requested_min} 开始；整批拒绝",
                    extra={
                        "session_id": session_id,
                        "expected_voucher_no": expected_start,
                        "requested_voucher_no": requested_min,
                    },
                )
            if requested_numbers != list(
                range(requested_numbers[0], requested_numbers[-1] + 1)
            ):
                raise ReceivableLedgerError(
                    "ledger_voucher_out_of_order",
                    f"会话 {session_id} 在本批中的出单序号必须按序递增且不重复，"
                    f"收到 {requested_numbers}；整批拒绝，余额与已记明细不变",
                    extra={
                        "session_id": session_id,
                        "requested_voucher_nos": requested_numbers,
                    },
                )
            missing = sorted(expected_numbers - requested_set)
            if missing:
                raise ReceivableLedgerError(
                    "ledger_voucher_sequence_gap",
                    f"会话 {session_id} 的凭证必须从首张起连续入账：已记至 "
                    f"{expected_start - 1}，本批最高为 {requested_max}，缺少出单序号 "
                    f"{missing}；整批拒绝，余额与已记明细不变",
                    extra={
                        "session_id": session_id,
                        "expected_voucher_no": missing[0],
                        "missing_voucher_nos": missing,
                    },
                )

        next_entry_no = store.count_receivable_entries(conn, account_id) + 1
        batch_net_change = 0.0
        entries: List[dict] = []

        for item, voucher_row in validated_rows:
            billing_type = voucher_row["billing_type"]
            billing_amount = float(voucher_row["amount_yuan"])
            signed_amount = _signed_amount(billing_type, billing_amount)
            entry_no = next_entry_no
            store.insert_receivable_entry(
                conn,
                entry_no=entry_no,
                account_id=account_id,
                session_id=item["session_id"],
                voucher_id=voucher_row["voucher_id"],
                voucher_no=int(voucher_row["voucher_no"]),
                revision_no=int(voucher_row["revision_no"]),
                billing_type=billing_type,
                billing_amount_yuan=_round6(billing_amount),
                amount_yuan=signed_amount,
                posted_at=posted_at,
            )
            inserted = store.get_receivable_entry(
                conn, account_id, item["session_id"], int(voucher_row["voucher_no"])
            )
            entries.append(_entry_from_row(inserted))
            batch_net_change += signed_amount
            next_entry_no += 1

        new_balance = _round6(current_balance + batch_net_change)
        store.upsert_receivable_account(
            conn,
            account_id=account_id,
            net_receivable_yuan=new_balance,
            created_at=created_at,
            updated_at=posted_at,
        )

        return {
            "account_id": account_id,
            "entries_posted": len(entries),
            "batch_net_change_yuan": _round6(batch_net_change),
            "net_receivable_yuan": new_balance,
            "entries": entries,
        }


def get_receivable_ledger(
    account_id: str,
    store: Optional[SessionStore] = None,
) -> dict:
    """按账户内入账次序查询全部明细与累计净应收。"""
    store = store or get_store()
    account_row = store.get_receivable_account_row(account_id)
    if account_row is None:
        raise ReceivableLedgerError(
            "account_not_found",
            f"账户 {account_id} 暂无入账记录（首次有效批次会自动建档）",
            404,
        )
    rows = store.list_receivable_entry_rows(account_id)
    entries = [_entry_from_row(row) for row in rows]
    return {
        "account_id": account_id,
        "net_receivable_yuan": float(account_row["net_receivable_yuan"]),
        "entry_count": len(entries),
        "entries": entries,
    }
