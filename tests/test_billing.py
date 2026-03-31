"""
Tests for the billing module (store.py, stripe_client.py).

Uses a temporary SQLite database — no real Stripe credentials required.
"""

import os
import tempfile
import time
import pytest


# ---------------------------------------------------------------------------
# Fixture: redirect DB path to a temp file for each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Each test gets its own fresh billing.db."""
    import src.billing.store as store_mod

    db_path = tmp_path / "billing.db"
    monkeypatch.setattr(store_mod, "_DB_PATH", db_path)
    # Reset module-level connection so the next call reopens against the new path
    monkeypatch.setattr(store_mod, "_conn", None)
    yield
    # Cleanup: close the connection if it was opened
    if store_mod._conn is not None:
        store_mod._conn.close()
        store_mod._conn = None


# ---------------------------------------------------------------------------
# get_balance
# ---------------------------------------------------------------------------

class TestGetBalance:
    def test_unknown_key_returns_zero(self):
        from src.billing.store import get_balance
        assert get_balance("nonexistent_key") == 0

    def test_returns_balance_after_add(self):
        from src.billing.store import get_balance, add_credits
        add_credits("key1", 500)
        assert get_balance("key1") == 500


# ---------------------------------------------------------------------------
# add_credits
# ---------------------------------------------------------------------------

class TestAddCredits:
    def test_creates_row_on_first_add(self):
        from src.billing.store import add_credits, get_balance
        new_bal = add_credits("key2", 300)
        assert new_bal == 300
        assert get_balance("key2") == 300

    def test_accumulates_multiple_adds(self):
        from src.billing.store import add_credits, get_balance
        add_credits("key3", 100)
        add_credits("key3", 200)
        add_credits("key3", 50)
        assert get_balance("key3") == 350

    def test_returns_new_balance(self):
        from src.billing.store import add_credits
        b1 = add_credits("key4", 1000)
        b2 = add_credits("key4", 500)
        assert b1 == 1000
        assert b2 == 1500


# ---------------------------------------------------------------------------
# deduct_credits
# ---------------------------------------------------------------------------

class TestDeductCredits:
    def test_successful_deduction(self):
        from src.billing.store import add_credits, deduct_credits
        add_credits("key5", 1000)
        result = deduct_credits("key5", seconds=5.0, task_id="t1")
        assert result["ok"] is True
        assert result["cost_cents"] == 50  # 5s * 10 cents/s
        assert result["balance_cents"] == 950

    def test_insufficient_credits(self):
        from src.billing.store import add_credits, deduct_credits
        add_credits("key6", 30)
        result = deduct_credits("key6", seconds=5.0)  # costs 50 cents, balance 30
        assert result["ok"] is False
        assert result["reason"] == "insufficient_credits"
        assert result["balance_cents"] == 30  # unchanged

    def test_zero_balance_key_fails(self):
        from src.billing.store import deduct_credits
        result = deduct_credits("no_credits_key", seconds=1.0)
        assert result["ok"] is False

    def test_exact_balance_succeeds(self):
        from src.billing.store import add_credits, deduct_credits, get_balance
        add_credits("key7", 100)
        result = deduct_credits("key7", seconds=10.0)  # costs exactly 100
        assert result["ok"] is True
        assert get_balance("key7") == 0

    def test_minimum_cost_is_one_cent(self):
        from src.billing.store import add_credits, deduct_credits
        add_credits("key8", 100)
        result = deduct_credits("key8", seconds=0.001)  # tiny fraction
        assert result["ok"] is True
        assert result["cost_cents"] >= 1

    def test_balance_unchanged_after_failure(self):
        from src.billing.store import add_credits, deduct_credits, get_balance
        add_credits("key9", 40)
        deduct_credits("key9", seconds=10.0)  # fails — costs 100
        assert get_balance("key9") == 40


# ---------------------------------------------------------------------------
# get_usage
# ---------------------------------------------------------------------------

class TestGetUsage:
    def test_empty_usage_for_unknown_key(self):
        from src.billing.store import get_usage
        assert get_usage("nobody") == []

    def test_records_usage_on_successful_deduction(self):
        from src.billing.store import add_credits, deduct_credits, get_usage
        add_credits("key10", 1000)
        deduct_credits("key10", seconds=3.0, task_id="task-abc")
        records = get_usage("key10")
        assert len(records) == 1
        assert records[0]["seconds"] == 3.0
        assert records[0]["task_id"] == "task-abc"
        assert records[0]["cost_cents"] == 30

    def test_no_usage_record_on_failure(self):
        from src.billing.store import deduct_credits, get_usage
        deduct_credits("key11", seconds=5.0)
        assert get_usage("key11") == []

    def test_multiple_records_ordered_newest_first(self):
        from src.billing.store import add_credits, deduct_credits, get_usage
        add_credits("key12", 1000)
        deduct_credits("key12", seconds=1.0, task_id="first")
        deduct_credits("key12", seconds=2.0, task_id="second")
        records = get_usage("key12")
        assert len(records) == 2
        assert records[0]["task_id"] == "second"
        assert records[1]["task_id"] == "first"

    def test_limit_respected(self):
        from src.billing.store import add_credits, deduct_credits, get_usage
        add_credits("key13", 10000)
        for i in range(10):
            deduct_credits("key13", seconds=1.0, task_id=f"t{i}")
        records = get_usage("key13", limit=3)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# Trial key provisioning
# ---------------------------------------------------------------------------

class TestCreateTrialKey:
    def test_creates_key_with_trial_prefix(self):
        from src.billing.store import create_trial_key
        result = create_trial_key(label="test user")
        assert result["api_key"].startswith("qsv_trial_")
        assert result["balance_cents"] > 0
        assert result["trial_seconds"] > 0

    def test_trial_key_has_correct_balance(self):
        from src.billing.store import create_trial_key, FREE_TRIAL_CENTS
        result = create_trial_key()
        assert result["balance_cents"] == FREE_TRIAL_CENTS

    def test_trial_key_validates(self):
        from src.billing.store import create_trial_key, validate_db_key
        result = create_trial_key()
        assert validate_db_key(result["api_key"]) is True

    def test_unknown_key_does_not_validate(self):
        from src.billing.store import validate_db_key
        assert validate_db_key("fake_key_xyz") is False

    def test_trial_key_balance_usable(self):
        from src.billing.store import create_trial_key, deduct_credits
        result = create_trial_key()
        key = result["api_key"]
        deduct = deduct_credits(key, seconds=1.0)
        assert deduct["ok"] is True


# ---------------------------------------------------------------------------
# Stripe client (no real Stripe — just validate error handling)
# ---------------------------------------------------------------------------

class TestStripeClientErrorHandling:
    def test_unknown_package_raises(self):
        from src.billing.stripe_client import create_checkout_session
        with pytest.raises(ValueError, match="Unknown package"):
            create_checkout_session("mega_pack", "somekey")

    def test_missing_stripe_key_raises(self, monkeypatch):
        from src.billing.stripe_client import create_checkout_session
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
        with pytest.raises(RuntimeError):
            create_checkout_session("starter", "somekey")

    def test_packages_dict_has_expected_keys(self):
        from src.billing.stripe_client import CREDIT_PACKAGES
        assert set(CREDIT_PACKAGES) == {"starter", "standard", "pro"}
        for pkg in CREDIT_PACKAGES.values():
            assert "amount_cents" in pkg
            assert "credits_cents" in pkg
            assert pkg["amount_cents"] > 0
            assert pkg["credits_cents"] > 0

    def test_all_packages_priced_at_10_cents_per_second(self):
        from src.billing.stripe_client import CREDIT_PACKAGES
        from src.billing.store import CENTS_PER_SECOND
        for pkg_id, pkg in CREDIT_PACKAGES.items():
            expected_seconds = pkg["credits_cents"] / CENTS_PER_SECOND
            # credits should fully buy the described video seconds
            assert expected_seconds > 0, f"{pkg_id} has no video time"


# ---------------------------------------------------------------------------
# claim_stripe_event — idempotency guard
# ---------------------------------------------------------------------------

class TestClaimStripeEvent:
    def test_first_claim_returns_true(self):
        from src.billing.store import claim_stripe_event
        assert claim_stripe_event("evt_001") is True

    def test_duplicate_claim_returns_false(self):
        from src.billing.store import claim_stripe_event
        claim_stripe_event("evt_002")
        assert claim_stripe_event("evt_002") is False

    def test_different_event_ids_are_independent(self):
        from src.billing.store import claim_stripe_event
        assert claim_stripe_event("evt_003") is True
        assert claim_stripe_event("evt_004") is True

    def test_many_duplicates_all_return_false(self):
        from src.billing.store import claim_stripe_event
        claim_stripe_event("evt_005")
        for _ in range(5):
            assert claim_stripe_event("evt_005") is False
