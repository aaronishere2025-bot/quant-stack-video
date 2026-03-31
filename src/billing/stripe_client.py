"""
Stripe integration for credit purchases.

Credit packages available:
  starter  — $5.00 → 500 cents  (50 s of video)
  standard — $10.00 → 1000 cents (100 s of video)
  pro      — $25.00 → 2500 cents (250 s of video)

Env vars required:
  STRIPE_SECRET_KEY       — Stripe secret key (sk_live_... or sk_test_...)
  STRIPE_WEBHOOK_SECRET   — Signing secret for webhook endpoint
  BILLING_BASE_URL        — e.g. https://yourdomain.com  (for success/cancel redirect)
"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Credit packages
# ---------------------------------------------------------------------------

CREDIT_PACKAGES = {
    "starter": {
        "name": "Starter Pack — 50 s of video",
        "amount_cents": 500,   # $5.00
        "credits_cents": 500,  # credits awarded
        "description": "50 seconds of generated video ($0.10/s)",
    },
    "standard": {
        "name": "Standard Pack — 100 s of video",
        "amount_cents": 1000,
        "credits_cents": 1000,
        "description": "100 seconds of generated video ($0.10/s)",
    },
    "pro": {
        "name": "Pro Pack — 250 s of video",
        "amount_cents": 2500,
        "credits_cents": 2500,
        "description": "250 seconds of generated video ($0.10/s)",
    },
}


def _stripe():
    """Lazy-import stripe so the server can start without it installed."""
    try:
        import stripe as _s
        key = os.environ.get("STRIPE_SECRET_KEY", "")
        if not key:
            raise RuntimeError("STRIPE_SECRET_KEY env var not set")
        _s.api_key = key
        return _s
    except ImportError:
        raise RuntimeError("stripe package not installed — run: pip install stripe")


def create_checkout_session(
    package_id: str,
    api_key: str,
    base_url: Optional[str] = None,
) -> dict:
    """
    Create a Stripe Checkout session for a credit package.

    Returns {"url": "https://checkout.stripe.com/...", "session_id": "cs_..."}.
    Raises ValueError for unknown package_id.
    Raises RuntimeError if Stripe is not configured.
    """
    pkg = CREDIT_PACKAGES.get(package_id)
    if pkg is None:
        raise ValueError(f"Unknown package '{package_id}'. Valid: {list(CREDIT_PACKAGES)}")

    stripe = _stripe()
    base = (base_url or os.environ.get("BILLING_BASE_URL", "http://localhost:8400")).rstrip("/")

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[
            {
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": pkg["name"],
                        "description": pkg["description"],
                    },
                    "unit_amount": pkg["amount_cents"],
                },
                "quantity": 1,
            }
        ],
        mode="payment",
        metadata={
            "api_key": api_key,
            "package_id": package_id,
            "credits_cents": str(pkg["credits_cents"]),
        },
        success_url=f"{base}/dashboard?purchase=success",
        cancel_url=f"{base}/dashboard?purchase=cancelled",
    )
    return {"url": session.url, "session_id": session.id}


def handle_webhook(payload: bytes, sig_header: str) -> Optional[dict]:
    """
    Validate and parse a Stripe webhook event.

    Returns a dict with action info when credits should be granted:
      {"action": "grant_credits", "api_key": "...", "credits_cents": N}
    Returns None for events that require no action.
    Raises ValueError / stripe.error.SignatureVerificationError on bad payload.
    """
    stripe = _stripe()
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET env var not set")

    event = stripe.Webhook.construct_event(payload, sig_header, secret)

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        if session.get("payment_status") == "paid":
            meta = session.get("metadata", {})
            return {
                "action": "grant_credits",
                "event_id": event["id"],
                "api_key": meta.get("api_key", ""),
                "credits_cents": int(meta.get("credits_cents", 0)),
                "package_id": meta.get("package_id", ""),
                "session_id": session["id"],
            }

    return None
