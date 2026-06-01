"""Email notification for daily stock picks.

Sends an HTML email with the top-10 picks table after each scheduled
pipeline run.  SMTP settings are persisted to a JSON config file so
they survive app restarts.
"""

from __future__ import annotations

import json
import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SMTP_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "smtp_config.json"
)


# ------------------------------------------------------------------
# Config persistence
# ------------------------------------------------------------------

def get_smtp_config() -> dict:
    """Load SMTP settings from disk."""
    if _SMTP_CONFIG_PATH.exists():
        try:
            with open(_SMTP_CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_smtp_config(cfg: dict) -> None:
    """Persist SMTP settings to disk."""
    with open(_SMTP_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def is_email_configured() -> bool:
    """Return True if enough SMTP settings exist to send email."""
    cfg = get_smtp_config()
    return bool(
        cfg.get("enabled")
        and cfg.get("smtp_server")
        and cfg.get("sender_email")
        and cfg.get("sender_password")
        and cfg.get("recipient_email")
    )


# ------------------------------------------------------------------
# Email formatting
# ------------------------------------------------------------------

def _format_picks_html(df: pd.DataFrame, run_time: str) -> str:
    """Build an HTML email body from a picks DataFrame."""
    if df.empty:
        return (
            "<h2>Daily Stock Picks — No Picks Today</h2>"
            f"<p>Run time: {run_time} UTC</p>"
            "<p>The elite pool was below 75 — weak signal day. "
            "No picks were recorded.</p>"
        )

    elite_pool = int(df["elite_pool_size"].iloc[0]) if "elite_pool_size" in df.columns else "N/A"

    # Build the picks table
    rows_html = ""
    display_cols = [
        ("rank", "Rank"),
        ("ticker", "Ticker"),
        ("close_price", "Close"),
        ("ensemble_score", "Score"),
        ("cls_proba", "Classifier P"),
        ("pred_mfd", "Pred MFD"),
        ("z_cls", "Z_cls"),
        ("z_ltr", "Z_ltr"),
        ("elite_pool_size", "Pool"),
        ("market_cap", "Market Cap"),
        ("sector", "Sector"),
        ("sentiment_score", "Sentiment"),
    ]

    # Filter to columns that exist
    cols = [(c, h) for c, h in display_cols if c in df.columns]

    header = "".join(
        f'<th style="padding:8px;border:1px solid #ddd;background:#f5f5f5;">{h}</th>'
        for _, h in cols
    )

    for _, row in df.iterrows():
        cells = ""
        for col, _ in cols:
            val = row.get(col, "")
            if col == "close_price" and pd.notna(val):
                cells += f'<td style="padding:8px;border:1px solid #ddd;">${val:,.2f}</td>'
            elif col == "market_cap" and pd.notna(val):
                mc = float(val)
                if mc >= 1e12:
                    cells += f'<td style="padding:8px;border:1px solid #ddd;">${mc/1e12:.1f}T</td>'
                elif mc >= 1e9:
                    cells += f'<td style="padding:8px;border:1px solid #ddd;">${mc/1e9:.1f}B</td>'
                elif mc >= 1e6:
                    cells += f'<td style="padding:8px;border:1px solid #ddd;">${mc/1e6:.0f}M</td>'
                else:
                    cells += f'<td style="padding:8px;border:1px solid #ddd;">{val}</td>'
            elif col in ("cls_proba", "pred_mfd", "ensemble_score") and pd.notna(val):
                cells += f'<td style="padding:8px;border:1px solid #ddd;">{float(val):.4f}</td>'
            elif col in ("z_cls", "z_ltr", "sentiment_score") and pd.notna(val):
                cells += f'<td style="padding:8px;border:1px solid #ddd;">{float(val):.3f}</td>'
            else:
                cells += f'<td style="padding:8px;border:1px solid #ddd;">{val}</td>'
        rows_html += f"<tr>{cells}</tr>\n"

    html = f"""
    <h2>📈 Daily Stock Picks — {run_time}</h2>
    <p><strong>Elite Pool Size:</strong> {elite_pool}
       &nbsp;|&nbsp; <strong>Picks:</strong> {len(df)}</p>

    <table style="border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:14px;">
    <thead><tr>{header}</tr></thead>
    <tbody>{rows_html}</tbody>
    </table>

    <p style="margin-top:16px;color:#888;font-size:12px;">
    This is an automated email from the AI Stock Predictor pipeline.
    Picks are model-based estimates, not financial advice.
    </p>
    """
    return html


# ------------------------------------------------------------------
# Email sending
# ------------------------------------------------------------------

def send_picks_email(df: pd.DataFrame) -> bool:
    """Send the daily picks email.  Returns True on success."""
    cfg = get_smtp_config()
    if not is_email_configured():
        logger.debug("Email not configured — skipping notification.")
        return False

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = (
            f"Stock Picks — {datetime.now().strftime('%Y-%m-%d')}"
            f" ({len(df)} picks)" if not df.empty
            else f"Stock Picks — {datetime.now().strftime('%Y-%m-%d')} (no picks)"
        )
        msg["From"] = cfg["sender_email"]
        msg["To"] = cfg["recipient_email"]

        html_body = _format_picks_html(df, run_time)
        msg.attach(MIMEText(html_body, "html"))

        smtp_server = cfg.get("smtp_server", "smtp.gmail.com")
        smtp_port = int(cfg.get("smtp_port", 587))

        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(cfg["sender_email"], cfg["sender_password"])
            server.send_message(msg)

        logger.info("Picks email sent to %s", cfg["recipient_email"])
        return True

    except Exception as e:
        logger.exception("Failed to send picks email: %s", e)
        return False


def send_test_email() -> tuple[bool, str]:
    """Send a test email to verify SMTP settings.  Returns (success, message)."""
    cfg = get_smtp_config()
    if not is_email_configured():
        return False, "Email not configured. Fill in all SMTP fields first."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Stock Predictor — Test Email"
        msg["From"] = cfg["sender_email"]
        msg["To"] = cfg["recipient_email"]

        html = (
            "<h2>Test Email</h2>"
            "<p>Your SMTP settings are working correctly. "
            "You will receive daily stock picks at this address.</p>"
            f"<p>Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>"
        )
        msg.attach(MIMEText(html, "html"))

        smtp_server = cfg.get("smtp_server", "smtp.gmail.com")
        smtp_port = int(cfg.get("smtp_port", 587))

        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(cfg["sender_email"], cfg["sender_password"])
            server.send_message(msg)

        return True, f"Test email sent to {cfg['recipient_email']}"

    except smtplib.SMTPAuthenticationError:
        return False, (
            "Authentication failed. For Gmail, use an App Password "
            "(not your regular password). "
            "Go to https://myaccount.google.com/apppasswords to create one."
        )
    except Exception as e:
        return False, f"Error: {e}"
