import logging
import smtplib
from email.message import EmailMessage
from email.utils import parseaddr

import httpx

from app.core.config import get_settings
from app.core.errors import AppError

log = logging.getLogger("agroai.email")

SUBJECTS = {
    "verify_email": "Verify your AGRO AI email",
    "reset_password": "Reset your AGRO AI password",
}


def send_otp_email(to: str, code: str, purpose: str) -> None:
    s = get_settings()
    subject = SUBJECTS.get(purpose, "Your AGRO AI code")
    body = (
        f"Your AGRO AI verification code is {code}.\n\n"
        f"It expires in {s.otp_ttl_minutes} minutes. If you did not request this, you can ignore this email."
    )
    if s.email_backend == "console":
        log.warning("[EMAIL:console] to=%s subject=%r code=%s", to, subject, code)
        return
    if s.email_backend == "brevo":
        _send_brevo(s, to, subject, body)
        return
    if not s.smtp_host:
        raise AppError(503, "email_unavailable", "Email delivery is not configured.")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = s.email_from, to, subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=10) as smtp:
            smtp.starttls()
            if s.smtp_user:
                smtp.login(s.smtp_user, s.smtp_password)
            smtp.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        log.exception("SMTP send failed")
        raise AppError(503, "email_unavailable", "We couldn't send the email right now. Try again shortly.") from exc


def _send_brevo(s, to: str, subject: str, body: str) -> None:
    """Brevo transactional email over HTTPS (https://developers.brevo.com/reference/sendtransacemail)."""
    key = s.brevo_api_key.get_secret_value().strip()
    sender_name, sender_email = parseaddr(s.email_from)
    if not key or not sender_email:
        raise AppError(503, "email_unavailable", "Email delivery is not configured.")
    sender = {"email": sender_email}
    if sender_name:
        sender["name"] = sender_name
    try:
        r = httpx.post(
            f"{s.brevo_base_url.rstrip('/')}/v3/smtp/email",
            headers={"api-key": key, "accept": "application/json", "content-type": "application/json"},
            json={"sender": sender, "to": [{"email": to}], "subject": subject, "textContent": body},
            timeout=10,
        )
    except httpx.HTTPError as exc:
        log.error("Brevo request failed: %s", type(exc).__name__)
        raise AppError(503, "email_unavailable", "We couldn't send the email right now. Try again shortly.") from exc
    if not 200 <= r.status_code < 300:
        # status and Brevo's error code only: never the key, the recipient or the message
        try:
            code = r.json().get("code")
        except ValueError:
            code = None
        log.error("Brevo rejected the email: status=%s code=%s", r.status_code, code)
        raise AppError(503, "email_unavailable", "We couldn't send the email right now. Try again shortly.")
