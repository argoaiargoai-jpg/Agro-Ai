import logging
import smtplib
from email.message import EmailMessage

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
