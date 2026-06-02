# ============================================================
# WORKSYNC — CALENDAR & VOICE MODULE
# Drop these imports, helpers, and routes into your existing app.py
# ============================================================

# ── STEP 1: Add these imports to the top of your app.py ─────
# (only add what you don't already have)
#
# import threading
# import pyttsx3
# import re
# import dateparser
# from datetime import datetime, timedelta
# from googleapiclient.errors import HttpError


# ============================================================
# CALENDAR HELPERS
# Paste these functions into your app.py
# ============================================================

from datetime import datetime, timedelta
from googleapiclient.errors import HttpError
import dateparser

PRIORITY_COLOR = {"High": "11", "Medium": "5", "Low": "2"}

VAGUE_DATES = {
    "today", "yesterday", "tomorrow", "recently", "now", "soon",
    "this year", "last year", "next year", "this week", "last week",
    "the early years", "early", "baseline", "annual", "monthly"
}


def parse_deadline_to_datetime(deadline_str):
    """Convert a deadline string to a Python datetime object."""
    if not deadline_str or deadline_str == "No deadline found":
        return None
    now = datetime.now()
    if deadline_str == "Tonight":  return now.replace(hour=20, minute=0)
    if deadline_str == "Today":    return now
    if deadline_str == "Tomorrow": return now + timedelta(days=1)
    try:
        return dateparser.parse(
            deadline_str,
            settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False}
        )
    except Exception:
        return None


def create_calendar_event(calendar_svc, email_data):
    """
    Create a Google Calendar all-day event for an email.
    email_data must have: subject, sender, priority, deadline, category, summary, link
    Returns the HTML link to the created event, or "" on failure.
    """
    event_dt = parse_deadline_to_datetime(email_data.get("deadline", ""))
    if event_dt is None:
        # Default: 9am tomorrow
        event_dt = datetime.now().replace(
            hour=9, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

    date_str      = event_dt.strftime("%Y-%m-%d")
    date_next_str = (event_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    event_body = {
        "summary": f"[{email_data['priority']}] {email_data['subject']}",
        "description": (
            f"From: {email_data['sender']}\n"
            f"Category: {email_data.get('category', '')}\n"
            f"Priority: {email_data['priority']}\n"
            f"Deadline: {email_data.get('deadline', 'N/A')}\n\n"
            f"Summary:\n{email_data.get('summary', '')}\n\n"
            f"Open in Gmail: {email_data.get('link', '')}"
        ),
        "start":   {"date": date_str},
        "end":     {"date": date_next_str},
        "colorId": PRIORITY_COLOR.get(email_data["priority"], "2"),
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": 60 * 24},
                {"method": "email", "minutes": 60 * 2},
            ]
        }
    }

    try:
        created = calendar_svc.events().insert(
            calendarId="primary", body=event_body
        ).execute()
        return created.get("htmlLink", "")
    except HttpError as err:
        print(f"  Calendar error: {err}")
        return ""


# ============================================================
# VOICE ASSISTANT HELPERS
# Paste these functions into your app.py
# ============================================================

import threading
import pyttsx3


def extract_sender_name(sender_str):
    if "<" in sender_str:
        return sender_str.split("<")[0].strip().replace('"', '')
    if "@" in sender_str:
        return sender_str.split("@")[0]
    return sender_str


def extract_sender_email(sender_str):
    import re
    match = re.search(r'<([^>]+)>', sender_str)
    if match:
        return match.group(1).strip()
    if "@" in sender_str:
        return sender_str.strip()
    return ""


def _speak_all(texts):
    """Init a fresh pyttsx3 engine, speak all texts, then destroy. Called in a thread."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        voices = engine.getProperty("voices")
        for v in voices:
            if "zira" in v.name.lower() or "female" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        for text in texts:
            if text and text.strip():
                engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Voice error: {e}")


def speak_lines(lines):
    """Run speech in a daemon thread with a fresh engine."""
    t = threading.Thread(target=_speak_all, args=(lines,), daemon=True)
    t.start()
    return t


def voice_briefing_thread(processed):
    """
    Build a full voice briefing from a list of processed email dicts.
    Each email dict should have: priority, sender, subject, deadline, cal_link, summary
    """
    high   = [e for e in processed if e.get("priority") == "High"]
    medium = [e for e in processed if e.get("priority") == "Medium"]
    low    = [e for e in processed if e.get("priority") == "Low"]
    total  = len(processed)

    lines = []

    if total == 0:
        lines.append("You have no important emails right now.")
        speak_lines(lines)
        return

    parts = []
    if high:   parts.append(f"{len(high)} high priority")
    if medium: parts.append(f"{len(medium)} medium priority")
    if low:    parts.append(f"{len(low)} low priority")

    lines.append(
        f"You have {total} important email{'s' if total != 1 else ''}. "
        f"{', '.join(parts)}. Here is your briefing."
    )

    for i, e in enumerate(processed, 1):
        name  = extract_sender_name(e.get("sender", ""))
        email = extract_sender_email(e.get("sender", ""))
        readable_email = email.replace("@", " at ").replace(".", " dot ")

        urgency = {
            "High":   "This is urgent.",
            "Medium": "This is moderately important.",
            "Low":    ""
        }.get(e.get("priority", ""), "")

        lines.append(f"Email {i}.")
        lines.append(f"From {name}. Email address: {readable_email}.")
        lines.append(f"Subject: {e.get('subject', 'No subject')}.")
        if urgency:
            lines.append(urgency)
        deadline = e.get("deadline", "")
        if deadline and deadline != "No deadline found":
            lines.append(f"Deadline: {deadline}.")
        if e.get("cal_link"):
            lines.append("A reminder has been added to your Google Calendar.")
        lines.append(f"Summary: {e.get('summary', '')}")

    lines.append("End of briefing. Have a productive day.")
    speak_lines(lines)


# ============================================================
# FLASK ROUTES
# Paste these routes into your app.py
# (make sure `app` and `get_services` / `fetch_emails` / `process_emails`
#  are already defined in your file before these routes)
# ============================================================

# from flask import jsonify, request
# _voice_thread = None   ← add this near your other globals


# ── Route: GET /calendar/events ──────────────────────────────
# Returns all email events that were added to Google Calendar
# (those with a cal_link). Frontend uses this to render the
# WorkSync calendar grid.
#
# @app.route("/calendar/events", methods=["GET"])
# def calendar_events():
#     gmail_svc, calendar_svc = get_services()
#     if not gmail_svc:
#         return jsonify({"error": "Not authenticated"}), 401
#     try:
#         raw = fetch_emails(gmail_svc)
#         results, _ = process_emails(raw, calendar_svc)
#         events = [
#             {
#                 "id":       e["id"],
#                 "subject":  e["subject"],
#                 "sender":   e["sender"],
#                 "summary":  e["summary"],
#                 "deadline": e["deadline"],
#                 "priority": e["priority"],
#                 "category": e["category"],
#                 "link":     e["link"],
#                 "cal_link": e["cal_link"],
#             }
#             for e in results
#             if e.get("cal_link")  # only events pushed to Google Calendar
#         ]
#         # Also return ALL emails (with/without cal_link) for WorkSync calendar display
#         all_events = [
#             {
#                 "id":       e["id"],
#                 "subject":  e["subject"],
#                 "sender":   e["sender"],
#                 "summary":  e["summary"],
#                 "deadline": e["deadline"],
#                 "priority": e["priority"],
#                 "category": e["category"],
#                 "link":     e["link"],
#                 "cal_link": e.get("cal_link", ""),
#             }
#             for e in results
#         ]
#         return jsonify({"events": events, "all": all_events})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# ── Route: POST /voice ────────────────────────────────────────
# Accepts { "emails": [...] } in JSON body, starts voice briefing
#
# @app.route("/voice", methods=["POST"])
# def voice_read():
#     global _voice_thread
#     data   = request.get_json()
#     emails = data.get("emails", [])
#     if not emails:
#         return jsonify({"error": "No emails provided"}), 400
#     if _voice_thread and _voice_thread.is_alive():
#         return jsonify({"error": "Already speaking"}), 400
#     _voice_thread = threading.Thread(
#         target=voice_briefing_thread, args=(emails,), daemon=True
#     )
#     _voice_thread.start()
#     return jsonify({"status": "ok", "message": "Voice briefing started"})


# ── Route: POST /voice/stop ───────────────────────────────────
#
# @app.route("/voice/stop", methods=["POST"])
# def voice_stop():
#     global _voice_thread
#     try:
#         import pyttsx3 as _p
#         e = _p.init()
#         e.stop()
#     except Exception:
#         pass
#     _voice_thread = None
#     return jsonify({"status": "ok"})
