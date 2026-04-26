import os
import re
import base64
import pickle
import threading
from datetime import datetime, timedelta

from flask import Flask, jsonify, request, session
from flask_cors import CORS

from google_auth_oauthlib.flow import InstalledAppFlow, Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

import spacy
import pyttsx3
import dateparser
from bs4 import BeautifulSoup          # best HTML cleaner (from code A)
from transformers import pipeline

# ============================================================
# APP SETUP
# ============================================================

app   = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.urandom(24)

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/calendar'
]

# ============================================================
# LOAD NLP MODELS (once at startup)
# ============================================================

print("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")

classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"   # lightweight, fast, reliable
)

PRIORITY_LABELS = [
    "urgent action required",
    "important information",
    "promotional or newsletter",
    "low priority"
]

print("Models loaded.")

# ============================================================
# VOICE ENGINE
# pyttsx3 breaks if reused across calls on Windows —
# safest fix is to init a fresh engine each time in its own thread
# ============================================================

def _speak_all(texts):
    """Init fresh engine, speak all texts, then destroy. Called in thread."""
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
    """Build all lines first, then speak in one fresh engine call."""
    high   = [e for e in processed if e["priority"] == "High"]
    medium = [e for e in processed if e["priority"] == "Medium"]
    low    = [e for e in processed if e["priority"] == "Low"]
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
        name  = extract_sender_name(e["sender"])
        email = extract_sender_email(e["sender"])
        readable_email = email.replace("@", " at ").replace(".", " dot ")

        urgency = {
            "High":   "This is urgent.",
            "Medium": "This is moderately important.",
            "Low":    ""
        }.get(e["priority"], "")

        lines.append(f"Email {i}.")
        lines.append(f"From {name}. Email address: {readable_email}.")
        lines.append(f"Subject: {e['subject']}.")
        if urgency:
            lines.append(urgency)
        if e.get("deadline") and e["deadline"] != "No deadline found":
            lines.append(f"Deadline: {e['deadline']}.")
        if e.get("cal_link"):
            lines.append("A reminder has been added to your Google Calendar.")
        lines.append(f"Summary: {e['summary']}")

    lines.append("End of briefing. Have a productive day.")
    speak_lines(lines)


# ============================================================
# SENDER HELPERS
# ============================================================

def extract_sender_name(sender_str):
    if "<" in sender_str:
        return sender_str.split("<")[0].strip().replace('"', '')
    if "@" in sender_str:
        return sender_str.split("@")[0]
    return sender_str


def extract_sender_email(sender_str):
    match = re.search(r'<([^>]+)>', sender_str)
    if match:
        return match.group(1).strip()
    if "@" in sender_str:
        return sender_str.strip()
    return ""


# ============================================================
# AUTHENTICATION
# ============================================================

def get_services():
    """Load credentials from token.pickle and return Gmail + Calendar services."""
    if not os.path.exists("token.pickle"):
        return None, None
    with open("token.pickle", "rb") as f:
        creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open("token.pickle", "wb") as f:
                pickle.dump(creds, f)
        else:
            return None, None
    gmail_svc    = build("gmail",    "v1", credentials=creds)
    calendar_svc = build("calendar", "v3", credentials=creds)
    return gmail_svc, calendar_svc


# ============================================================
# FETCH EMAILS
# ============================================================

def fetch_emails(gmail_svc, limit=30):
    results  = gmail_svc.users().messages().list(
        userId='me',
        labelIds=['INBOX'],          # INBOX only — no category filter blocks legit emails
        maxResults=limit,
        q="newer_than:7d -in:spam"
    ).execute()

    messages = results.get("messages", [])
    emails   = []

    for msg in messages:
        msg_id  = msg["id"]
        message = gmail_svc.users().messages().get(
            userId="me", id=msg_id, format="full"
        ).execute()

        headers = message["payload"]["headers"]
        subject = ""
        sender  = ""
        for h in headers:
            if h["name"] == "Subject": subject = h["value"]
            if h["name"] == "From":    sender  = h["value"]

        body    = ""
        payload = message["payload"]

        if "parts" in payload:
            for part in payload["parts"]:
                if part["mimeType"] == "text/plain":
                    data = part["body"].get("data")
                    if data:
                        body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                        break
        else:
            data = payload["body"].get("data")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

        # Use BeautifulSoup for best HTML cleaning (code A strength)
        body = clean_html(body)

        emails.append({
            "id": msg_id, "subject": subject,
            "sender": sender, "body": body
        })

    return emails


# ============================================================
# HTML + BODY CLEANING  (BeautifulSoup from code A = best cleaner)
# ============================================================

def clean_html(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[a-f0-9]{20,}', '', text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    text = re.sub(r'[-=_*#~`]{3,}', '', text)
    text = re.sub(r'\b\d{5,}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# SPAM / PROMO FILTER  (combined keyword sets from both codes)
# ============================================================

SPAM_DOMAINS = [
    "nykaa.com", "himalayawellness.com", "economictimesnews.com",
    "livemint.com", "medium.com", "kaggle.com", "jobalert.indeed.com",
    "internshala.com", "updates.internshala.com", "mail.internshala.com",
    "mailer.", "mailchimp", "sendgrid", "amazonses.com", "bounce."
]

SPAM_SUBJECT_PATTERNS = [
    r"\d+%\s*off", r"flash sale", r"\bnewsletter\b",
    r"\bdigest\b", r"unsubscribe", r"\bpromotion\b",
    r"free shipping", r"\bcoupon\b", r"buy now", r"shop now",
    r"order now", r"clearance", r"advertisement"
]

# Only check sender — body check was catching legit emails like
# "please confirm your meeting" (contains "confirm" = matched "offer")
def is_promotional(sender, subject, body=""):
    sl = sender.lower()
    su = subject.lower()

    for d in SPAM_DOMAINS:
        if d in sl: return True
    for p in SPAM_SUBJECT_PATTERNS:
        if re.search(p, su): return True
    return False


# ============================================================
# SUMMARIZER  (spaCy sentence scoring from code B = best)
# ============================================================

IMPORTANT_VERBS = {
    "apply", "submit", "complete", "confirm", "verify", "update",
    "register", "pay", "review", "respond", "check", "attend",
    "join", "accept", "approve", "reject", "invite", "schedule",
    "download", "access", "reset", "authorize", "alert", "notify",
    "sign", "renew", "expire", "cancel", "activate", "enable", "disable"
}


def score_sentence(sent_text):
    doc   = nlp(sent_text)
    score = 0.0
    score += len(doc.ents) * 1.5
    score += sum(2 for t in doc if t.lemma_.lower() in IMPORTANT_VERBS)
    score += sum(0.5 for t in doc if t.dep_ in ("nsubj", "dobj", "pobj"))
    score += len(list(doc.noun_chunks)) * 0.5
    wc     = len(sent_text.split())
    score -= 0.15 * abs(wc - 18)
    if wc < 5 or wc > 60:
        score -= 5
    return score


def summarize_email(subject, body):
    # For very short emails (like "urgent meeting at 10pm"),
    # just return the body directly — it IS the summary
    word_count = len(body.split())
    if word_count < 30:
        return body.strip() if body.strip() else subject.strip()

    doc       = nlp(body[:2000])
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 8]

    if not sentences:
        return body.strip()[:220] if body.strip() else subject.strip()

    # Score and pick best
    scored = sorted(sentences[:15], key=score_sentence, reverse=True)
    best   = scored[0]

    # If score is very low still return first meaningful sentence
    # rather than falling back to subject — body content is more useful
    if score_sentence(best) < 0:
        return sentences[0][:220].strip()

    return best[:220].strip()


# ============================================================
# PRIORITY DETECTION  (rule scoring + NLI tiebreaker from code B)
# ============================================================

HIGH_PRIORITY_SENDERS = [
    "accounts.google.com", "google.com", "github.com", "amazon.com",
    "paypal.com", "bank", "gov", "university", "college",
    "hr@", "careers@", "recruitment", "zoom.us", "microsoft.com"
]

HIGH_PATTERNS = [
    r"\burgent\b", r"\basap\b", r"\bimmediately\b", r"\baction required\b",
    r"\bdeadline\b", r"\bexpires?\b", r"\bdue (today|tomorrow|tonight)\b",
    r"\brespond by\b", r"\btime.sensitive\b", r"\bsecurity alert\b",
    r"\bsuspicious\b", r"\bunauthorized\b", r"\bpassword\b",
    r"\bverif(y|ication)\b", r"\bpayment (due|failed|declined)\b",
    r"\boffer letter\b", r"\binterview\b", r"\bselected\b", r"\brejected\b",
    r"\bfinal (reminder|notice|warning)\b", r"\baccount (locked|suspended)\b",
    r"\bmeeting\b", r"\bat \d{1,2}(:\d{2})?\s*(am|pm)\b",  # "meeting at 10pm"
    r"\bcall\b", r"\btoday\b", r"\btonight\b", r"\btomorrow\b",
    r"\bjoin\b", r"\battend\b", r"\breminder\b"
]

LOW_PATTERNS = [
    r"\bnewsletter\b", r"\bdigest\b", r"\bunsubscribe\b", r"\bweekly\b",
    r"\bdaily\b", r"\bpromotion\b", r"\bsale\b", r"\b\d+% off\b",
    r"\bdiscount\b", r"\bdeal\b", r"\bwellness\b", r"\bexclusive offer\b",
    r"\bflash\b", r"\bnoreply\b", r"\bno-reply\b"
]

# Category words from code A — great for task extraction
ACTION_WORDS = {
    "submit", "complete", "review", "prepare", "finish", "update", "send",
    "upload", "deliver", "finalize", "check", "verify", "fix", "resolve",
    "respond", "reply", "follow", "draft", "edit", "compile", "create",
    "build", "deploy", "test", "configure", "setup", "maintain", "evaluate"
}

MEETING_WORDS = {
    "meeting", "schedule", "arrange", "call", "conference", "discussion",
    "sync", "appointment", "join", "attend", "set meeting", "schedule call"
}

APPROVAL_WORDS = {
    "approve", "confirm", "authorize", "validate", "acknowledge",
    "accept", "agree", "consent", "verify approval", "confirm receipt"
}


def classify_category(text):
    """Code A's category classification — kept as-is, it works well."""
    tl = text.lower()
    for w in ACTION_WORDS:
        if w in tl: return "Action Required"
    for w in MEETING_WORDS:
        if w in tl: return "Meeting"
    for w in APPROVAL_WORDS:
        if w in tl: return "Approval Needed"
    return "Informational"


def detect_priority(subject, sender, body):
    if is_promotional(sender, subject, body):
        return "Low"

    text  = (subject + " " + body).lower()
    score = 0

    for p in HIGH_PATTERNS:
        if re.search(p, text): score += 3
    for p in LOW_PATTERNS:
        if re.search(p, text): score -= 2

    for trusted in HIGH_PRIORITY_SENDERS:
        if trusted in sender.lower():
            score += 2
            break

    # NLI tiebreaker — only when rules are inconclusive
    if -1 <= score <= 1:
        try:
            snippet = (subject + " " + body[:400])[:512]
            result  = classifier(snippet, PRIORITY_LABELS)
            top, conf = result["labels"][0], result["scores"][0]
            if   top == "urgent action required"    and conf > 0.60: score += 3
            elif top == "important information"     and conf > 0.60: score += 1
            elif top == "promotional or newsletter" and conf > 0.60: score -= 2
            elif top == "low priority"              and conf > 0.60: score -= 1
        except Exception:
            pass

    if score >= 3:   return "High"
    elif score >= 1: return "Medium"
    else:            return "Low"


# ============================================================
# DEADLINE EXTRACTION  (code B spaCy NER = more robust)
# ============================================================

VAGUE_DATES = {
    "today", "yesterday", "tomorrow", "recently", "now", "soon",
    "this year", "last year", "next year", "this week", "last week",
    "the early years", "early", "baseline", "annual", "monthly"
}


def extract_deadline(text):
    # Quick keyword check from code A first (fast)
    tl = text.lower()
    if "tonight"  in tl: return "Tonight"
    if "today"    in tl: return "Today"
    if "tomorrow" in tl: return "Tomorrow"

    # Then spaCy NER for real dates (code B strength)
    doc   = nlp(text[:1000])
    dates = []
    for ent in doc.ents:
        if ent.label_ == "DATE":
            val = ent.text.strip()
            if val.lower() not in VAGUE_DATES and len(val) > 3:
                dates.append(val)

    # Regex date pattern from code A as fallback
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    for match in re.finditer(date_pattern, text):
        val = match.group()
        if val not in dates:
            dates.append(val)

    seen, dedup = set(), []
    for d in dates:
        if d.lower() not in seen:
            seen.add(d.lower())
            dedup.append(d)

    return ", ".join(dedup[:3]) if dedup else "No deadline found"


def parse_deadline_to_datetime(deadline_str):
    if deadline_str in ("No deadline found", "Today", "Tomorrow", "Tonight"):
        now = datetime.now()
        if deadline_str == "Tonight":  return now.replace(hour=20, minute=0)
        if deadline_str == "Tomorrow": return now + timedelta(days=1)
        return now
    try:
        return dateparser.parse(
            deadline_str,
            settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False}
        )
    except Exception:
        return None


# ============================================================
# GOOGLE CALENDAR
# ============================================================

PRIORITY_COLOR = {"High": "11", "Medium": "5", "Low": "2"}


def create_calendar_event(calendar_svc, email_data):
    event_dt = parse_deadline_to_datetime(email_data["deadline"])
    if event_dt is None:
        event_dt = datetime.now().replace(
            hour=9, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

    date_str      = event_dt.strftime("%Y-%m-%d")
    date_next_str = (event_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    event_body = {
        "summary": f"[{email_data['priority']}] {email_data['subject']}",
        "description": (
            f"From: {email_data['sender']}\n"
            f"Category: {email_data['category']}\n"
            f"Priority: {email_data['priority']}\n"
            f"Deadline: {email_data['deadline']}\n\n"
            f"Summary:\n{email_data['summary']}\n\n"
            f"Open in Gmail: {email_data['link']}"
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
# PROCESS EMAILS — main analysis pipeline
# ============================================================

def process_emails(emails, calendar_svc):
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    processed      = []
    skipped        = 0

    for e in emails:
        body = e["body"]

        if is_promotional(e["sender"], e["subject"], body):
            skipped += 1
            continue

        if not body or body.strip() == "":
            skipped += 1
            continue

        summary  = summarize_email(e["subject"], body)
        deadline = extract_deadline(e["subject"] + " " + body)
        priority = detect_priority(e["subject"], e["sender"], body)
        category = classify_category(e["subject"] + " " + body)
        link     = f"https://mail.google.com/mail/u/0/#all/{e['id']}"

        email_data = {
            "id":       e["id"],
            "subject":  e["subject"],
            "sender":   e["sender"],
            "summary":  summary,
            "deadline": deadline,
            "priority": priority,
            "category": category,
            "link":     link,
            "cal_link": ""
        }

        # Auto-add High + Medium to calendar
        if priority in ("High", "Medium") and calendar_svc:
            cal_link = create_calendar_event(calendar_svc, email_data)
            email_data["cal_link"] = cal_link

        processed.append(email_data)

    processed.sort(key=lambda x: priority_order.get(x["priority"], 2))
    return processed, skipped


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/auth/login", methods=["GET"])
def login():
    """Start OAuth flow — opens browser for Google sign-in."""
    try:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as f:
            pickle.dump(creds, f)
        return jsonify({"status": "ok", "message": "Authenticated successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/auth/status", methods=["GET"])
def auth_status():
    """Check if user is authenticated."""
    if not os.path.exists("token.pickle"):
        return jsonify({"authenticated": False})
    try:
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)
        if creds and creds.valid:
            gmail_svc = build("gmail", "v1", credentials=creds)
            profile   = gmail_svc.users().getProfile(userId="me").execute()
            return jsonify({
                "authenticated": True,
                "email": profile.get("emailAddress", "")
            })
        return jsonify({"authenticated": False})
    except Exception:
        return jsonify({"authenticated": False})


@app.route("/auth/logout", methods=["POST"])
def logout():
    if os.path.exists("token.pickle"):
        os.remove("token.pickle")
    return jsonify({"status": "ok"})


@app.route("/emails", methods=["GET"])
def get_emails():
    """Fetch and analyze emails."""
    gmail_svc, calendar_svc = get_services()
    if not gmail_svc:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        raw     = fetch_emails(gmail_svc)
        results, skipped = process_emails(raw, calendar_svc)
        return jsonify({
            "emails":  results,
            "skipped": skipped,
            "total":   len(results)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/todos", methods=["GET"])
def get_todos():
    """Extract to-do items from Action Required emails."""
    gmail_svc, calendar_svc = get_services()
    if not gmail_svc:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        raw     = fetch_emails(gmail_svc)
        results, _ = process_emails(raw, calendar_svc)
        todos   = [
            {
                "subject":  e["subject"],
                "sender":   e["sender"],
                "summary":  e["summary"],
                "deadline": e["deadline"],
                "priority": e["priority"],
                "link":     e["link"],
                "done":     False
            }
            for e in results
            if e["category"] == "Action Required" or e["priority"] == "High"
        ]
        return jsonify({"todos": todos})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_voice_thread = None

@app.route("/voice", methods=["POST"])
def voice_read():
    """Trigger voice briefing in background thread."""
    global _voice_thread
    data   = request.get_json()
    emails = data.get("emails", [])
    if not emails:
        return jsonify({"error": "No emails provided"}), 400

    # Don't start if already speaking
    if _voice_thread and _voice_thread.is_alive():
        return jsonify({"error": "Already speaking"}), 400

    _voice_thread = threading.Thread(
        target=voice_briefing_thread, args=(emails,), daemon=True
    )
    _voice_thread.start()
    return jsonify({"status": "ok", "message": "Voice briefing started"})


@app.route("/voice/stop", methods=["POST"])
def voice_stop():
    """Stop voice — kill the thread by letting it die naturally via engine crash."""
    global _voice_thread
    try:
        # Init a fresh engine just to stop any active speech
        import pyttsx3 as _p
        e = _p.init()
        e.stop()
    except Exception:
        pass
    _voice_thread = None
    return jsonify({"status": "ok"})


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("WorkSync backend running on http://localhost:5000")
    app.run(debug=False, port=5000)