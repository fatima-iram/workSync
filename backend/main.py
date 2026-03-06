import os
import base64
from datetime import datetime, timedelta
from dateutil import parser as date_parser

# Gmail API
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pickle

# NLP
import spacy

nlp = spacy.load("en_core_web_sm")

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


# ------------------------
# GMAIL AUTH
# ------------------------

def authenticate_gmail():
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service


# ------------------------
# FETCH EMAILS
# ------------------------

def get_recent_emails(service, max_results=5):
    results = service.users().messages().list(
        userId='me',
        labelIds=['INBOX'],
        maxResults=max_results,
        q="newer_than:3d -category:promotions -category:social -in:spam"
    ).execute()

    messages = results.get('messages', [])
    email_data = []

    for msg in messages:
        msg_id = msg['id']
        message = service.users().messages().get(
            userId='me',
            id=msg_id,
            format='full'
        ).execute()

        headers = message['payload']['headers']
        subject = ""
        sender = ""

        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            if header['name'] == 'From':
                sender = header['value']

        body = ""
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    body = base64.urlsafe_b64decode(
                        part['body']['data']).decode('utf-8', errors="ignore")
        else:
            if 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']).decode('utf-8', errors="ignore")

        email_data.append({
            "id": msg_id,
            "subject": subject,
            "sender": sender,
            "body": body
        })

    return email_data


# ------------------------
# NLP ENGINE
# ------------------------

ACTION_VERBS = [
    "submit", "complete", "review", "attend",
    "prepare", "send", "update",
    "meet", "join", "discuss", "schedule",
    "confirm", "reply", "approve", "sign",
    "upload", "register", "apply"
]

REQUEST_WORDS = [
    "please", "kindly", "ensure",
    "required", "must", "important"
]

SPAM_KEYWORDS = [
    "buy now", "limited offer", "discount", "sale",
    "free trial", "earn money", "click here",
    "lottery"
]


def is_spam(text):
    text_lower = text.lower()
    return any(word in text_lower for word in SPAM_KEYWORDS)


def extract_deadline(text):
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "DATE":
            try:
                return date_parser.parse(ent.text, fuzzy=True)
            except:
                continue
    return None


def detect_task(text):
    doc = nlp(text)

    has_action = False
    has_request = False

    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in ACTION_VERBS:
            has_action = True

    for word in REQUEST_WORDS:
        if word in text.lower():
            has_request = True

    if has_action and has_request:
        return True

    if extract_deadline(text) and has_action:
        return True

    return False


def calculate_priority(text, deadline):
    score = 0
    text_lower = text.lower()

    if "urgent" in text_lower or "asap" in text_lower:
        score += 3

    if deadline and deadline < datetime.now() + timedelta(days=1):
        score += 3

    if score >= 5:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"


def generate_gmail_link(message_id):
    return f"https://mail.google.com/mail/u/0/#all/{message_id}"
# ------------------------
# PROCESS EMAILS
# ------------------------

def process_emails(emails):
    print("Total emails fetched:", len(emails))

    for email in emails:
        combined_text = email["subject"] + " " + email["body"]

        if is_spam(combined_text):
            continue

        deadline = extract_deadline(combined_text)
        task_detected = detect_task(combined_text)

        if task_detected:
            priority = calculate_priority(combined_text, deadline)
            gmail_link = generate_gmail_link(email["id"])

            print("\n==============================")
            print("📌 TASK DETECTED")
            print("Subject   :", email["subject"])
            print("Sender    :", email["sender"])
            print("Deadline  :", deadline)
            print("Priority  :", priority)
            print("Redirect  :", gmail_link)
            print("==============================\n")


# ------------------------
# MAIN
# ------------------------

if __name__ == "__main__":
    print("Connecting to Gmail...")
    service = authenticate_gmail()

    print("Fetching emails...")
    emails = get_recent_emails(service)

    print("Processing emails...")
    process_emails(emails)