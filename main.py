from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import base64
from transformers import pipeline
import re
from bs4 import BeautifulSoup
import json
import edge_tts

import asyncio
import pygame
import tempfile
import os

#Voice Assistant
pygame.mixer.init()

def speak(text): 
    print("🔊", text)  # also print
    
    async def _speak():
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            temp_path = f.name
        
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        await communicate.save(temp_path)
        
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.unload()
        os.remove(temp_path)
    
    asyncio.run(_speak())
    

def speak_summary(tasks):
    high = sum(1 for t in tasks if t["priority"] == "High Priority")
    today = sum(1 for t in tasks if t["deadline"] == "Today")

    speak(f"You have {len(tasks)} important emails.")
    speak(f"{high} are high priority.")
    speak(f"{today} have deadlines today.")


# Gmail API permission
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# DistilBERT model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

labels = [
"Action Required",
"Meeting",
"Informational",
"Approval Needed",
"High Priority"
]

# Gmail authentication
def authenticate_gmail():

    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json',
        SCOPES
    )

    creds = flow.run_local_server(port=0)

    service = build('gmail', 'v1', credentials=creds)

    return service


# Fetch emails
def fetch_emails(service):

    results = service.users().messages().list(
        userId='me',
        labelIds=['INBOX','CATEGORY_PERSONAL'],
        maxResults=5
    ).execute()

    messages = results.get('messages', [])

    emails = []

    for msg in messages:

        message_id = msg["id"]   # <-- message_id defined here

        txt = service.users().messages().get(
            userId='me',
            id=message_id
        ).execute()

        payload = txt['payload']
        headers = payload['headers']

        subject = ""
        sender = ""

        for header in headers:
            if header['name'] == "Subject":
                subject = header['value']

            if header['name'] == "From":
                sender = header['value']

        body = ""

        parts = payload.get('parts')
        if parts:
            data = parts[0]['body'].get('data')
        else:
            data = payload['body'].get('data')
        if data:
            text = base64.urlsafe_b64decode(data).decode("utf-8")
            body=text
        else:
            body=""

        # Gmail redirect link
        gmail_link = f"https://mail.google.com/mail/u/0/#all/{message_id}"
        
        emails.append({
            "subject": subject,
            "sender": sender,
            "body": body,
            "gmail_link": gmail_link
        })

    return emails

action_words = {
"submit","complete","review","prepare","finish","update","send",
"upload","deliver","finalize","check","verify","fix","resolve",
"investigate","respond","reply","follow","draft","edit","revise",
"compile","summarize","create","build","design","develop",
"implement","deploy","test","debug","install","configure",
"setup","maintain","monitor","track","evaluate","assess"
}

meeting_words = {
"meeting","schedule","arrange","plan","coordinate","call",
"conference","discussion","sync","appointment","join",
"attend","set meeting","schedule call","prepare slides"
}

approval_words = {
"approve","confirm","authorize","validate","acknowledge",
"accept","agree","consent","verify approval","confirm receipt"
}

informational_words = {
"inform","notify","announce","share","update","report",
"broadcast","circulate","mention","state","declare",
"reminder","notice","alert"
}

priority_words = {
"urgent","immediately","asap","priority","important",
"critical","deadline","today","tomorrow","now",
"action required","time sensitive"
}

spam_keywords = {
"sale","offer","discount","deal","buy now","limited offer",
"unsubscribe","promotion","advertisement","shop now",
"order now","exclusive deal","coupon","free shipping",
"clearance","marketing","newsletter"
}

def is_spam(text):

    text = text.lower()

    for word in spam_keywords:
        if word in text:
            return True

    return False


#Deadline extraction
def extract_deadline(text):

    text = text.lower()

    # simple deadline keywords
    if "today" in text:
        return "Today"

    if "tomorrow" in text:
        return "Tomorrow"

    if "tonight" in text:
        return "Tonight"

    # detect date patterns like 12/06/2026
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
    match = re.search(date_pattern, text)

    if match:
        return match.group()

    return "Not specified"

# Priority detection
def detect_priority(text):

    text = text.lower()
    
    for word in priority_words:
        if word in text:
            return "High Priority"
    return "Low Priority"

# Classify email
def classify_email(text):

    text_lower = text.lower()

    # Action detection
    for word in action_words:
        if word in text_lower:
            return "Action Required"

    # Meeting detection
    for word in meeting_words:
        if word in text_lower:
            return "Meeting / Coordination"

    # Approval detection
    for word in approval_words:
        if word in text_lower:
            return "Approval Needed"
        
    #remove empty sentence    
    if not text or text.strip() == "":
        return "Informational"

    # DistilBERT analysis
    result = classifier(
        text[:512], 
        candidate_labels=labels
        )
    return result["labels"][0]


# Remove HTML tags
def clean_email(text):
    
    if not text:
        return ""

    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Remove extra spaces and symbols
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# Main program
def main():

    print("Connecting to Gmail...")

    service = authenticate_gmail()

    print("Fetching emails...")

    emails = fetch_emails(service)

    print("\nAnalyzing Emails...\n")

    tasks = [] 
    speech_queue = []

    for email in emails:

        cleaned_email = clean_email(email["body"])

        if not cleaned_email or cleaned_email.strip() == "":
            continue
        #Skip spam emails
        if is_spam(cleaned_email):
            continue
        category = classify_email(cleaned_email)
        deadline = extract_deadline(cleaned_email)
        priority = detect_priority(cleaned_email)

        task = {
            "subject": email["subject"],
            "sender": email["sender"],
            "category": category,
            "deadline": deadline,
            "priority": priority,
            "gmail_link": email["gmail_link"]
        }

        tasks.append(task)

        # 🗣️ SPEAK EACH EMAIL
        message = (
            f"Email from {email['sender']}. "
            f"Subject: {email['subject']}. "
            f"This is categorized as {category}. "
            f"Deadline is {deadline}. "
            f"Priority level is {priority}."
        )
        speech_queue.append(message)


        print("\n📊 TASK OBJECT")
        print(json.dumps(task, indent=4))

        print("EMAIL TEXT:")
        print(cleaned_email[:200])
        print("-" * 60)

    # ✅ All processing done, now speak
    print("\n🔊 Starting voice readout...\n")
    for message in speech_queue:
        speak(message)  # your existing speak() function works fine here

    speak_summary(tasks) 

if __name__ == "__main__":
    main()
