## Email Task Classification Module

This module fetches emails from Gmail using the Gmail API and classifies them using a transformer-based model.

### Features
- Gmail API integration
- Spam filtering
- Email cleaning and preprocessing
- Zero-shot classification using BART (`facebook/bart-large-mnli`)
- Deadline extraction
- Priority detection
- Structured task output

### Example Output

{
  "subject": "Project Report Submission",
  "sender": "professor@college.edu",
  "category": "Action Required",
  "deadline": "Tomorrow",
  "priority": "High Priority"
}
