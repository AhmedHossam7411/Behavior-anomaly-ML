import json
import re
from groq import Groq
from app.config import GROQ_API_KEY

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def analyze_behavior(confidence, features, context=None):
    context_str = json.dumps({
        "page":  features.get("CurrentPage")  or features.get("currentPage"),
        "stage": features.get("Context")      or features.get("context")
    }, indent=4)

    prompt = f"""
You are a cybersecurity analyst.

Analyze the user behavior using the features provided.

 context={context_str}

IMPORTANT:
- Mention specific features (TypingRate, ClickRate, MouseMoveRate, etc.)
- Explain WHY the behavior is suspicious or normal
- Do NOT say "based on prediction"
- Be concise and technical
- Provide actionable insights if possible
- Differentiate between bot activity and human anomalies

BOT DETECTION RULES (SECOND HIGHEST PRIORITY — evaluate before context rules):

0. StdMouseSpeed is the single most reliable bot indicator:
   - StdMouseSpeed < 0.05: CRITICAL — human mouse movement ALWAYS has variance due to muscle tremor and attention shifts. Near-zero standard deviation is physically impossible for a human and is definitive proof of scripted movement. Flag as HIGH regardless of page context.
   - StdMouseSpeed < 0.05 AND TypingRate = 0 AND ClickRate > 5: Complete bot signature — HIGH risk. Do NOT apply the "TypingRate=0 is normal" context rule here; the combination is the anomaly.
   - AvgMouseIdle < 20ms: Humans always pause briefly between actions. Near-zero idle time means the script never waits — automated behavior.
   - StdClickInterval < 10 AND ClickCount > 20: Robots click at perfectly regular intervals; human click timing varies naturally.

ATTACK SIGNAL RULES (ABSOLUTE PRIORITY — evaluate before anything else):

1. ATTACK STRINGS (riskLevel MUST be HIGH):
   - HackingStringDetected = 1: user typed SQL injection, XSS, path traversal, or similar. Reference DetectedPatterns in your reason.
   - SuspiciousPasteDetected = 1: user pasted an attack payload. Pasting bypasses keystroke monitoring — treat as equally serious.

2. ABNORMAL INPUT (HIGH risk):
   - AbnormalInputDetected = 1: user submitted input exceeding 500 characters — likely fuzzing or buffer overflow attempt.

3. DEVTOOLS USAGE (MEDIUM to HIGH depending on combination):
   - DevToolsShortcutCount > 3: excessive DevTools key presses — likely actively inspecting tokens/requests.
   - DevToolsDetected = 1 AND DevToolsShortcutCount > 0: DevTools is open and was opened via shortcut — suspicious in a production government app. MEDIUM risk alone, HIGH if combined with other signals.
   - DevToolsDetected = 1 alone: could be a developer — flag as LOW/MEDIUM only.

4. CHALLENGE BYPASS ATTEMPTS (HIGH risk):
   - UnauthorizedAttempts > 2: user repeatedly tried to navigate past an active security challenge — clear evasion behavior.
   - UnauthorizedAttempts > 0: flag as MEDIUM, note the bypass attempts.

5. PASTE BEHAVIOR (factor in context):
   - PasteCount > 5 in a 30-second window: unusually high paste frequency — possible automated or scripted input.

No context rule below can override rules 1 or 2. A user typing attack strings or pasting payloads is HIGH risk regardless of page or behavioral metrics.

CONTEXT RULES (apply only when no attack signals AND no bot signals are present):
- Consider if the behavior matches the page.
- Login and Registration forms typically involve typing. Note: Password managers may cause lower typing rates.
- Navigation, Admin, and Dashboard pages are primarily about reading and clicking.
- IT IS PERFECTLY NORMAL to have absolutely no typing (TypingRate = 0) on a navigation or admin page — BUT ONLY when StdMouseSpeed is within human range (> 0.1). If StdMouseSpeed is near-zero, the "normal TypingRate=0" rule does NOT apply.
- Reserve HIGH risk for obvious bot activity: near-zero StdMouseSpeed, perfectly uniform click intervals, near-zero idle time.

Confidence: {confidence}

Features:
{json.dumps(features, indent=2)}

Respond ONLY in JSON. Choose riskLevel from "LOW", "MEDIUM", or "HIGH":
{{
  "riskLevel": "<Determine Risk: LOW, MEDIUM, or HIGH>",
  "reason": "Short explanation referencing actual features and context rules"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    content = response.choices[0].message.content

    # Safe JSON extraction
    try:
        json_str = re.search(r"\{.*\}", content, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return {
            "riskLevel": "UNKNOWN",
            "reason": content
        }