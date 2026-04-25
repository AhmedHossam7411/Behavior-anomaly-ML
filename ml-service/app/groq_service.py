import json
import re
from groq import Groq
from app.config import GROQ_API_KEY

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def analyze_behavior( confidence, features,context=None):
    context_str = json.dumps({
        "page": features.get("CurrentPage"),
        "stage": features.get("Context")
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

CONTEXT RULES (CRITICAL):
- Consider if the behavior matches the page.
- Login and Registration forms typically involve typing. Note: Password managers may cause lower typing rates.
- Navigation, Admin, and Dashboard pages are primarily about reading and clicking. 
- IT IS PERFECTLY NORMAL to have absolutely no typing (TypingRate = 0) on a navigation or admin page. Do not flag this as an anomaly.
- Reserve HIGH risk for obvious bot activity (e.g., extremely rapid, perfectly uniform click intervals, zero reading/idle time) or extreme mismatches.

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