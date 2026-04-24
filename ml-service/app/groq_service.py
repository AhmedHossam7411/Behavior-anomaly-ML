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
- differentiate between bot activity and human anomalies

- Consider if behavior matches the page
- Login page should involve typing
- Navigation pages should involve mouse movement
- Flag mismatches (e.g. no typing on login page, excessive clicking on navigation)

Confidence: {confidence}

Features:
{json.dumps(features, indent=2)}

Respond ONLY in JSON:
{{
  "riskLevel": "HIGH",
  "reason": "Short explanation referencing actual features"
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