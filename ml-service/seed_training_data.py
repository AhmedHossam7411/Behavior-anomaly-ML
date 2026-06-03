
import pyodbc
import numpy as np
import uuid
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=GovernmentTaskManagementDB;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

PAGES     = ['/departments', '/tasks', '/documents', '/admin', '/login']
CONTEXTS  = ['postAuth', 'preAuth']
UAS       = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1',
]
PLATFORMS   = ['Win32', 'MacIntel', 'Linux x86_64', 'iPhone']
RESOLUTIONS = ['1920x1080', '1536x864', '1440x900', '2560x1440', '1280x800']
TIMEZONES   = ['Africa/Cairo', 'Europe/London', 'America/New_York', 'Asia/Dubai']
ATTACK_PATTERNS = [
    '[SQL Injection] UNION SELECT in Input',
    '[SQL Injection] DROP TABLE in Input',
    '[SQL Injection] UNION SELECT in URL',
    '[XSS] <script> in Input',
    '[XSS] onerror= in Input',
    '[XSS] <img onerror> in Input',
    '[Command Injection] ; unix-cmd in Input',
    '[Command Injection] /bin/sh in Input',
    '[Command Injection] Shellshock () { in Input',
    '[Path Traversal] ../ in URL',
    '[Path Traversal] /etc/passwd in URL',
    '[SSTI] {{...}} Jinja2/Twig in Input',
    '[XXE] <!ENTITY> in Input',
]

INSERT_SQL = """
INSERT INTO BehaviorWindows (
    SessionId, CurrentPage, UserId, Timestamp,
    AvgMouseSpeed, StdMouseSpeed, MouseMoveCount,
    AvgMouseIdle, StdMouseIdle,
    AvgClickDuration, StdClickDuration, ClickCount,
    AvgClickInterval, StdClickInterval,
    AvgDwell, StdDwell, AvgFlight, StdFlight, KeyEventCount,
    TypingRate, Context, ClickRate, MouseMoveRate,
    AvgPreClickSpeed, StdPreClickSpeed,
    UserAgent, Language, ScreenResolution, TimeZone, Platform, HardwareConcurrency,
    HackingStringDetected, DetectedPatterns,
    PasteCount, SuspiciousPasteDetected,
    DevToolsShortcutCount, AbnormalInputDetected,
    DevToolsDetected, UnauthorizedAttempts
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

def ts():
    return datetime.now() - timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

def fp(lo, hi, dp=4): return round(float(np.random.uniform(lo, hi)), dp)
def ri(lo, hi):       return int(np.random.randint(lo, hi + 1))

def normal_row(idx):
    np.random.seed(idx * 7 + 1)
    return (
        str(uuid.uuid4()), random.choice(PAGES),
        f'user-{idx % 6 + 1}', ts(),
        fp(0.5, 3.0),   fp(0.20, 0.90),  ri(20, 130),
        fp(50, 800),    fp(20, 200),
        fp(80, 250),    fp(20, 80),       ri(2, 20),
        fp(2000, 20000),fp(500, 5000),
        fp(80, 300),    fp(20, 80),
        fp(150, 800),   fp(50, 200),      ri(5, 80),
        fp(0.1, 2.5),   'postAuth',
        fp(0.05, 0.50), fp(0.5, 5.0),
        fp(0.5, 2.5),   fp(0.1, 0.8),
        random.choice(UAS), 'en-US',
        random.choice(RESOLUTIONS), random.choice(TIMEZONES),
        random.choice(PLATFORMS), ri(4, 16),
        0, None, 0, 0, 0, 0, 0, 0
    )

def bot_row():
    return (
        str(uuid.uuid4()), random.choice(['/departments', '/admin', '/tasks']),
        f'bot-{ri(1000, 9999)}', ts(),
        fp(2.3, 2.7),   fp(0.005, 0.030), ri(100, 200),
        fp(8, 15),      fp(0.5, 2.0),
        fp(48, 52),     fp(1.0, 3.0),     ri(80, 150),
        fp(195, 205),   fp(2.0, 5.0),
        fp(1.8, 2.2),   fp(0.1, 0.3),
        fp(1.8, 2.2),   fp(0.1, 0.3),    0,
        0.0,            'postAuth',
        fp(10, 15),     fp(18, 25),
        fp(2.3, 2.6),   fp(0.01, 0.05),
        'python-requests/2.31.0', 'en-US',
        '1920x1080', 'UTC', 'Win32', 4,
        0, None, 0, 0, 0, 0, 0, 0
    )

def malicious_row(attack_type):
    hacking = suspicious_paste = abnormal_input = devtools_detected = 0
    paste_count = devtools_shortcuts = unauthorized = 0
    detected_patterns = None

    if attack_type == 'sql':
        hacking = 1
        detected_patterns = random.choice([
            '[SQL Injection] UNION SELECT in Input',
            '[SQL Injection] DROP TABLE in Input',
            '[SQL Injection] UNION SELECT in URL',
        ])
    elif attack_type == 'xss':
        hacking = 1
        detected_patterns = random.choice([
            '[XSS] <script> in Input',
            '[XSS] <img onerror> in Input',
            '[XSS] onerror= in Input',
        ])
    elif attack_type == 'paste':
        hacking = 1
        suspicious_paste = 1
        paste_count = ri(1, 5)
        abnormal_input = random.choice([0, 0, 1])
        detected_patterns = random.choice([
            '[Command Injection] ; unix-cmd in Input',
            '[XSS] <img onerror> in Input',
            '[SSTI] {{...}} Jinja2/Twig in Input',
        ])
    elif attack_type == 'command':
        hacking = 1
        abnormal_input = 1
        detected_patterns = random.choice([
            '[Command Injection] /bin/sh in Input',
            '[Command Injection] Shellshock () { in Input',
            '[Command Injection] curl http in Input',
        ])
    elif attack_type == 'path':
        hacking = 1
        detected_patterns = random.choice([
            '[Path Traversal] ../ in URL',
            '[Path Traversal] /etc/passwd in URL',
            '[XXE] <!ENTITY> in Input',
        ])
    elif attack_type == 'devtools':
        devtools_shortcuts = ri(4, 10)
        devtools_detected = 1
        paste_count = ri(2, 8)
    elif attack_type == 'probe':
        devtools_shortcuts = ri(5, 12)
        devtools_detected = 1
        unauthorized = ri(3, 7)

    return (
        str(uuid.uuid4()),
        random.choice(['/admin', '/departments', '/tasks']),
        f'attacker-{ri(100, 999)}', ts(),
        fp(0.8, 2.5),   fp(0.3, 0.7),    ri(20, 100),
        fp(60, 600),    fp(20, 150),
        fp(90, 220),    fp(25, 70),       ri(3, 25),
        fp(3000, 18000),fp(600, 4000),
        fp(90, 280),    fp(25, 75),
        fp(160, 750),   fp(55, 190),      ri(10, 60),
        fp(0.2, 2.0),   'postAuth',
        fp(0.1, 0.45),  fp(0.6, 4.5),
        fp(0.6, 2.2),   fp(0.1, 0.7),
        random.choice(UAS), 'en-US',
        random.choice(RESOLUTIONS), random.choice(TIMEZONES),
        random.choice(PLATFORMS), ri(4, 16),
        hacking, detected_patterns,
        paste_count, suspicious_paste,
        devtools_shortcuts, abnormal_input,
        devtools_detected, unauthorized
    )

def mixed_row():
    return (
        str(uuid.uuid4()), '/admin',
        f'suspect-{ri(1, 99)}', ts(),
        fp(2.0, 2.6),   fp(0.02, 0.10),  ri(80, 160),
        fp(10, 30),     fp(2, 8),
        fp(48, 55),     fp(2, 6),        ri(50, 120),
        fp(195, 215),   fp(3, 10),
        fp(2, 5),       fp(0.2, 0.5),
        fp(2, 5),       fp(0.2, 0.5),    ri(0, 10),
        fp(0, 0.3),     'postAuth',
        fp(8, 14),      fp(15, 22),
        fp(2.0, 2.5),   fp(0.02, 0.08),
        random.choice(UAS), 'en-US',
        '1920x1080', 'UTC', 'Win32', 4,
        1,
        random.choice(ATTACK_PATTERNS),
        ri(2, 6),
        1,
        ri(3, 8),
        random.randint(0, 1),
        1,
        ri(2, 6)
    )

print("Seeding BehaviorWindows...\n")

for i in range(35):
    cursor.execute(INSERT_SQL, normal_row(i))
print("  ✓  35  normal human rows")

for _ in range(20):
    cursor.execute(INSERT_SQL, bot_row())
print("  ✓  20  bot rows")

ATTACK_TYPES = ['sql', 'xss', 'paste', 'command', 'path', 'devtools', 'probe']
for at in ATTACK_TYPES:
    for _ in range(3):
        cursor.execute(INSERT_SQL, malicious_row(at))
print(f"  ✓  21  malicious user rows  ({len(ATTACK_TYPES)} attack types × 3)")

for _ in range(10):
    cursor.execute(INSERT_SQL, mixed_row())
print("  ✓  10  mixed rows  (bot behavior + attack signals)\n")

conn.commit()
conn.close()

print("✅  86 rows inserted into BehaviorWindows.")
print("   Restart the Python ML service to retrain TabPFN on the new data.")
