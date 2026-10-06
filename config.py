import os

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")

PHONE = os.getenv("PHONE")

SOURCE_CHANNELS = [
    x.strip()
    for x in os.getenv("SOURCE_CHANNELS", "").split(",")
    if x.strip()
]

TARGET_CHANNEL = os.getenv("TARGET_CHANNEL")

SESSION_NAME = "telegram_forwarder"

LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))
