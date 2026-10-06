import os


API_ID = int(os.getenv("API_ID"))

API_HASH = os.getenv("API_HASH")

TELEGRAM_SESSION = os.getenv("TELEGRAM_SESSION")

SOURCE_CHANNELS = [
    x.strip()
    for x in os.getenv(
        "SOURCE_CHANNELS",
        ""
    ).split(",")
    if x.strip()
]

TARGET_CHANNEL = os.getenv(
    "TARGET_CHANNEL"
)
