import os
import re
import asyncio
import tempfile
from datetime import datetime, timedelta, timezone

from PIL import Image
import pytesseract

from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto

from config import (
    API_ID,
    API_HASH,
    PHONE,
    SOURCE_CHANNELS,
    TARGET_CHANNEL,
    SESSION_NAME,
    LOOKBACK_HOURS
)


# =========================
# 联系方式检测规则
# =========================

PHONE_PATTERN = re.compile(
    r'(?<!\d)'
    r'(?:\+?\d[\d\s\-().]{7,}\d)'
    r'(?!\d)',
    re.I
)

TELEGRAM_PATTERN = re.compile(
    r'(?:https?://)?t\.me/[A-Za-z0-9_]+',
    re.I
)

USERNAME_PATTERN = re.compile(
    r'@[A-Za-z0-9_]{4,}',
    re.I
)

WHATSAPP_PATTERN = re.compile(
    r'(?:whatsapp|wa\.me)',
    re.I
)

CONTACT_WORD_PATTERN = re.compile(
    r'(telegram|联系|客服|加我|私聊|whatsapp|wechat|微信|line|'
    r'contact|customer service)',
    re.I
)


def contains_contact(text):
    """
    检测文字中是否存在联系方式
    """

    if not text:
        return False

    patterns = [
        PHONE_PATTERN,
        TELEGRAM_PATTERN,
        USERNAME_PATTERN,
        WHATSAPP_PATTERN
    ]

    for pattern in patterns:
        if pattern.search(text):
            return True

    return False


def ocr_image(image_path):
    """
    OCR识别图片文字
    """

    try:
        image = Image.open(image_path)

        # 放大图片，提高OCR识别率
        width, height = image.size

        if width < 1600:
            scale = 1600 / width
            image = image.resize(
                (int(width * scale), int(height * scale))
            )

        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        return text

    except Exception as e:
        print("OCR error:", e)
        return ""


def image_contains_contact(image_path):

    text = ocr_image(image_path)

    print("OCR:")
    print(text)

    if contains_contact(text):
        return True

    return False


# =========================
# 日期判断
# =========================

def is_today(message_date):

    now = datetime.now(timezone.utc)

    start = now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    return message_date >= start


# =========================
# Telegram
# =========================

client = TelegramClient(
    SESSION_NAME,
    API_ID,
    API_HASH
)


async def process_message(message):

    if not message:
        return

    # =========================
    # 只处理今天
    # =========================

    if not is_today(message.date):
        return

    print(
        f"Processing message {message.id} "
        f"{message.date}"
    )

    text = message.message or ""

    # =========================
    # 先检查文字
    # =========================

    if contains_contact(text):

        print(
            f"SKIP {message.id}: "
            "text contains contact information"
        )

        return

    # =========================
    # 图片
    # =========================

    if isinstance(message.media, MessageMediaPhoto):

        temp_dir = tempfile.mkdtemp()

        try:

            image_path = await client.download_media(
                message.media,
                file=temp_dir
            )

            if not image_path:
                return

            # OCR
            if image_contains_contact(image_path):

                print(
                    f"SKIP {message.id}: "
                    "image contains contact information"
                )

                return

            # =========================
            # 转发图片 + 原文字
            # =========================

            await client.send_file(
                TARGET_CHANNEL,
                image_path,
                caption=text
            )

            print(
                f"FORWARDED IMAGE: {message.id}"
            )

        finally:

            try:
                os.remove(image_path)
            except:
                pass

    else:

        # =========================
        # 纯文字
        # =========================

        if text.strip():

            await client.send_message(
                TARGET_CHANNEL,
                text
            )

            print(
                f"FORWARDED TEXT: {message.id}"
            )


async def main():

    await client.start(
        phone=PHONE
    )

    print("Telegram connected")

    for source in SOURCE_CHANNELS:

        print(
            f"Scanning channel: {source}"
        )

        try:

            entity = await client.get_entity(
                source
            )

            async for message in client.iter_messages(
                entity,
                limit=200
            ):

                await process_message(
                    message
                )

        except Exception as e:

            print(
                f"Channel error: {source}"
            )

            print(e)


if __name__ == "__main__":

    asyncio.run(main())
