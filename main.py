import os
import re
import json
import asyncio
import tempfile

from PIL import Image
import pytesseract

from telethon import TelegramClient
from telethon.sessions import StringSession


# =========================
# GitHub Secrets
# =========================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION = os.environ["TELEGRAM_SESSION"]

SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ["SOURCE_CHANNELS"].split(",")
    if x.strip()
]

TARGET_CHANNEL = os.environ["TARGET_CHANNEL"].strip()


# =========================
# processed.json
# =========================

PROCESSED_FILE = "processed.json"


def load_processed():

    if not os.path.exists(PROCESSED_FILE):

        return {
            "messages": []
        }

    try:

        with open(
            PROCESSED_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {
            "messages": []
        }


def save_processed(data):

    with open(
        PROCESSED_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


processed = load_processed()


# =========================
# 联系方式检测
# =========================

CONTACT_PATTERNS = [

    # 手机号
    r"\b(?:\+?\d[\d\s\-]{7,}\d)\b",

    # Telegram
    r"(?:https?://)?(?:t\.me|telegram\.me)/[A-Za-z0-9_]+",

    # Telegram 用户名
    r"@[A-Za-z][A-Za-z0-9_]{4,31}",

    # WhatsApp
    r"(?:https?://)?wa\.me/\d+",

    # WhatsApp 联系方式
    r"whatsapp",

    # 微信
    r"微信",

    # WeChat
    r"wechat",

    # 常见联系方式关键词
    r"客服",
    r"联系",
    r"加我",
    r"私聊",
    r"contact",
    r"customer service",
    r"扫码",
    r"二维码",
]


def contains_contact(text):

    if not text:
        return False

    text_lower = text.lower()

    for pattern in CONTACT_PATTERNS:

        if re.search(
            pattern,
            text_lower,
            re.IGNORECASE
        ):

            return True

    return False


# =========================
# 图片 OCR
# =========================

def image_contains_contact(image_path):

    try:

        print(
            "OCR image:",
            image_path
        )

        image = Image.open(
            image_path
        )

        # 图片太小，放大后 OCR
        width, height = image.size

        if width < 1600:

            ratio = 1600 / width

            image = image.resize(
                (
                    int(width * ratio),
                    int(height * ratio)
                )
            )

        ocr_text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        print(
            "OCR TEXT:",
            repr(ocr_text)
        )

        return contains_contact(
            ocr_text
        )

    except Exception as e:

        print(
            "OCR ERROR:",
            e
        )

        return False


# =========================
# 判断是否今天
# =========================

def is_today(message):

    from datetime import datetime, timezone

    if not message.date:

        return False

    message_date = (
        message.date
        .astimezone(timezone.utc)
        .date()
    )

    today = datetime.now(
        timezone.utc
    ).date()

    return message_date == today


# =========================
# 处理一条消息
# =========================

async def process_message(
    client,
    message,
    source_name
):

    key = f"{source_name}:{message.id}"

    # 已经处理过
    if key in processed.get(
        "messages",
        []
    ):

        print(
            "ALREADY PROCESSED:",
            key
        )

        return


    # 不是今天
    if not is_today(message):

        return


    text = message.text or ""


    # =========================
    # 图片消息
    # =========================

    if message.photo:

        image_path = None

        try:

            temp_dir = tempfile.mkdtemp()

            print(
                "Downloading image:",
                key
            )

            image_path = await client.download_media(
                message,
                file=temp_dir
            )

            if not image_path:

                print(
                    "IMAGE DOWNLOAD FAILED:",
                    key
                )

                return


            # =========================
            # OCR 检查图片
            # =========================

            print(
                "Checking image for contact information..."
            )

            image_has_contact = image_contains_contact(
                image_path
            )


            if image_has_contact:

                print(
                    "SKIP IMAGE: CONTACT DETECTED",
                    key
                )

                processed.setdefault(
                    "messages",
                    []
                ).append(key)

                save_processed(
                    processed
                )

                return


            # =========================
            # 检查图片文字说明
            # =========================

            if contains_contact(text):

                print(
                    "SKIP IMAGE CAPTION: CONTACT DETECTED",
                    key
                )

                processed.setdefault(
                    "messages",
                    []
                ).append(key)

                save_processed(
                    processed
                )

                return


            # =========================
            # 转发图片
            # =========================

            print(
                "Forwarding image:",
                key
            )

            await client.send_file(
                TARGET_CHANNEL,
                image_path,
                caption=(
                    text
                    if text.strip()
                    else None
                ),
                parse_mode=None
            )


            print(
                "FORWARDED IMAGE:",
                key
            )


            processed.setdefault(
                "messages",
                []
            ).append(key)

            save_processed(
                processed
            )


        except Exception as e:

            print(
                "IMAGE ERROR:",
                key,
                e
            )

            return


        finally:

            if image_path:

                try:

                    os.remove(
                        image_path
                    )

                except Exception:

                    pass


        return


    # =========================
    # 纯文字消息
    # =========================

    if text.strip():

        if contains_contact(text):

            print(
                "SKIP TEXT: CONTACT DETECTED",
                key
            )

            processed.setdefault(
                "messages",
                []
            ).append(key)

            save_processed(
                processed
            )

            return


        try:

            print(
                "Forwarding text:",
                key
            )

            await client.send_message(
                TARGET_CHANNEL,
                text
            )

            print(
                "FORWARDED TEXT:",
                key
            )

            processed.setdefault(
                "messages",
                []
            ).append(key)

            save_processed(
                processed
            )

        except Exception as e:

            print(
                "TEXT ERROR:",
                key,
                e
            )


# =========================
# 主程序
# =========================

async def main():

    print("==============================")
    print("Telegram Channel Forwarder")
    print("==============================")

    print(
        "Source channels:",
        SOURCE_CHANNELS
    )

    print(
        "Target channel:",
        TARGET_CHANNEL
    )


    client = TelegramClient(
        StringSession(SESSION),
        API_ID,
        API_HASH
    )


    await client.start()

    print(
        "Telegram connected."
    )


    try:

        for source in SOURCE_CHANNELS:

            print(
                "\n=============================="
            )

            print(
                "Scanning:",
                source
            )

            print(
                "=============================="
            )


            try:

                entity = await client.get_entity(
                    source
                )

            except Exception as e:

                print(
                    "SOURCE CHANNEL ERROR:",
                    source,
                    e
                )

                continue


            # =========================
            # 扫描今天的消息
            # =========================

            async for message in client.iter_messages(
                entity,
                limit=200
            ):

                # 一旦发现不是今天的消息
                # 后面的也不用继续扫描
                if not is_today(message):

                    break


                await process_message(
                    client,
                    message,
                    source
                )


    finally:

        await client.disconnect()

        print(
            "Telegram disconnected."
        )


# =========================
# 程序入口
# =========================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
