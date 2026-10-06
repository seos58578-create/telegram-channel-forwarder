import os
import re
import json
import asyncio
import tempfile

from datetime import datetime, timezone, timedelta

from PIL import Image
import pytesseract

from telethon import TelegramClient
from telethon.sessions import StringSession


# =========================================================
# 配置
# =========================================================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION = os.environ["TELEGRAM_SESSION"]

SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ["SOURCE_CHANNELS"].split(",")
    if x.strip()
]

TARGET_CHANNEL = os.environ["TARGET_CHANNEL"].strip()

PROCESSED_FILE = "processed.json"

# 北京时间 UTC+8
BEIJING_TZ = timezone(
    timedelta(hours=8)
)


# =========================================================
# processed.json
# =========================================================

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

            data = json.load(f)

        if not isinstance(data, dict):

            return {
                "messages": []
            }

        data.setdefault(
            "messages",
            []
        )

        return data

    except Exception as e:

        print(
            "processed.json READ ERROR:",
            e
        )

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


# =========================================================
# 联系方式检测
# =========================================================

CONTACT_PATTERNS = [

    # 手机号
    r"\b(?:\+?\d[\d\s\-]{7,}\d)\b",

    # Telegram 链接
    r"(?:https?://)?(?:t\.me|telegram\.me)/[A-Za-z0-9_]+",

    # Telegram 用户名
    r"@[A-Za-z][A-Za-z0-9_]{4,31}",

    # WhatsApp
    r"(?:https?://)?wa\.me/\d+",

    # 微信
    r"微信",

    r"wechat",

    # WhatsApp
    r"whatsapp",

    # 联系方式关键词
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

    for pattern in CONTACT_PATTERNS:

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):

            print(
                "CONTACT DETECTED BY TEXT:",
                pattern
            )

            return True

    return False


# =========================================================
# 图片 OCR
# =========================================================

def image_contains_contact(image_path):

    try:

        print(
            "OCR IMAGE:",
            image_path
        )

        image = Image.open(
            image_path
        )

        width, height = image.size

        print(
            "IMAGE SIZE:",
            width,
            "x",
            height
        )

        # 放大图片
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
            "OCR RESULT:",
            repr(ocr_text[:500])
        )

        if contains_contact(
            ocr_text
        ):

            print(
                "IMAGE CONTACT: YES"
            )

            return True

        print(
            "IMAGE CONTACT: NO"
        )

        return False

    except Exception as e:

        print(
            "OCR ERROR:",
            repr(e)
        )

        return False


# =========================================================
# 判断北京时间是不是今天
# =========================================================

def is_today(message):

    if not message.date:

        return False

    message_time = (
        message.date
        .astimezone(
            BEIJING_TZ
        )
    )

    today = datetime.now(
        BEIJING_TZ
    ).date()

    return (
        message_time.date()
        == today
    )


# =========================================================
# 处理消息
# =========================================================

async def process_message(
    client,
    message,
    source_name
):

    key = (
        f"{source_name}:{message.id}"
    )

    print(
        "--------------------------------"
    )

    print(
        "PROCESS MESSAGE:",
        key
    )

    print(
        "DATE:",
        message.date
    )

    print(
        "IS TODAY:",
        is_today(message)
    )

    print(
        "HAS PHOTO:",
        bool(message.photo)
    )

    text = message.text or ""

    print(
        "TEXT:",
        repr(text[:200])
    )


    # 已经处理
    if key in processed.get(
        "messages",
        []
    ):

        print(
            "SKIP: ALREADY PROCESSED"
        )

        return


    # 不是今天
    if not is_today(message):

        print(
            "SKIP: NOT TODAY"
        )

        return


    # =====================================================
    # 图片消息
    # =====================================================

    if message.photo:

        image_path = None

        try:

            temp_dir = tempfile.mkdtemp()

            print(
                "DOWNLOADING IMAGE..."
            )

            image_path = await client.download_media(
                message,
                file=temp_dir
            )

            if not image_path:

                print(
                    "IMAGE DOWNLOAD FAILED"
                )

                return


            print(
                "IMAGE DOWNLOADED:",
                image_path
            )


            # OCR
            if image_contains_contact(
                image_path
            ):

                print(
                    "SKIP IMAGE: CONTACT FOUND"
                )

                processed[
                    "messages"
                ].append(key)

                save_processed(
                    processed
                )

                return


            # 图片说明文字检查
            if contains_contact(
                text
            ):

                print(
                    "SKIP IMAGE CAPTION: CONTACT FOUND"
                )

                processed[
                    "messages"
                ].append(key)

                save_processed(
                    processed
                )

                return


            # =================================================
            # 发送图片
            # =================================================

            print(
                "SENDING IMAGE TO:",
                TARGET_CHANNEL
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
                "SUCCESS: IMAGE FORWARDED",
                key
            )


            processed[
                "messages"
            ].append(key)

            save_processed(
                processed
            )


        except Exception as e:

            print(
                "IMAGE SEND ERROR:",
                repr(e)
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


    # =====================================================
    # 纯文字
    # =====================================================

    if text.strip():

        if contains_contact(
            text
        ):

            print(
                "SKIP TEXT: CONTACT FOUND"
            )

            processed[
                "messages"
            ].append(key)

            save_processed(
                processed
            )

            return


        try:

            print(
                "SENDING TEXT TO:",
                TARGET_CHANNEL
            )

            await client.send_message(
                TARGET_CHANNEL,
                text
            )

            print(
                "SUCCESS: TEXT FORWARDED",
                key
            )


            processed[
                "messages"
            ].append(key)

            save_processed(
                processed
            )


        except Exception as e:

            print(
                "TEXT SEND ERROR:",
                repr(e)
            )


# =========================================================
# 主程序
# =========================================================

async def main():

    print(
        "========================================"
    )

    print(
        "Telegram Channel Forwarder"
    )

    print(
        "========================================"
    )

    print(
        "SOURCE CHANNELS:",
        SOURCE_CHANNELS
    )

    print(
        "TARGET CHANNEL:",
        TARGET_CHANNEL
    )

    print(
        "BEIJING DATE:",
        datetime.now(
            BEIJING_TZ
        )
    )


    # =====================================================
    # Telegram Client
    # =====================================================

    client = TelegramClient(
        StringSession(SESSION),
        API_ID,
        API_HASH
    )


    print(
        "CONNECTING TELEGRAM..."
    )

    await client.start()

    print(
        "TELEGRAM CONNECTED"
    )


    try:

        # =================================================
        # 遍历源频道
        # =================================================

        for source in SOURCE_CHANNELS:

            print(
                "\n========================================"
            )

            print(
                "SCANNING SOURCE:",
                source
            )

            print(
                "========================================"
            )


            try:

                entity = await client.get_entity(
                    source
                )

                print(
                    "SOURCE ENTITY:",
                    entity
                )

                print(
                    "SOURCE ID:",
                    getattr(
                        entity,
                        "id",
                        None
                    )
                )

                print(
                    "SOURCE TITLE:",
                    getattr(
                        entity,
                        "title",
                        None
                    )
                )


            except Exception as e:

                print(
                    "GET SOURCE ERROR:",
                    repr(e)
                )

                continue


            # =================================================
            # 读取最近 50 条
            # =================================================

            count = 0

            today_count = 0

            async for message in client.iter_messages(
                entity,
                limit=50
            ):

                count += 1

                print(
                    "\nMESSAGE FOUND:",
                    count
                )

                print(
                    "ID:",
                    message.id
                )

                print(
                    "DATE:",
                    message.date
                )

                print(
                    "BEIJING DATE:",
                    (
                        message.date
                        .astimezone(
                            BEIJING_TZ
                        )
                        if message.date
                        else None
                    )
                )

                print(
                    "PHOTO:",
                    bool(message.photo)
                )

                print(
                    "TEXT:",
                    repr(
                        (message.text or "")[:100]
                    )
                )


                if is_today(message):

                    today_count += 1

                    await process_message(
                        client,
                        message,
                        source
                    )

                else:

                    print(
                        "NOT TODAY - CONTINUE SCANNING"
                    )


            print(
                "\nSOURCE SCAN COMPLETE:",
                source
            )

            print(
                "TOTAL MESSAGES READ:",
                count
            )

            print(
                "TODAY MESSAGES:",
                today_count
            )


    finally:

        await client.disconnect()

        print(
            "\nTELEGRAM DISCONNECTED"
        )


# =========================================================
# 程序入口
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
