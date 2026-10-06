import os
import re
import json
import tempfile
import asyncio
from datetime import datetime, timezone

from PIL import Image
import pytesseract

from telethon import TelegramClient
from telethon.sessions import StringSession


# =========================================================
# 环境变量
# =========================================================

API_ID = int(os.environ["API_ID"])

API_HASH = os.environ["API_HASH"]

TELEGRAM_SESSION = os.environ["TELEGRAM_SESSION"]

SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ.get(
        "SOURCE_CHANNELS",
        ""
    ).split(",")
    if x.strip()
]

TARGET_CHANNEL = os.environ["TARGET_CHANNEL"]


# =========================================================
# 已处理消息
# =========================================================

PROCESSED_FILE = "processed.json"


def load_processed():

    if not os.path.exists(PROCESSED_FILE):
        return {}

    try:

        with open(
            PROCESSED_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


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

PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d[\d\s\-().]{7,}\d)"
    r"(?!\d)",
    re.I
)


TELEGRAM_LINK_PATTERN = re.compile(
    r"(?:https?://)?"
    r"(?:t\.me|telegram\.me)/"
    r"[A-Za-z0-9_]+",
    re.I
)


TELEGRAM_USERNAME_PATTERN = re.compile(
    r"@[A-Za-z0-9_]{4,}",
    re.I
)


WHATSAPP_PATTERN = re.compile(
    r"(?:whatsapp|wa\.me)",
    re.I
)


WECHAT_PATTERN = re.compile(
    r"(?:wechat|weixin|微信|微信号)",
    re.I
)


CONTACT_KEYWORD_PATTERN = re.compile(
    r"(?:"
    r"telegram|"
    r"whatsapp|"
    r"wechat|"
    r"weixin|"
    r"微信|"
    r"微信号|"
    r"客服|"
    r"联系|"
    r"加我|"
    r"私聊|"
    r"contact|"
    r"customer\s*service|"
    r"扫码|"
    r"二维码"
    r")",
    re.I
)


def contains_contact(text):

    if not text:
        return False

    patterns = [
        PHONE_PATTERN,
        TELEGRAM_LINK_PATTERN,
        TELEGRAM_USERNAME_PATTERN,
        WHATSAPP_PATTERN,
        WECHAT_PATTERN,
        CONTACT_KEYWORD_PATTERN,
    ]

    for pattern in patterns:

        if pattern.search(text):

            return True

    return False


# =========================================================
# OCR
# =========================================================

def ocr_image(image_path):

    try:

        image = Image.open(
            image_path
        )

        # RGB
        image = image.convert("RGB")

        width, height = image.size

        # 图片太小则放大
        if width < 1600:

            scale = 1600 / width

            image = image.resize(
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        return text or ""

    except Exception as e:

        print(
            "OCR ERROR:",
            e
        )

        return ""


def image_contains_contact(
    image_path
):

    text = ocr_image(
        image_path
    )

    print(
        "========== OCR =========="
    )

    print(text)

    print(
        "=========================="
    )

    return contains_contact(text)


# =========================================================
# 判断是不是今天
# =========================================================

def is_today(message_date):

    now = datetime.now(
        timezone.utc
    )

    today = now.date()

    return (
        message_date.astimezone(
            timezone.utc
        ).date()
        == today
    )


# =========================================================
# Telegram Client
# =========================================================

client = TelegramClient(
    StringSession(
        TELEGRAM_SESSION
    ),
    API_ID,
    API_HASH
)


# =========================================================
# 判断消息是否已经处理
# =========================================================

def message_key(
    channel,
    message_id
):

    return (
        f"{channel}:{message_id}"
    )


# =========================================================
# 处理消息
# =========================================================

async def process_message(
    source_name,
    message
):

    if not message:

        return False

    # ----------------------------------
    # 只处理今天
    # ----------------------------------

    if not is_today(
        message.date
    ):

        return False

    key = message_key(
        source_name,
        message.id
    )

    # ----------------------------------
    # 防止重复
    # ----------------------------------

    if key in processed.get(
        "messages",
        []
    ):

        print(
            "SKIP DUPLICATE:",
            key
        )

        return False

    text = (
        message.message
        or ""
    )

    # ----------------------------------
    # 文字联系方式检测
    # ----------------------------------

    if contains_contact(text):

        print(
            "SKIP CONTACT TEXT:",
            key
        )

        processed.setdefault(
            "messages",
            []
        ).append(key)

        save_processed(
            processed
        )

        return False

    # =====================================================
    # 图片
    # =====================================================

    if message.photo:

        image_path = None

        try:

            temp_dir = tempfile.mkdtemp()

            image_path = (
                await client.download_media(
                    message,
                    file=temp_dir
                )
            )

            if not image_path:

                print(
                    "IMAGE DOWNLOAD FAILED:",
                    key
                )

                return False

            # ----------------------------------
            # OCR
            # ----------------------------------

            if image_contains_contact(
                image_path
            ):

                print(
                    "SKIP CONTACT IMAGE:",
                    key
                )

                processed.setdefault(
                    "messages",
                    []
                ).append(key)

                save_processed(
                    processed
                )

                return False

            # ----------------------------------
            # 转发图片
            # ----------------------------------

            await client.send_file(
                TARGET_CHANNEL,
                image_path,
                caption=text or None
            )

            print(
                "FORWARDED IMAGE:",
                key
            )

        except Exception as e:

            print(
                "IMAGE ERROR:",
                key,
                e
            )

            return False

        finally:

            if image_path:

                try:

                    os.remove(
                        image_path
                    )

                except Exception:

                    pass

    # =====================================================
    # 纯文字
    # =====================================================

    elif text.strip():

        try:

            await client.send_message(
                TARGET_CHANNEL,
                text
            )

            print(
                "FORWARDED TEXT:",
                key
            )

        except Exception as e:

            print(
                "TEXT ERROR:",
                key,
                e
            )

            return False

    else:

        print(
            "SKIP EMPTY:",
            key
        )

        return False

    # ----------------------------------
    # 记录成功处理
    # ----------------------------------

    processed.setdefault(
        "messages",
        []
    ).append(key)

    # 防止 processed.json 无限增长
    processed["messages"] = (
        processed["messages"][-5000:]
    )

    save_processed(
        processed
    )

    return True


# =========================================================
# 扫描来源频道
# =========================================================

async def scan_channel(
    source
):

    print(
        "\n================================"
    )

    print(
        "SCANNING:",
        source
    )

    print(
        "================================"
    )

    try:

        entity = await client.get_entity(
            source
        )

        count = 0

        async for message in client.iter_messages(
            entity,
            limit=200
        ):

            if not is_today(
                message.date
            ):

                # iter_messages 是倒序，
                # 遇到昨天消息就可以结束
                break

            await process_message(
                source,
                message
            )

            count += 1

        print(
            "MESSAGES CHECKED:",
            count
        )

    except Exception as e:

        print(
            "CHANNEL ERROR:",
            source,
            e
        )


# =========================================================
# 主程序
# =========================================================

async def main():

    print(
        "================================"
    )

    print(
        "Telegram Forwarder Starting..."
    )

    print(
        "================================"
    )

    print(
        "SOURCE CHANNELS:",
        SOURCE_CHANNELS
    )

    print(
        "TARGET CHANNEL:",
        TARGET_CHANNEL
    )

    await client.start()

    print(
        "Telegram login successful."
    )

    for source in SOURCE_CHANNELS:

        await scan_channel(
            source
        )

    await client.disconnect()

    print(
        "\nDONE."
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )
