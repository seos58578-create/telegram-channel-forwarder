# -*- coding: utf-8 -*-

import asyncio
import io
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from PIL import (
    Image,
    ImageEnhance,
    ImageOps,
    ImageDraw,
    ImageFont,
)

from telethon import TelegramClient, utils
from telethon.sessions import StringSession

from telegram import (
    Bot,
    InputFile,
    InputMediaPhoto,
)
from telegram.error import TelegramError


# ============================================================
# 1. 基础配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_FILE = BASE_DIR / "processed.json"

API_ID = int(
    os.environ.get("API_ID", "0")
)

API_HASH = os.environ.get(
    "API_HASH",
    ""
).strip()

TELEGRAM_SESSION = os.environ.get(
    "TELEGRAM_SESSION",
    ""
).strip()

BOT_TOKEN = os.environ.get(
    "BOT_TOKEN",
    ""
).strip()

SOURCE_CHANNELS_RAW = os.environ.get(
    "SOURCE_CHANNELS",
    ""
).strip()

TARGET_CHANNEL_RAW = os.environ.get(
    "TARGET_CHANNEL",
    ""
).strip()

SCAN_LIMIT_RAW = os.environ.get(
    "SCAN_LIMIT",
    "0"
).strip()

try:
    SCAN_LIMIT = int(
        SCAN_LIMIT_RAW or "0"
    )
except Exception:
    SCAN_LIMIT = 0


# 北京时间 UTC+8
BEIJING_TZ = timezone(
    timedelta(hours=8)
)


# ============================================================
# 2. 日志
# ============================================================

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.getLogger(
    "telethon"
).setLevel(logging.WARNING)

logging.getLogger(
    "telethon.network"
).setLevel(logging.WARNING)

logging.getLogger(
    "telethon.extensions"
).setLevel(logging.WARNING)

logging.getLogger(
    "telegram"
).setLevel(logging.WARNING)

logging.getLogger(
    "httpx"
).setLevel(logging.WARNING)


# ============================================================
# 3. OCR
# ============================================================

OCR_LANG = (
    "eng+chi_sim+vie+por"
)

OCR_CONFIGS = [
    "--psm 6",
    "--psm 11",
]


# ============================================================
# 4. 联系方式规则
# ============================================================

CONTACT_PATTERNS = [

    (
        "TELEGRAM_LINK",
        re.compile(
            r"(?:https?://)?"
            r"(?:www\.)?"
            r"(?:t\.me|telegram\.me)/"
            r"[A-Za-z0-9_+/=?-]{4,64}",
            re.IGNORECASE,
        ),
    ),

    (
        "TELEGRAM_USERNAME",
        re.compile(
            r"@[A-Za-z]"
            r"[A-Za-z0-9_]{4,31}"
        ),
    ),

    (
        "WHATSAPP_LINK",
        re.compile(
            r"(?:https?://)?"
            r"(?:www\.)?"
            r"wa\.me/"
            r"\d{7,15}",
            re.IGNORECASE,
        ),
    ),

    (
        "WHATSAPP_TEXT",
        re.compile(
            r"\bwhatsapp\b",
            re.IGNORECASE,
        ),
    ),

    (
        "WECHAT",
        re.compile(
            r"\bwechat\b",
            re.IGNORECASE,
        ),
    ),

    (
        "WECHAT_CN",
        re.compile(
            r"(微信|微信号|微[信號号])",
            re.IGNORECASE,
        ),
    ),

    (
        "CN_PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?86[- ]?)?"
            r"1[3-9]\d{9}"
            r"(?!\d)"
        ),
    ),

    (
        "VN_PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?84[- ]?)?"
            r"(?:0?3|0?5|0?7|0?8|0?9)"
            r"\d{8}"
            r"(?!\d)"
        ),
    ),
]


STRONG_CONTACT_TYPES = {
    "TELEGRAM_LINK",
    "TELEGRAM_USERNAME",
    "WHATSAPP_LINK",
    "WHATSAPP_TEXT",
    "WECHAT",
    "WECHAT_CN",
}


PHONE_CONTACT_TYPES = {
    "CN_PHONE",
    "VN_PHONE",
}


# ============================================================
# 5. 全局变量
# ============================================================

client = None
bot = None

BOT_TARGET_CHANNEL = None


# ============================================================
# 6. 时间
# ============================================================

def now_beijing():

    return datetime.now(
        BEIJING_TZ
    )


# ============================================================
# 7. 频道名称处理
# ============================================================

def normalize_channel(value):

    value = (
        value or ""
    ).strip()

    if not value:
        return ""

    prefixes = [
        "https://t.me/",
        "http://t.me/",
        "https://telegram.me/",
        "http://telegram.me/",
    ]

    for prefix in prefixes:

        if value.startswith(prefix):

            value = value[
                len(prefix):
            ]

            break

    value = value.rstrip("/")

    if value.startswith("@"):
        return value

    if value.startswith("-100"):
        return value

    if re.fullmatch(
        r"-?\d+",
        value
    ):
        return value

    return "@" + value


def parse_source_channels():

    if not SOURCE_CHANNELS_RAW:
        return []

    result = []

    for item in SOURCE_CHANNELS_RAW.split(","):

        item = normalize_channel(
            item
        )

        if item:
            result.append(item)

    return result


# ============================================================
# 8. processed.json
# ============================================================

def load_processed():

    if not PROCESSED_FILE.exists():

        return set()

    try:

        with open(
            PROCESSED_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        messages = data.get(
            "messages",
            []
        )

        if not isinstance(
            messages,
            list,
        ):
            return set()

        return {
            str(x)
            for x in messages
        }

    except Exception as e:

        print(
            f"⚠️ processed.json 读取失败: {e}"
        )

        return set()


def save_processed(
    processed
):

    normalized = {
        str(x)
        for x in processed
    }

    sorted_messages = sorted(
        normalized
    )

    data = {
        "messages":
            sorted_messages[-10000:]
    }

    tmp_file = (
        PROCESSED_FILE.with_suffix(
            ".tmp"
        )
    )

    with open(
        tmp_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp_file.replace(
        PROCESSED_FILE
    )


def message_key(
    source_index,
    message_id,
):

    return (
        f"{source_index}:{message_id}"
    )


def is_already_processed(
    processed,
    source_index,
    message_id,
):

    key = message_key(
        source_index,
        message_id,
    )

    if key in processed:
        return True

    # 兼容旧版本
    if str(message_id) in processed:
        return True

    return False


# ============================================================
# 9. 联系方式检测
# ============================================================

def detect_contacts(text):

    if not text:
        return []

    found = []

    for (
        contact_type,
        pattern,
    ) in CONTACT_PATTERNS:

        try:

            matches = pattern.findall(
                text
            )

            for match in matches:

                if isinstance(
                    match,
                    tuple,
                ):

                    match = " ".join(
                        match
                    )

                found.append(
                    {
                        "type":
                            contact_type,
                        "value":
                            str(match),
                    }
                )

        except Exception:

            continue

    return found


def format_contacts(
    contacts
):

    if not contacts:
        return ""

    return ", ".join(
        f"{x['type']}={x['value']}"
        for x in contacts
    )


# ============================================================
# 10. 二维码
# ============================================================

def detect_qr(image):

    try:

        img = image.convert(
            "RGB"
        )

        width, height = img.size

        max_side = max(
            width,
            height,
        )

        if max_side > 1800:

            scale = (
                1800 /
                max_side
            )

            img = img.resize(
                (
                    int(width * scale),
                    int(height * scale),
                ),
                Image.Resampling.LANCZOS,
            )

        frame = np.array(
            img
        )

        detector = (
            cv2.QRCodeDetector()
        )

        # 多二维码
        try:

            result = (
                detector.detectAndDecodeMulti(
                    frame
                )
            )

            if result:

                retval = result[0]
                decoded_info = result[1]

                if retval:

                    return True

                if decoded_info:

                    for value in decoded_info:

                        if (
                            value
                            and str(value).strip()
                        ):

                            return True

        except Exception:

            pass

        # 单二维码
        try:

            data, points, _ = (
                detector.detectAndDecode(
                    frame
                )
            )

            if data:

                return True

            if points is not None:

                return True

        except Exception:

            pass

    except Exception as e:

        print(
            f"⚠️ QR 检测异常: {e}"
        )

    return False


# ============================================================
# 11. OCR 图片
# ============================================================

def prepare_ocr_images(
    image
):

    result = []

    try:

        img = image.convert(
            "RGB"
        )

        width, height = img.size

        max_side = max(
            width,
            height,
        )

        # OCR 最大边限制 1800
        if max_side > 1800:

            scale = (
                1800 /
                max_side
            )

            img = img.resize(
                (
                    int(width * scale),
                    int(height * scale),
                ),
                Image.Resampling.LANCZOS,
            )

        # 原图
        result.append(
            (
                "ORIGINAL",
                img.copy()
            )
        )

        # 灰度 + 增强
        gray = ImageOps.grayscale(
            img
        )

        enhanced = (
            ImageEnhance.Contrast(
                gray
            ).enhance(1.8)
        )

        enhanced = (
            ImageEnhance.Sharpness(
                enhanced
            ).enhance(1.5)
        )

        result.append(
            (
                "ENHANCED",
                enhanced
            )
        )

        # 二值化
        threshold = enhanced.point(
            lambda p:
                255
                if p > 150
                else 0
        )

        result.append(
            (
                "THRESHOLD",
                threshold
            )
        )

    except Exception as e:

        print(
            f"⚠️ OCR 预处理失败: {e}"
        )

    return result


def ocr_image(
    image
):

    all_text = []

    phone_hits = 0

    variants = (
        prepare_ocr_images(
            image
        )
    )

    for (
        variant_name,
        variant_image,
    ) in variants:

        for config in OCR_CONFIGS:

            try:

                text = (
                    pytesseract.image_to_string(
                        variant_image,
                        lang=OCR_LANG,
                        config=config,
                    )
                )

            except Exception:

                continue

            if not text:
                continue

            text = text.strip()

            if not text:
                continue

            all_text.append(
                text
            )

            contacts = detect_contacts(
                text
            )

            # 强联系方式
            strong = [
                x
                for x in contacts
                if x["type"]
                in STRONG_CONTACT_TYPES
            ]

            if strong:

                return (
                    "\n".join(
                        all_text
                    ),
                    strong,
                )

            # 手机号
            phone_contacts = [
                x
                for x in contacts
                if x["type"]
                in PHONE_CONTACT_TYPES
            ]

            if phone_contacts:

                phone_hits += 1

                # 两次 OCR 识别到手机号
                # 才判定为联系方式
                if phone_hits >= 2:

                    combined = (
                        "\n".join(
                            all_text
                        )
                    )

                    return (
                        combined,
                        detect_contacts(
                            combined
                        ),
                    )

    return (
        "\n".join(
            all_text
        ),
        [],
    )


# ============================================================
# 12. 图片检测
# ============================================================

def inspect_image(
    image
):

    # 第一层：二维码
    if detect_qr(image):

        return {
            "blocked": True,
            "reason": "QR_CODE",
            "ocr_text": "",
            "contacts": [],
        }

    # 第二层：OCR
    ocr_text, contacts = (
        ocr_image(
            image
        )
    )

    if contacts:

        return {
            "blocked": True,
            "reason":
                "CONTACT_IN_IMAGE",
            "ocr_text":
                ocr_text,
            "contacts":
                contacts,
        }

    return {
        "blocked": False,
        "reason": "",
        "ocr_text":
            ocr_text,
        "contacts": [],
    }


# ============================================================
# 13. 水印
# ============================================================

def find_font():

    candidates = [

        "/usr/share/fonts/opentype/noto/"
        "NotoSansCJK-Regular.ttc",

        "/usr/share/fonts/opentype/noto/"
        "NotoSansCJK-Bold.ttc",

        "/usr/share/fonts/noto-cjk/"
        "NotoSansCJK-Regular.ttc",

        "/usr/share/fonts/truetype/noto/"
        "NotoSansCJK-Regular.ttc",
    ]

    for path in candidates:

        if os.path.exists(path):

            return path

    return None


def add_watermark(
    image
):

    try:

        img = image.convert(
            "RGBA"
        )

        width, height = img.size

        overlay = Image.new(
            "RGBA",
            img.size,
            (0, 0, 0, 0),
        )

        draw = ImageDraw.Draw(
            overlay
        )

        font_path = find_font()

        font_size = max(
            22,
            min(
                width,
                height
            ) // 22,
        )

        if font_path:

            try:

                font = (
                    ImageFont.truetype(
                        font_path,
                        font_size,
                    )
                )

            except Exception:

                font = (
                    ImageFont.load_default()
                )

        else:

            font = (
                ImageFont.load_default()
            )

        text = ""

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        text_height = (
            bbox[3] - bbox[1]
        )

        margin = max(
            12,
            width // 80,
        )

        x = (
            width
            - text_width
            - margin
        )

        y = (
            height
            - text_height
            - margin
        )

        padding = max(
            5,
            font_size // 8,
        )

        draw.rounded_rectangle(
            [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding,
            ],
            radius=8,
            fill=(
                0,
                0,
                0,
                120,
            ),
        )

        draw.text(
            (x, y),
            text,
            font=font,
            fill=(
                255,
                255,
                255,
                210,
            ),
        )

        result = (
            Image.alpha_composite(
                img,
                overlay,
            )
        )

        return result.convert(
            "RGB"
        )

    except Exception as e:

        print(
            f"⚠️ 水印失败: {e}"
        )

        return image.convert(
            "RGB"
        )


def image_to_bytes(
    image
):

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=90,
        optimize=True,
    )

    output.seek(0)

    return output.getvalue()


# ============================================================
# 14. Telegram 登录
# ============================================================

async def connect_telegram():

    global client

    print()
    print(
        "CONNECTING TELEGRAM..."
    )

    if not API_ID:

        raise RuntimeError(
            "API_ID 未设置"
        )

    if not API_HASH:

        raise RuntimeError(
            "API_HASH 未设置"
        )

    if not TELEGRAM_SESSION:

        raise RuntimeError(
            "TELEGRAM_SESSION 未设置"
        )

    client = TelegramClient(
        StringSession(
            TELEGRAM_SESSION
        ),
        API_ID,
        API_HASH,
    )

    await client.connect()

    if not await client.is_user_authorized():

        raise RuntimeError(
            "TELEGRAM_SESSION 无效"
        )

    print(
        "✅ TELEGRAM CONNECTED"
    )

    me = await client.get_me()

    username = getattr(
        me,
        "username",
        None,
    )

    if username:

        print(
            f"TELEGRAM ACCOUNT: "
            f"@{username}"
        )

    else:

        print(
            f"TELEGRAM ACCOUNT: "
            f"{me.id}"
        )


# ============================================================
# 15. 目标频道
# ============================================================

async def resolve_target_entity():

    target = normalize_channel(
        TARGET_CHANNEL_RAW
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TARGET ENTITY:"
    )

    entity = await client.get_entity(
        target
    )

    print(
        entity
    )

    title = getattr(
        entity,
        "title",
        None,
    )

    entity_id = getattr(
        entity,
        "id",
        None,
    )

    username = getattr(
        entity,
        "username",
        None,
    )

    print(
        f"TARGET TITLE: "
        f"{title}"
    )

    print(
        f"TARGET ID: "
        f"{entity_id}"
    )

    print(
        f"TARGET USERNAME: "
        f"{('@' + username) if username else None}"
    )

    print(
        "=" * 70
    )

    return entity


# ============================================================
# 16. Bot 诊断
# ============================================================

async def diagnose_bot_target(
    target_entity
):

    global bot
    global BOT_TARGET_CHANNEL

    print()
    print(
        "=" * 70
    )

    print(
        "INITIALIZING BOT..."
    )

    print(
        "=" * 70
    )

    bot = Bot(
        token=BOT_TOKEN
    )

    await bot.initialize()

    print(
        "✅ BOT INITIALIZED"
    )

    print(
        "=" * 70
    )

    print(
        "🤖 BOT TARGET DIAGNOSTIC"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    print()
    print(
        "🔎 1. CHECK BOT TOKEN..."
    )

    try:

        bot_me = await bot.get_me()

        print(
            "   ✅ BOT TOKEN VALID"
        )

        print(
            f"   BOT ID: "
            f"{bot_me.id}"
        )

        print(
            f"   BOT USERNAME: "
            f"@{bot_me.username}"
        )

        print(
            f"   BOT NAME: "
            f"{bot_me.first_name}"
        )

    except Exception as e:

        print(
            "   ❌ BOT TOKEN INVALID"
        )

        print(
            f"   {type(e).__name__}: "
            f"{e}"
        )

        raise RuntimeError(
            "BOT_TOKEN 无效"
        )

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    print()
    print(
        "🔎 2. TELETHON TARGET INFO..."
    )

    title = getattr(
        target_entity,
        "title",
        None,
    )

    entity_id = getattr(
        target_entity,
        "id",
        None,
    )

    username = getattr(
        target_entity,
        "username",
        None,
    )

    print(
        f"   TITLE: {title}"
    )

    print(
        f"   TELETHON ID: "
        f"{entity_id}"
    )

    print(
        f"   USERNAME: "
        f"{('@' + username) if username else None}"
    )

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    print()
    print(
        "🔎 3. GENERATE BOT TARGET..."
    )

    if username:

        BOT_TARGET_CHANNEL = (
            "@" + username
        )

        print(
            "   ✅ PUBLIC CHANNEL"
        )

        print(
            f"   BOT TARGET: "
            f"{BOT_TARGET_CHANNEL}"
        )

    else:

        marked_peer_id = (
            utils.get_peer_id(
                target_entity
            )
        )

        BOT_TARGET_CHANNEL = int(
            marked_peer_id
        )

        print(
            "   ✅ PRIVATE CHANNEL"
        )

        print(
            f"   BOT TARGET: "
            f"{BOT_TARGET_CHANNEL}"
        )

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    print()
    print(
        "🔎 4. BOT GET CHAT..."
    )

    try:

        chat = await bot.get_chat(
            chat_id=BOT_TARGET_CHANNEL
        )

        print(
            "   ✅ BOT CAN ACCESS TARGET"
        )

        print(
            f"   CHAT ID: "
            f"{chat.id}"
        )

        print(
            f"   CHAT TYPE: "
            f"{chat.type}"
        )

        print(
            f"   CHAT TITLE: "
            f"{getattr(chat, 'title', None)}"
        )

        print(
            f"   CHAT USERNAME: "
            f"{getattr(chat, 'username', None)}"
        )

    except Exception as e:

        print(
            "   ❌ BOT GET CHAT FAILED"
        )

        print(
            f"   {type(e).__name__}: "
            f"{e}"
        )

        raise RuntimeError(
            "Bot 无法访问目标频道"
        )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    print()
    print(
        "🔎 5. CHECK BOT MEMBERSHIP..."
    )

    try:

        member = (
            await bot.get_chat_member(
                chat_id=BOT_TARGET_CHANNEL,
                user_id=bot_me.id,
            )
        )

        print(
            "   ✅ BOT MEMBERSHIP CHECK PASSED"
        )

        print(
            f"   STATUS: "
            f"{member.status}"
        )

        if (
            member.status
            == "administrator"
        ):

            can_post = getattr(
                member,
                "can_post_messages",
                None,
            )

            if can_post is False:

                raise RuntimeError(
                    "Bot 是频道管理员，但没有 "
                    "can_post_messages 权限"
                )

            print(
                "   ✅ BOT CAN POST MESSAGES"
            )

        elif (
            member.status
            == "creator"
        ):

            print(
                "   ✅ BOT IS CHANNEL CREATOR"
            )

        else:

            print(
                f"   ⚠️ BOT STATUS: "
                f"{member.status}"
            )

            print(
                "   ⚠️ 继续执行，"
                "最终以实际发送结果判断"
            )

    except TelegramError as e:

        error_text = str(e)

        print(
            "   ⚠️ BOT MEMBERSHIP CHECK UNAVAILABLE"
        )

        print(
            f"   {type(e).__name__}: "
            f"{error_text}"
        )

        if (
            "Member list is inaccessible"
            in error_text
        ):

            print(
                "   ⚠️ Telegram 不允许当前 Bot "
                "查询该频道成员列表"
            )

            print(
                "   ⚠️ 这不代表 Bot 没有发布权限"
            )

            print(
                "   ⚠️ 跳过成员检查，继续执行"
            )

        else:

            print(
                "   ⚠️ 跳过成员检查，继续执行"
            )

    except RuntimeError:

        raise

    except Exception as e:

        print(
            "   ⚠️ BOT MEMBERSHIP CHECK ERROR"
        )

        print(
            f"   {type(e).__name__}: "
            f"{e}"
        )

        print(
            "   ⚠️ 跳过成员检查，继续执行"
        )

    print()
    print(
        "   ✅ BOT TARGET DIAGNOSTIC CONTINUE"
    )

    print(
        "   ⚠️ 实际发送结果将最终确认 Bot 发布权限"
    )

    print(
        "=" * 70
    )


# ============================================================
# 17. 下载图片
# ============================================================

async def download_message_image(
    message
):

    try:

        data = io.BytesIO()

        result = (
            await client.download_media(
                message,
                file=data,
            )
        )

        if result is None:

            return None

        raw = data.getvalue()

        if not raw:

            return None

        image = Image.open(
            io.BytesIO(raw)
        )

        image.load()

        return image.convert(
            "RGB"
        )

    except Exception as e:

        print(
            f"⚠️ 图片下载失败 "
            f"message={message.id}: "
            f"{e}"
        )

        return None


# ============================================================
# 18. Bot 发送文字
# ============================================================

async def bot_send_text(
    text,
    retries=3,
):

    if not text:

        return False

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            await bot.send_message(
                chat_id=BOT_TARGET_CHANNEL,
                text=text,
                disable_web_page_preview=True,
            )

            print(
                "   ✅ BOT TEXT SENT"
            )

            return True

        except Exception as e:

            print(
                f"   ⚠️ BOT TEXT RETRY "
                f"{attempt}/{retries}: "
                f"{e}"
            )

            if attempt < retries:

                await asyncio.sleep(
                    2
                )

    print(
        "   ❌ BOT TEXT SEND FAILED"
    )

    return False


# ============================================================
# 19. Bot 单图
# ============================================================

async def bot_send_photo(
    image_bytes,
    caption=None,
    retries=3,
):

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            # 每次重试都创建新的 BytesIO
            image_stream = io.BytesIO(
                image_bytes
            )

            image_stream.seek(0)

            photo_file = InputFile(
                image_stream,
                filename="gjp_single.jpg",
            )

            await bot.send_photo(
                chat_id=BOT_TARGET_CHANNEL,
                photo=photo_file,
                caption=(
                    caption[:1024]
                    if caption
                    else None
                ),
            )

            print(
                "   ✅ BOT PHOTO SENT"
            )

            return True

        except Exception as e:

            print(
                f"   ⚠️ BOT PHOTO RETRY "
                f"{attempt}/{retries}: "
                f"{e}"
            )

            if attempt < retries:

                await asyncio.sleep(
                    2
                )

    print(
        "   ❌ BOT PHOTO SEND FAILED"
    )

    return False


# ============================================================
# 20. Bot 相册
# ============================================================
#
# 这里是本次重点修复部分。
#
# 原问题：
#
#   Can't parse inputmedia: media not found
#
# 主要原因是 multipart media 处理时，
# 内存文件没有使用明确、独立的媒体引用。
#
# 现在：
#
#   1. 每张图片创建独立 BytesIO
#   2. 每张图片唯一 filename
#   3. 每次重试重新创建 media
#   4. 每 10 张拆分
#
# ============================================================

async def bot_send_album(
    items,
    retries=3,
):

    if not items:

        return False

    # Telegram MediaGroup 最大 10 张
    chunks = [
        items[i:i + 10]
        for i in range(
            0,
            len(items),
            10,
        )
    ]

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"   📦 SEND ALBUM "
            f"{chunk_index}/"
            f"{len(chunks)} "
            f"({len(chunk)} images)"
        )

        success = False

        for attempt in range(
            1,
            retries + 1,
        ):

            try:

                media = []

                # ------------------------------------------------
                # 每一次重试都重新创建整个 media 数组
                # ------------------------------------------------

                for image_index, item in enumerate(
                    chunk,
                    start=1,
                ):

                    image_bytes = item[
                        "image_bytes"
                    ]

                    if not image_bytes:

                        raise RuntimeError(
                            f"相册第 "
                            f"{image_index} 张图片为空"
                        )

                    # 独立 BytesIO
                    image_stream = (
                        io.BytesIO(
                            image_bytes
                        )
                    )

                    image_stream.seek(0)

                    # 每张图片独立文件名
                    filename = (
                        f"gjp_album_"
                        f"{chunk_index}_"
                        f"{image_index}_"
                        f"{item.get('message_id', image_index)}.jpg"
                    )

                    photo_file = InputFile(
                        image_stream,
                        filename=filename,
                    )

                    caption = item.get(
                        "caption"
                    )

                    # Telegram 相册：
                    # caption 只给有内容的图片
                    media_item = (
                        InputMediaPhoto(
                            media=photo_file,
                            caption=(
                                caption[:1024]
                                if caption
                                else None
                            ),
                        )
                    )

                    media.append(
                        media_item
                    )

                # ------------------------------------------------
                # 必须确保 1~10 张
                # ------------------------------------------------

                if not media:

                    raise RuntimeError(
                        "相册 media 为空"
                    )

                if len(media) > 10:

                    raise RuntimeError(
                        "单个 Telegram 相册超过 10 张"
                    )

                # ------------------------------------------------
                # 发送
                # ------------------------------------------------

                await bot.send_media_group(
                    chat_id=BOT_TARGET_CHANNEL,
                    media=media,
                )

                print(
                    "   ✅ ALBUM SENT"
                )

                success = True

                break

            except Exception as e:

                print(
                    f"   ⚠️ BOT ALBUM RETRY "
                    f"{attempt}/{retries}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                if attempt < retries:

                    await asyncio.sleep(
                        3
                    )

        if not success:

            print(
                "   ❌ ALBUM SEND FAILED"
            )

            return False

    return True


# ============================================================
# 21. 单图处理
# ============================================================

async def process_image_message(
    message
):

    caption = (
        message.message
        if getattr(
            message,
            "message",
            None,
        )
        else ""
    )

    # --------------------------------------------------------
    # Caption 先检测
    # --------------------------------------------------------

    if caption:

        caption_contacts = (
            detect_contacts(
                caption
            )
        )

        if caption_contacts:

            print(
                "   🚫 IMAGE SKIPPED: "
                "CONTACT_IN_CAPTION"
            )

            print(
                "   CONTACTS: "
                + format_contacts(
                    caption_contacts
                )
            )

            return {
                "status": "blocked",
                "reason":
                    "CONTACT_IN_CAPTION",
            }

    # --------------------------------------------------------
    # 下载
    # --------------------------------------------------------

    image = (
        await download_message_image(
            message
        )
    )

    if image is None:

        return {
            "status": "failed",
            "reason":
                "DOWNLOAD_FAILED",
        }

    # --------------------------------------------------------
    # 二维码 + OCR
    # --------------------------------------------------------

    inspection = (
        inspect_image(
            image
        )
    )

    if inspection["blocked"]:

        print(
            f"   🚫 IMAGE SKIPPED: "
            f"{inspection['reason']}"
        )

        if inspection.get(
            "contacts"
        ):

            print(
                "   CONTACTS: "
                + format_contacts(
                    inspection[
                        "contacts"
                    ]
                )
            )

        return {
            "status": "blocked",
            "reason":
                inspection[
                    "reason"
                ],
        }

    # --------------------------------------------------------
    # 加水印
    # --------------------------------------------------------

    watermarked = (
        add_watermark(
            image
        )
    )

    image_bytes = (
        image_to_bytes(
            watermarked
        )
    )

    return {
        "status": "ready",
        "image_bytes":
            image_bytes,
        "caption":
            caption or None,
    }


# ============================================================
# 22. 获取北京时间当天消息
# ============================================================

async def collect_today_messages():

    source_channels = (
        parse_source_channels()
    )

    if not source_channels:

        raise RuntimeError(
            "SOURCE_CHANNELS 未设置"
        )

    now = now_beijing()

    today_start = datetime(
        now.year,
        now.month,
        now.day,
        tzinfo=BEIJING_TZ,
    )

    tomorrow_start = (
        today_start
        + timedelta(days=1)
    )

    print()
    print(
        "=" * 70
    )

    print(
        "📥 COLLECTING TODAY'S MESSAGES"
    )

    print(
        "=" * 70
    )

    print(
        f"BEIJING DATE: "
        f"{today_start.date()}"
    )

    all_messages = []

    for (
        source_index,
        source,
    ) in enumerate(
        source_channels
    ):

        print()
        print(
            f"🔎 SOURCE #{source_index + 1}: "
            f"{source}"
        )

        try:

            entity = (
                await client.get_entity(
                    source
                )
            )

            title = getattr(
                entity,
                "title",
                None,
            )

            print(
                f"   TITLE: {title}"
            )

            count = 0

            async for message in (
                client.iter_messages(
                    entity,
                    limit=(
                        SCAN_LIMIT
                        if SCAN_LIMIT > 0
                        else None
                    ),
                )
            ):

                if not message:
                    continue

                message_date = (
                    message.date
                )

                if not message_date:
                    continue

                if (
                    message_date.tzinfo
                    is None
                ):

                    message_date = (
                        message_date.replace(
                            tzinfo=timezone.utc
                        )
                    )

                beijing_date = (
                    message_date.astimezone(
                        BEIJING_TZ
                    )
                )

                # 已经到昨天
                if (
                    beijing_date
                    < today_start
                ):

                    break

                if (
                    beijing_date
                    >= tomorrow_start
                ):

                    continue

                all_messages.append(
                    {
                        "source_index":
                            source_index,

                        "source":
                            source,

                        "entity":
                            entity,

                        "message":
                            message,

                        "date":
                            beijing_date,
                    }
                )

                count += 1

            print(
                f"   ✅ TODAY MESSAGES: "
                f"{count}"
            )

        except Exception as e:

            print(
                f"   ❌ SOURCE FAILED: "
                f"{type(e).__name__}: "
                f"{e}"
            )

    # --------------------------------------------------------
    # 排序
    # --------------------------------------------------------

    all_messages.sort(
        key=lambda x: (
            x["date"],
            x["source_index"],
            x["message"].id,
        )
    )

    print()
    print(
        f"TOTAL TODAY MESSAGES: "
        f"{len(all_messages)}"
    )

    return all_messages


# ============================================================
# 23. 普通消息处理
# ============================================================

async def process_normal_message(
    item,
    processed,
):

    source_index = (
        item["source_index"]
    )

    source = item[
        "source"
    ]

    message = item[
        "message"
    ]

    key = message_key(
        source_index,
        message.id,
    )

    # --------------------------------------------------------
    # 去重
    # --------------------------------------------------------

    if is_already_processed(
        processed,
        source_index,
        message.id,
    ):

        print(
            f"⏭️ SKIP PROCESSED "
            f"[{source}] "
            f"message={message.id}"
        )

        return False

    text = (
        message.message
        if getattr(
            message,
            "message",
            None,
        )
        else ""
    )

    # --------------------------------------------------------
    # 图片
    # --------------------------------------------------------

    if message.photo:

        print()
        print(
            f"🖼️ IMAGE "
            f"[{source}] "
            f"message={message.id}"
        )

        result = (
            await process_image_message(
                message
            )
        )

        if result["status"] == "blocked":

            processed.add(
                key
            )

            return True

        if result["status"] != "ready":

            print(
                "   ⚠️ IMAGE PROCESS FAILED"
            )

            return False

        success = (
            await bot_send_photo(
                result[
                    "image_bytes"
                ],
                result.get(
                    "caption"
                ),
            )
        )

        if success:

            processed.add(
                key
            )

            print(
                "   ✅ IMAGE PROCESSED"
            )

            return True

        return False

    # --------------------------------------------------------
    # 文字
    # --------------------------------------------------------

    if text:

        contacts = (
            detect_contacts(
                text
            )
        )

        if contacts:

            print()
            print(
                f"🚫 TEXT SKIPPED "
                f"[{source}] "
                f"message={message.id}"
            )

            print(
                "   CONTACTS: "
                + format_contacts(
                    contacts
                )
            )

            processed.add(
                key
            )

            return True

        print()
        print(
            f"📝 TEXT "
            f"[{source}] "
            f"message={message.id}"
        )

        success = (
            await bot_send_text(
                text
            )
        )

        if success:

            processed.add(
                key
            )

            return True

        return False

    # --------------------------------------------------------
    # 其他
    # --------------------------------------------------------

    print()
    print(
        f"⚠️ UNSUPPORTED MESSAGE "
        f"[{source}] "
        f"message={message.id}"
    )

    processed.add(
        key
    )

    return True


# ============================================================
# 24. 相册处理
# ============================================================

async def process_albums(
    album_groups,
    processed,
):

    if not album_groups:

        return

    print()
    print(
        "=" * 70
    )

    print(
        "📚 PROCESSING TELEGRAM ALBUMS"
    )

    print(
        "=" * 70
    )

    for (
        album_key,
        items,
    ) in album_groups.items():

        items.sort(
            key=lambda x: (
                x["date"],
                x["message"].id,
            )
        )

        source_index = (
            items[0][
                "source_index"
            ]
        )

        source = items[0][
            "source"
        ]

        print()
        print(
            f"📚 ALBUM "
            f"{album_key} "
            f"[{source}] "
            f"MESSAGES={len(items)}"
        )

        valid_items = []

        for item in items:

            message = item[
                "message"
            ]

            key = message_key(
                source_index,
                message.id,
            )

            # ------------------------------------------------
            # 已处理
            # ------------------------------------------------

            if is_already_processed(
                processed,
                source_index,
                message.id,
            ):

                print(
                    f"   ⏭️ SKIP PROCESSED "
                    f"message={message.id}"
                )

                continue

            # ------------------------------------------------
            # 图片独立检测
            # ------------------------------------------------

            result = (
                await process_image_message(
                    message
                )
            )

            # ------------------------------------------------
            # 联系方式 / QR
            # ------------------------------------------------

            if (
                result["status"]
                == "blocked"
            ):

                processed.add(
                    key
                )

                print(
                    f"   🚫 FILTERED "
                    f"message={message.id}"
                )

                continue

            # ------------------------------------------------
            # 下载/处理失败
            # ------------------------------------------------

            if (
                result["status"]
                != "ready"
            ):

                print(
                    f"   ❌ PROCESS FAILED "
                    f"message={message.id}"
                )

                continue

            # ------------------------------------------------
            # 合格
            # ------------------------------------------------

            valid_items.append(
                {
                    "key":
                        key,

                    "message_id":
                        message.id,

                    "image_bytes":
                        result[
                            "image_bytes"
                        ],

                    "caption":
                        result.get(
                            "caption"
                        ),
                }
            )

        # ----------------------------------------------------
        # 全部被过滤
        # ----------------------------------------------------

        if not valid_items:

            print(
                "   ℹ️ NO VALID IMAGES"
            )

            continue

        # ----------------------------------------------------
        # 发送相册
        # ----------------------------------------------------

        success = (
            await bot_send_album(
                valid_items
            )
        )

        # ----------------------------------------------------
        # 成功才写入 processed
        # ----------------------------------------------------

        if success:

            for item in valid_items:

                processed.add(
                    item["key"]
                )

            print(
                f"   ✅ ALBUM PROCESSED "
                f"({len(valid_items)} images)"
            )

        else:

            print(
                "   ❌ ALBUM PUBLISH FAILED"
            )

            print(
                "   ⚠️ 本次不会写入 processed.json"
            )


# ============================================================
# 25. 全部消息处理
# ============================================================

async def process_messages(
    all_messages,
    processed,
):

    print()
    print(
        "=" * 70
    )

    print(
        "🔄 PROCESSING MESSAGES"
    )

    print(
        "=" * 70
    )

    album_groups = (
        defaultdict(list)
    )

    normal_messages = []

    # --------------------------------------------------------
    # 分离相册
    # --------------------------------------------------------

    for item in all_messages:

        message = item[
            "message"
        ]

        grouped_id = getattr(
            message,
            "grouped_id",
            None,
        )

        if (
            grouped_id
            and message.photo
        ):

            album_key = (
                item[
                    "source_index"
                ],
                str(
                    grouped_id
                ),
            )

            album_groups[
                album_key
            ].append(
                item
            )

        else:

            normal_messages.append(
                item
            )

    # --------------------------------------------------------
    # 普通消息
    # --------------------------------------------------------

    total_normal = len(
        normal_messages
    )

    for index, item in enumerate(
        normal_messages,
        start=1,
    ):

        print()
        print(
            f"📊 NORMAL "
            f"{index}/{total_normal}"
        )

        try:

            await process_normal_message(
                item,
                processed,
            )

        except Exception as e:

            source = item[
                "source"
            ]

            message = item[
                "message"
            ]

            print(
                f"❌ MESSAGE ERROR "
                f"[{source}] "
                f"message={message.id}: "
                f"{type(e).__name__}: "
                f"{e}"
            )

    # --------------------------------------------------------
    # 相册
    # --------------------------------------------------------

    await process_albums(
        album_groups,
        processed,
    )


# ============================================================
# 26. 主程序
# ============================================================

async def main():

    global bot

    print(
        "=" * 70
    )

    print(
        "TELEGRAM AUTO FORWARDER"
    )

    print(
        "=" * 70
    )

    print(
        f"BEIJING TIME: "
        f"{now_beijing()}"
    )

    source_channels = (
        parse_source_channels()
    )

    print(
        f"SOURCE CHANNELS: "
        f"{source_channels}"
    )

    print(
        f"TARGET CHANNEL RAW: "
        f"{TARGET_CHANNEL_RAW}"
    )

    print(
        f"TARGET CHANNEL NORMALIZED: "
        f"{normalize_channel(TARGET_CHANNEL_RAW)}"
    )

    print(
        f"SCAN LIMIT: "
        f"{SCAN_LIMIT}"
    )

    # --------------------------------------------------------
    # processed
    # --------------------------------------------------------

    processed = load_processed()

    print(
        f"PROCESSED COUNT: "
        f"{len(processed)}"
    )

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    await connect_telegram()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_entity = (
        await resolve_target_entity()
    )

    # --------------------------------------------------------
    # Bot
    # --------------------------------------------------------

    await diagnose_bot_target(
        target_entity
    )

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    all_messages = (
        await collect_today_messages()
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    await process_messages(
        all_messages,
        processed,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_processed(
        processed
    )

    print()
    print(
        "=" * 70
    )

    print(
        "✅ ALL TASKS COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"PROCESSED COUNT: "
        f"{len(processed)}"
    )

    print(
        f"BEIJING TIME END: "
        f"{now_beijing()}"
    )

    print(
        "=" * 70
    )


# ============================================================
# 27. 清理
# ============================================================

async def cleanup():

    global bot
    global client

    try:

        if bot:

            await bot.shutdown()

    except Exception:

        pass

    try:

        if client:

            await client.disconnect()

    except Exception:

        pass


# ============================================================
# 28. 程序入口
# ============================================================

if __name__ == "__main__":

    async def runner():

        try:

            await main()

        finally:

            await cleanup()

    try:

        asyncio.run(
            runner()
        )

    except KeyboardInterrupt:

        print(
            "⚠️ PROGRAM INTERRUPTED"
        )

    except Exception as e:

        print()
        print(
            "=" * 70
        )

        print(
            "❌ FATAL ERROR"
        )

        print(
            "=" * 70
        )

        print(
            f"{type(e).__name__}: "
            f"{e}"
        )

        print(
            "=" * 70
        )

        raise
