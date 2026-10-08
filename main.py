# -*- coding: utf-8 -*-

import asyncio
import io
import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont

from telethon import TelegramClient, utils
from telethon.sessions import StringSession

from telegram import (
    Bot,
    InputFile,
    InputMediaPhoto,
)
from telegram.error import TelegramError


# ============================================================
# 基础配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_FILE = BASE_DIR / "processed.json"

API_ID = int(os.environ.get("API_ID", "0"))
API_HASH = os.environ.get("API_HASH", "").strip()
TELEGRAM_SESSION = os.environ.get("TELEGRAM_SESSION", "").strip()

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()

SOURCE_CHANNELS_RAW = os.environ.get("SOURCE_CHANNELS", "").strip()
TARGET_CHANNEL_RAW = os.environ.get("TARGET_CHANNEL", "").strip()

SCAN_LIMIT_RAW = os.environ.get("SCAN_LIMIT", "0").strip()

try:
    SCAN_LIMIT = int(SCAN_LIMIT_RAW or "0")
except Exception:
    SCAN_LIMIT = 0


# 北京时间 UTC+8
BEIJING_TZ = timezone(timedelta(hours=8))


# ============================================================
# OCR 配置
# ============================================================

OCR_LANG = "eng+chi_sim+vie+por"

OCR_CONFIGS = [
    "--psm 6",
    "--psm 11",
    "--psm 12",
]


# ============================================================
# 联系方式识别
# ============================================================

CONTACT_PATTERNS = [

    (
        "TELEGRAM_LINK",
        re.compile(
            r"(?:https?://)?(?:www\.)?"
            r"(?:t\.me|telegram\.me)/"
            r"[A-Za-z0-9_+/=?-]{4,64}",
            re.IGNORECASE,
        ),
    ),

    (
        "TELEGRAM_USERNAME",
        re.compile(
            r"@[A-Za-z][A-Za-z0-9_]{4,31}"
        ),
    ),

    (
        "WHATSAPP_LINK",
        re.compile(
            r"(?:https?://)?(?:www\.)?"
            r"wa\.me/\d{7,15}",
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
            r"(微信|微[信號号]|微信号)",
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
            r"(?:0?3|0?5|0?7|0?8|0?9)\d{8}"
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
# 全局变量
# ============================================================

BOT_TARGET_CHANNEL = None

client = None
bot = None


# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


# ============================================================
# 工具函数
# ============================================================

def now_beijing():
    return datetime.now(BEIJING_TZ)


def normalize_channel(value):
    value = (value or "").strip()

    if not value:
        return ""

    value = value.strip()

    if value.startswith("https://t.me/"):
        value = value[len("https://t.me/"):]

    elif value.startswith("http://t.me/"):
        value = value[len("http://t.me/"):]

    elif value.startswith("https://telegram.me/"):
        value = value[len("https://telegram.me/"):]

    elif value.startswith("http://telegram.me/"):
        value = value[len("http://telegram.me/"):]

    value = value.rstrip("/")

    if value.startswith("@"):
        return value

    if value.startswith("-100"):
        return value

    if re.fullmatch(r"-?\d+", value):
        return value

    return "@" + value


def parse_source_channels():
    if not SOURCE_CHANNELS_RAW:
        return []

    result = []

    for item in SOURCE_CHANNELS_RAW.split(","):
        item = normalize_channel(item)

        if item:
            result.append(item)

    return result


def load_processed():
    if not PROCESSED_FILE.exists():
        return set()

    try:
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data.get("messages", [])

        if not isinstance(messages, list):
            return set()

        return {str(x) for x in messages}

    except Exception as e:
        print(f"⚠️ 读取 processed.json 失败: {e}")
        return set()


def save_processed(processed):
    normalized = {str(x) for x in processed}

    def sort_key(value):
        try:
            return (0, int(value))
        except Exception:
            return (1, str(value))

    sorted_messages = sorted(
        normalized,
        key=sort_key,
    )

    data = {
        "messages": sorted_messages[-10000:]
    }

    tmp_file = PROCESSED_FILE.with_suffix(".tmp")

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp_file.replace(PROCESSED_FILE)


def message_key(source_index, message_id):
    """
    多源频道必须把 source_index 加进去，
    避免不同频道存在相同 message.id 导致误判重复。
    """
    return f"{source_index}:{message_id}"


def legacy_message_key(message_id):
    """
    兼容旧版本 processed.json。
    """
    return str(message_id)


def is_already_processed(processed, source_index, message_id):
    key = message_key(source_index, message_id)

    if key in processed:
        return True

    # 兼容以前只保存 message.id 的版本
    if legacy_message_key(message_id) in processed:
        return True

    return False


# ============================================================
# 联系方式识别
# ============================================================

def detect_contacts(text):
    if not text:
        return []

    found = []

    for contact_type, pattern in CONTACT_PATTERNS:
        try:
            matches = pattern.findall(text)

            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(match)

                    found.append(
                        {
                            "type": contact_type,
                            "value": str(match),
                        }
                    )

        except Exception:
            continue

    return found


def has_contact(text):
    contacts = detect_contacts(text)

    return len(contacts) > 0


def format_contacts(contacts):
    if not contacts:
        return ""

    return ", ".join(
        f"{x['type']}={x['value']}"
        for x in contacts
    )


# ============================================================
# QR CODE
# ============================================================

def detect_qr(image):
    """
    检测图片中是否存在二维码。
    只要检测到二维码，就认为该图片不允许发布。
    """

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        img = np.array(image)

        detector = cv2.QRCodeDetector()

        # 多二维码
        try:
            result = detector.detectAndDecodeMulti(img)

            if result and len(result) >= 2:
                retval = result[0]
                decoded_info = result[1]

                if retval:
                    return True

                if decoded_info:
                    for value in decoded_info:
                        if value and str(value).strip():
                            return True

        except Exception:
            pass

        # 单二维码
        try:
            data, points, _ = detector.detectAndDecode(img)

            if data and str(data).strip():
                return True

            if points is not None:
                return True

        except Exception:
            pass

    except Exception as e:
        print(f"⚠️ QR 检测异常: {e}")

    return False


# ============================================================
# OCR 图片处理
# ============================================================

def prepare_ocr_images(image):
    images = []

    try:
        img = image.convert("RGB")

        images.append(
            ("ORIGINAL", img.copy())
        )

        gray = ImageOps.grayscale(img)

        images.append(
            ("GRAYSCALE", gray.copy())
        )

        enhanced = ImageEnhance.Contrast(gray).enhance(2.2)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)

        images.append(
            ("ENHANCED", enhanced.copy())
        )

        threshold = enhanced.point(
            lambda p: 255 if p > 150 else 0
        )

        images.append(
            ("THRESHOLD", threshold.copy())
        )

        # 放大图片
        width, height = img.size

        if width < 1500:
            scale = 1500 / max(width, 1)

            new_size = (
                int(width * scale),
                int(height * scale),
            )

            enlarged = img.resize(
                new_size,
                Image.Resampling.LANCZOS,
            )

            images.append(
                ("ENLARGED", enlarged)
            )

    except Exception as e:
        print(f"⚠️ OCR 图片预处理失败: {e}")

    return images


def ocr_image(image):
    """
    对图片做多种 OCR。
    返回：
        text
        contacts
    """

    all_text = []

    try:
        variants = prepare_ocr_images(image)

        for variant_name, variant_image in variants:

            for config in OCR_CONFIGS:

                try:
                    text = pytesseract.image_to_string(
                        variant_image,
                        lang=OCR_LANG,
                        config=config,
                    )

                    if text:
                        text = text.strip()

                        if text:
                            all_text.append(text)

                            contacts = detect_contacts(text)

                            # 强联系方式：
                            # 一次识别出来即可判定
                            strong = [
                                x for x in contacts
                                if x["type"] in STRONG_CONTACT_TYPES
                            ]

                            if strong:
                                return "\n".join(all_text), contacts

                except Exception:
                    continue

    except Exception as e:
        print(f"⚠️ OCR 异常: {e}")

    combined_text = "\n".join(all_text)

    contacts = detect_contacts(combined_text)

    # 电话号码更加严格：
    # 至少两个 OCR 结果中出现联系方式
    phone_values = []

    for text in all_text:
        contacts = detect_contacts(text)

        for contact in contacts:
            if contact["type"] in PHONE_CONTACT_TYPES:
                phone_values.append(
                    contact["value"]
                )

    if len(phone_values) >= 2:
        contacts = detect_contacts(combined_text)

        return combined_text, contacts

    # 如果只是一次 OCR 识别出的电话号码，
    # 不直接判定，降低误杀
    non_phone_contacts = [
        x
        for x in detect_contacts(combined_text)
        if x["type"] not in PHONE_CONTACT_TYPES
    ]

    if non_phone_contacts:
        return combined_text, non_phone_contacts

    return combined_text, []


# ============================================================
# 图片联系方式总检测
# ============================================================

def inspect_image(image):
    """
    返回：

    {
        "blocked": True/False,
        "reason": "...",
        "ocr_text": "...",
        "contacts": [...]
    }
    """

    # ----------------------------
    # 1. QR
    # ----------------------------

    if detect_qr(image):

        return {
            "blocked": True,
            "reason": "QR_CODE",
            "ocr_text": "",
            "contacts": [],
        }

    # ----------------------------
    # 2. OCR
    # ----------------------------

    ocr_text, contacts = ocr_image(image)

    if contacts:

        return {
            "blocked": True,
            "reason": "CONTACT_IN_IMAGE",
            "ocr_text": ocr_text,
            "contacts": contacts,
        }

    return {
        "blocked": False,
        "reason": "",
        "ocr_text": ocr_text,
        "contacts": [],
    }


# ============================================================
# 水印
# ============================================================

def find_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]

    for path in candidates:

        if os.path.exists(path):
            return path

    return None


def add_watermark(image):
    """
    添加：
        85H官方频道
    """

    try:
        img = image.convert("RGBA")

        width, height = img.size

        overlay = Image.new(
            "RGBA",
            img.size,
            (0, 0, 0, 0),
        )

        draw = ImageDraw.Draw(overlay)

        font_path = find_font()

        font_size = max(
            22,
            min(width, height) // 22,
        )

        if font_path:

            try:
                font = ImageFont.truetype(
                    font_path,
                    font_size,
                )

            except Exception:
                font = ImageFont.load_default()

        else:
            font = ImageFont.load_default()

        text = "85H官方频道"

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        margin = max(
            12,
            width // 80,
        )

        x = width - text_width - margin
        y = height - text_height - margin

        # 黑色半透明背景
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
            fill=(0, 0, 0, 120),
        )

        # 白色文字
        draw.text(
            (x, y),
            text,
            font=font,
            fill=(255, 255, 255, 210),
        )

        result = Image.alpha_composite(
            img,
            overlay,
        )

        return result.convert("RGB")

    except Exception as e:

        print(f"⚠️ 添加水印失败: {e}")

        return image.convert("RGB")


def image_to_bytes(image):
    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    output.seek(0)

    return output.getvalue()


# ============================================================
# Telethon
# ============================================================

async def connect_telegram():

    global client

    print()
    print("CONNECTING TELEGRAM...")

    if not API_ID:
        raise RuntimeError("API_ID 未设置")

    if not API_HASH:
        raise RuntimeError("API_HASH 未设置")

    if not TELEGRAM_SESSION:
        raise RuntimeError("TELEGRAM_SESSION 未设置")

    client = TelegramClient(
        StringSession(TELEGRAM_SESSION),
        API_ID,
        API_HASH,
    )

    await client.connect()

    if not await client.is_user_authorized():
        raise RuntimeError(
            "TELEGRAM_SESSION 无效或已经失效"
        )

    print("✅ TELEGRAM CONNECTED")

    me = await client.get_me()

    username = getattr(me, "username", None)

    if username:
        username_text = f"@{username}"
    else:
        username_text = str(me.id)

    print(f"TELEGRAM ACCOUNT: {username_text}")


async def resolve_target_entity():

    target = normalize_channel(
        TARGET_CHANNEL_RAW
    )

    print("=" * 70)
    print("TARGET ENTITY:")

    entity = await client.get_entity(target)

    print(entity)

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

    print(f"TARGET TITLE: {title}")
    print(f"TARGET ID: {entity_id}")
    print(
        f"TARGET USERNAME: "
        f"{('@' + username) if username else None}"
    )

    print("=" * 70)

    return entity


# ============================================================
# Bot 目标诊断
# ============================================================

async def diagnose_bot_target(target_entity):

    global bot
    global BOT_TARGET_CHANNEL

    print()
    print("=" * 70)
    print("INITIALIZING BOT...")
    print("=" * 70)

    bot = Bot(
        token=BOT_TOKEN
    )

    await bot.initialize()

    print("✅ BOT INITIALIZED")

    print("=" * 70)
    print("🤖 BOT TARGET DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Bot Token
    # --------------------------------------------------------

    print()
    print("🔎 1. CHECK BOT TOKEN...")

    try:

        bot_me = await bot.get_me()

        print("   ✅ BOT TOKEN VALID")
        print(f"   BOT ID: {bot_me.id}")
        print(
            f"   BOT USERNAME: "
            f"@{bot_me.username}"
        )
        print(
            f"   BOT NAME: "
            f"{bot_me.first_name}"
        )

    except Exception as e:

        print("   ❌ BOT TOKEN INVALID")
        print(f"   {type(e).__name__}: {e}")

        raise RuntimeError(
            "BOT_TOKEN 无效"
        )

    # --------------------------------------------------------
    # 2. Telethon Target
    # --------------------------------------------------------

    print()
    print("🔎 2. TELETHON TARGET INFO...")

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

    print(f"   TITLE: {title}")
    print(f"   TELETHON ID: {entity_id}")
    print(f"   USERNAME: {('@' + username) if username else None}")

    # --------------------------------------------------------
    # 3. Generate Bot target
    # --------------------------------------------------------

    print()
    print("🔎 3. GENERATE BOT TARGET...")

    if username:

        BOT_TARGET_CHANNEL = "@" + username

        print("   ✅ PUBLIC CHANNEL")
        print(
            f"   BOT TARGET: "
            f"{BOT_TARGET_CHANNEL}"
        )

    else:

        marked_peer_id = utils.get_peer_id(
            target_entity
        )

        BOT_TARGET_CHANNEL = int(
            marked_peer_id
        )

        print("   ✅ PRIVATE CHANNEL")
        print(
            f"   BOT TARGET: "
            f"{BOT_TARGET_CHANNEL}"
        )

    # --------------------------------------------------------
    # 4. Bot getChat
    # --------------------------------------------------------

    print()
    print("🔎 4. BOT GET CHAT...")

    try:

        chat = await bot.get_chat(
            chat_id=BOT_TARGET_CHANNEL
        )

        print("   ✅ BOT CAN ACCESS TARGET")

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

        print("   ❌ BOT GET CHAT FAILED")
        print(
            f"   {type(e).__name__}: {e}"
        )

        raise RuntimeError(
            "Bot 无法访问目标频道。"
            "请检查 Bot 是否已经加入目标频道，"
            "以及 TARGET_CHANNEL 是否正确。"
        )

    # --------------------------------------------------------
    # 5. Bot membership
    #
    # 注意：
    # Telegram 某些频道会直接返回：
    #
    # Member list is inaccessible
    #
    # 这种情况不能作为 Fatal Error。
    # --------------------------------------------------------

    print()
    print("🔎 5. CHECK BOT MEMBERSHIP...")

    try:

        member = await bot.get_chat_member(
            chat_id=BOT_TARGET_CHANNEL,
            user_id=bot_me.id,
        )

        print(
            "   ✅ BOT MEMBERSHIP CHECK PASSED"
        )

        print(
            f"   STATUS: "
            f"{member.status}"
        )

        if member.status == "administrator":

            can_post = getattr(
                member,
                "can_post_messages",
                None,
            )

            if can_post is False:

                print(
                    "   ❌ BOT 没有发布消息权限"
                )

                raise RuntimeError(
                    "Bot 是频道管理员，但没有 "
                    "can_post_messages 权限"
                )

            print(
                "   ✅ BOT CAN POST MESSAGES"
            )

        elif member.status == "creator":

            print(
                "   ✅ BOT IS CHANNEL CREATOR"
            )

        elif member.status in (
            "member",
            "restricted",
        ):

            print(
                f"   ⚠️ BOT STATUS: "
                f"{member.status}"
            )

            print(
                "   ⚠️ 将继续执行，"
                "最终以实际发送结果判断权限"
            )

        elif member.status in (
            "left",
            "kicked",
        ):

            print(
                f"   ❌ BOT STATUS: "
                f"{member.status}"
            )

            raise RuntimeError(
                "Bot 不在目标频道中"
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
                "   ⚠️ 无法确认 Bot 成员状态"
            )

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
        "   ⚠️ Bot 是否真正拥有发布权限，"
        "将由实际发送结果最终确认"
    )

    print("=" * 70)


# ============================================================
# 下载 Telegram 图片
# ============================================================

async def download_message_image(message):

    try:

        data = io.BytesIO()

        result = await client.download_media(
            message,
            file=data,
        )

        if result is None:
            return None

        if isinstance(result, bytes):
            raw = result

        else:
            raw = data.getvalue()

        if not raw:
            return None

        image = Image.open(
            io.BytesIO(raw)
        )

        image.load()

        return image.convert("RGB")

    except Exception as e:

        print(
            f"⚠️ 下载图片失败 "
            f"message={message.id}: {e}"
        )

        return None


# ============================================================
# Bot 发送文字
# ============================================================

async def bot_send_text(text, retries=3):

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
                f"   ✅ BOT TEXT SENT "
                f"(attempt {attempt})"
            )

            return True

        except Exception as e:

            print(
                f"   ⚠️ BOT TEXT RETRY "
                f"{attempt}/{retries}: {e}"
            )

            if attempt < retries:
                await asyncio.sleep(2)

    print(
        "   ❌ BOT TEXT SEND FAILED"
    )

    return False


# ============================================================
# Bot 发送单张图片
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

            photo_file = InputFile(
                io.BytesIO(image_bytes),
                filename="85h.jpg",
            )

            await bot.send_photo(
                chat_id=BOT_TARGET_CHANNEL,
                photo=photo_file,
                caption=caption[:1024]
                if caption
                else None,
            )

            print(
                f"   ✅ BOT PHOTO SENT "
                f"(attempt {attempt})"
            )

            return True

        except Exception as e:

            print(
                f"   ⚠️ BOT PHOTO RETRY "
                f"{attempt}/{retries}: {e}"
            )

            if attempt < retries:
                await asyncio.sleep(2)

    print(
        "   ❌ BOT PHOTO SEND FAILED"
    )

    return False


# ============================================================
# Bot 发送相册
# ============================================================

async def bot_send_album(
    items,
    retries=3,
):

    if not items:
        return False

    # Telegram Bot API MediaGroup 最多 10 个
    chunks = [
        items[i:i + 10]
        for i in range(
            0,
            len(items),
            10,
        )
    ]

    all_success = True

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"   📦 SEND ALBUM "
            f"{chunk_index}/{len(chunks)} "
            f"({len(chunk)} images)"
        )

        success = False

        for attempt in range(
            1,
            retries + 1,
        ):

            try:

                media = []

                for item in chunk:

                    raw = item["image_bytes"]

                    photo = InputFile(
                        io.BytesIO(raw),
                        filename="85h.jpg",
                    )

                    caption = item.get(
                        "caption"
                    )

                    media.append(
                        InputMediaPhoto(
                            media=photo,
                            caption=caption[:1024]
                            if caption
                            else None,
                        )
                    )

                await bot.send_media_group(
                    chat_id=BOT_TARGET_CHANNEL,
                    media=media,
                )

                print(
                    f"   ✅ ALBUM SENT "
                    f"(attempt {attempt})"
                )

                success = True
                break

            except Exception as e:

                print(
                    f"   ⚠️ BOT ALBUM RETRY "
                    f"{attempt}/{retries}: {e}"
                )

                if attempt < retries:
                    await asyncio.sleep(3)

        if not success:

            print(
                "   ❌ ALBUM SEND FAILED"
            )

            all_success = False

            # 后面的 album chunk 不继续发送
            break

    return all_success


# ============================================================
# 处理单张图片
# ============================================================

async def process_image_message(
    message,
):

    image = await download_message_image(
        message
    )

    if image is None:

        return {
            "status": "failed",
            "reason": "DOWNLOAD_FAILED",
        }

    # --------------------------------------------------------
    # 图片检测
    # --------------------------------------------------------

    inspection = inspect_image(
        image
    )

    if inspection["blocked"]:

        reason = inspection["reason"]

        print(
            f"   🚫 IMAGE SKIPPED: "
            f"{reason}"
        )

        if inspection.get("contacts"):

            print(
                "   CONTACTS: "
                + format_contacts(
                    inspection["contacts"]
                )
            )

        return {
            "status": "blocked",
            "reason": reason,
        }

    # --------------------------------------------------------
    # 添加水印
    # --------------------------------------------------------

    watermarked = add_watermark(
        image
    )

    image_bytes = image_to_bytes(
        watermarked
    )

    caption = (
        message.message
        if getattr(message, "message", None)
        else None
    )

    # --------------------------------------------------------
    # Caption 再检查一次
    # --------------------------------------------------------

    if caption:

        caption_contacts = detect_contacts(
            caption
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
                "reason": "CONTACT_IN_CAPTION",
            }

    return {
        "status": "ready",
        "image_bytes": image_bytes,
        "caption": caption,
    }


# ============================================================
# 获取当天消息
# ============================================================

async def collect_today_messages():

    source_channels = parse_source_channels()

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

    tomorrow_start = today_start + timedelta(
        days=1
    )

    print()
    print("=" * 70)
    print("📥 COLLECTING TODAY'S MESSAGES")
    print("=" * 70)

    print(
        f"BEIJING TODAY: "
        f"{today_start}"
    )

    all_messages = []

    for source_index, source in enumerate(
        source_channels
    ):

        print()
        print(
            f"🔎 SOURCE #{source_index + 1}: "
            f"{source}"
        )

        try:

            entity = await client.get_entity(
                source
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

            async for message in client.iter_messages(
                entity,
                limit=SCAN_LIMIT if SCAN_LIMIT > 0 else None,
            ):

                if not message:
                    continue

                message_date = message.date

                if message_date is None:
                    continue

                if message_date.tzinfo is None:
                    message_date = message_date.replace(
                        tzinfo=timezone.utc
                    )

                beijing_date = message_date.astimezone(
                    BEIJING_TZ
                )

                # Telegram iter_messages 默认从新到旧
                if beijing_date < today_start:
                    break

                if beijing_date >= tomorrow_start:
                    continue

                all_messages.append(
                    {
                        "source_index": source_index,
                        "source": source,
                        "entity": entity,
                        "message": message,
                        "date": beijing_date,
                    }
                )

                count += 1

            print(
                f"   ✅ TODAY MESSAGES: {count}"
            )

        except Exception as e:

            print(
                f"   ❌ SOURCE FAILED: "
                f"{type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # 跨频道按照真实时间排序
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
# 普通消息处理
# ============================================================

async def process_normal_message(
    item,
    processed,
):

    source_index = item["source_index"]
    source = item["source"]
    message = item["message"]

    key = message_key(
        source_index,
        message.id,
    )

    if is_already_processed(
        processed,
        source_index,
        message.id,
    ):

        return False

    text = (
        message.message
        if getattr(message, "message", None)
        else ""
    )

    # --------------------------------------------------------
    # 图片消息
    # --------------------------------------------------------

    if message.photo:

        print()
        print(
            f"🖼️ IMAGE "
            f"[{source}] "
            f"message={message.id}"
        )

        result = await process_image_message(
            message
        )

        if result["status"] == "blocked":

            # 被过滤掉的消息视为已经处理
            processed.add(key)

            print(
                "   ✅ FILTERED "
                "AND MARKED PROCESSED"
            )

            return True

        if result["status"] != "ready":

            print(
                "   ⚠️ IMAGE PROCESS FAILED"
            )

            return False

        success = await bot_send_photo(
            result["image_bytes"],
            result.get("caption"),
        )

        if success:

            processed.add(key)

            print(
                "   ✅ IMAGE PROCESSED"
            )

            return True

        print(
            "   ❌ IMAGE PUBLISH FAILED"
        )

        return False

    # --------------------------------------------------------
    # 普通文字消息
    # --------------------------------------------------------

    if text:

        contacts = detect_contacts(
            text
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
                + format_contacts(contacts)
            )

            processed.add(key)

            return True

        print()
        print(
            f"📝 TEXT "
            f"[{source}] "
            f"message={message.id}"
        )

        success = await bot_send_text(
            text
        )

        if success:

            processed.add(key)

            print(
                "   ✅ TEXT PROCESSED"
            )

            return True

        return False

    # --------------------------------------------------------
    # 其他类型
    # --------------------------------------------------------

    print()
    print(
        f"⚠️ UNSUPPORTED MESSAGE "
        f"[{source}] "
        f"message={message.id}"
    )

    processed.add(key)

    return True


# ============================================================
# 相册处理
# ============================================================

async def process_albums(
    album_groups,
    processed,
):

    if not album_groups:
        return

    print()
    print("=" * 70)
    print("📚 PROCESSING TELEGRAM ALBUMS")
    print("=" * 70)

    # 按 source_index + grouped_id
    # 已经在 collect 阶段完成分组
    for album_key, items in album_groups.items():

        items.sort(
            key=lambda x: (
                x["date"],
                x["message"].id,
            )
        )

        source_index = items[0][
            "source_index"
        ]

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

        # ----------------------------------------------------
        # 每张图片独立检测
        # ----------------------------------------------------

        for item in items:

            message = item["message"]

            key = message_key(
                source_index,
                message.id,
            )

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

            result = await process_image_message(
                message
            )

            if result["status"] == "blocked":

                processed.add(key)

                print(
                    f"   🚫 FILTERED "
                    f"message={message.id}"
                )

                continue

            if result["status"] != "ready":

                print(
                    f"   ❌ PROCESS FAILED "
                    f"message={message.id}"
                )

                continue

            valid_items.append(
                {
                    "key": key,
                    "message_id": message.id,
                    "image_bytes": result[
                        "image_bytes"
                    ],
                    "caption": result.get(
                        "caption"
                    ),
                }
            )

        # ----------------------------------------------------
        # 全部被过滤
        # ----------------------------------------------------

        if not valid_items:

            print(
                "   ℹ️ ALBUM HAS NO VALID IMAGES"
            )

            continue

        # ----------------------------------------------------
        # 发送相册
        # ----------------------------------------------------

        success = await bot_send_album(
            valid_items
        )

        if success:

            for item in valid_items:
                processed.add(
                    item["key"]
                )

            print(
                f"   ✅ ALBUM PROCESSED "
                f"({len(valid_items)} valid images)"
            )

        else:

            print(
                "   ❌ ALBUM PUBLISH FAILED"
            )

            # 不标记失败图片
            # 下次执行继续尝试


# ============================================================
# 处理全部消息
# ============================================================

async def process_messages(
    all_messages,
    processed,
):

    print()
    print("=" * 70)
    print("🔄 PROCESSING MESSAGES")
    print("=" * 70)

    album_groups = defaultdict(list)

    normal_messages = []

    # --------------------------------------------------------
    # 先识别相册
    # --------------------------------------------------------

    for item in all_messages:

        message = item["message"]

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
                item["source_index"],
                str(grouped_id),
            )

            album_groups[
                album_key
            ].append(item)

        else:

            normal_messages.append(
                item
            )

    # --------------------------------------------------------
    # 按全局时间顺序处理普通消息
    # --------------------------------------------------------

    for item in normal_messages:

        try:

            await process_normal_message(
                item,
                processed,
            )

        except Exception as e:

            source = item["source"]
            message = item["message"]

            print(
                f"❌ MESSAGE ERROR "
                f"[{source}] "
                f"message={message.id}: "
                f"{type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # 相册
    # --------------------------------------------------------

    await process_albums(
        album_groups,
        processed,
    )


# ============================================================
# 主程序
# ============================================================

async def main():

    global bot

    print("=" * 70)
    print("TELEGRAM AUTO FORWARDER")
    print("=" * 70)

    print(
        f"BEIJING TIME: {now_beijing()}"
    )

    source_channels = parse_source_channels()

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
        f"SCAN LIMIT: {SCAN_LIMIT}"
    )

    processed = load_processed()

    print(
        f"PROCESSED COUNT: "
        f"{len(processed)}"
    )

    # --------------------------------------------------------
    # Telegram User
    # --------------------------------------------------------

    await connect_telegram()

    # --------------------------------------------------------
    # Target Entity
    # --------------------------------------------------------

    target_entity = await resolve_target_entity()

    # --------------------------------------------------------
    # Bot
    # --------------------------------------------------------

    try:

        await diagnose_bot_target(
            target_entity
        )

    except Exception:

        # 只有真正无法访问目标频道、
        # Token 无效、明确没有发布权限等情况才退出
        raise

    # --------------------------------------------------------
    # 获取今天消息
    # --------------------------------------------------------

    all_messages = (
        await collect_today_messages()
    )

    # --------------------------------------------------------
    # 开始处理
    # --------------------------------------------------------

    await process_messages(
        all_messages,
        processed,
    )

    # --------------------------------------------------------
    # 保存 processed
    # --------------------------------------------------------

    save_processed(
        processed
    )

    print()
    print("=" * 70)
    print("✅ ALL TASKS COMPLETED")
    print("=" * 70)

    print(
        f"PROCESSED COUNT: "
        f"{len(processed)}"
    )

    print(
        f"BEIJING TIME END: "
        f"{now_beijing()}"
    )

    print("=" * 70)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "⚠️ PROGRAM INTERRUPTED"
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("❌ FATAL ERROR")
        print("=" * 70)

        print(
            f"{type(e).__name__}: {e}"
        )

        print("=" * 70)

        raise

    finally:

        # ----------------------------------------------------
        # 关闭 Bot
        # ----------------------------------------------------

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

        try:

            asyncio.run(
                cleanup()
            )

        except Exception:

            pass
