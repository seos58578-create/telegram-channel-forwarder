import os
import re
import json
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import pytesseract

from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont

from telethon import TelegramClient
from telegram import Bot, InputMediaPhoto
from telegram.error import TelegramError, TimedOut, NetworkError


# ============================================================
# 基础配置
# ============================================================

BEIJING_TZ = timezone(timedelta(hours=8))

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
TELEGRAM_SESSION = os.environ["TELEGRAM_SESSION"]

BOT_TOKEN = os.environ["BOT_TOKEN"]

SOURCE_CHANNELS_RAW = os.environ["SOURCE_CHANNELS"]
TARGET_CHANNEL = os.environ["TARGET_CHANNEL"]

SCAN_LIMIT_RAW = os.environ.get("SCAN_LIMIT", "0").strip()

PROCESSED_FILE = Path("processed.json")

WATERMARK_TEXT = ""

# OCR 语言
OCR_LANG = "eng+chi_sim+vie+por"

# 单个相册 Telegram 最多 10 张
MEDIA_GROUP_LIMIT = 10

# Bot 上传超时
READ_TIMEOUT = 90
WRITE_TIMEOUT = 180
CONNECT_TIMEOUT = 30
POOL_TIMEOUT = 30

# 重试次数
SEND_RETRIES = 3


# ============================================================
# 频道配置
# ============================================================

SOURCE_CHANNELS = [
    x.strip()
    for x in SOURCE_CHANNELS_RAW.split(",")
    if x.strip()
]


def normalize_target(value):
    """
    TARGET_CHANNEL:
    @username
    username
    -100xxxxxxxxxx
    """

    value = value.strip()

    if re.fullmatch(r"-?\d+", value):
        return int(value)

    if not value.startswith("@"):
        value = "@" + value

    return value


TARGET_CHAT_ID = normalize_target(TARGET_CHANNEL)


# ============================================================
# 联系方式识别
# ============================================================

CONTACT_PATTERNS = [

    (
        "TELEGRAM_LINK",
        re.compile(
            r"(?:https?://)?"
            r"(?:www\.)?"
            r"(?:t\.me|telegram\.me)/"
            r"[A-Za-z0-9_+/=?-]{4,64}",
            re.IGNORECASE
        )
    ),

    (
        "TELEGRAM_USERNAME",
        re.compile(
            r"@[A-Za-z][A-Za-z0-9_]{4,31}"
        )
    ),

    (
        "WHATSAPP_LINK",
        re.compile(
            r"(?:https?://)?"
            r"(?:www\.)?"
            r"wa\.me/"
            r"\d{7,15}",
            re.IGNORECASE
        )
    ),

    (
        "WECHAT",
        re.compile(
            r"\bwechat\b",
            re.IGNORECASE
        )
    ),

    (
        "CN_PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?86[- ]?)?"
            r"1[3-9]\d{9}"
            r"(?!\d)"
        )
    ),

    (
        "VN_PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?84[- ]?)?"
            r"(?:0?3|0?5|0?7|0?8|0?9)"
            r"\d{8}"
            r"(?!\d)"
        )
    ),
]


STRONG_CONTACT_TYPES = {
    "TELEGRAM_LINK",
    "TELEGRAM_USERNAME",
    "WHATSAPP_LINK",
    "WECHAT",
}


PHONE_CONTACT_TYPES = {
    "CN_PHONE",
    "VN_PHONE",
}


# ============================================================
# processed.json
# ============================================================

def load_processed():
    if not PROCESSED_FILE.exists():
        return set()

    try:
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data.get("messages", [])

        return {
            str(x)
            for x in messages
        }

    except Exception as e:
        print(f"⚠️ processed.json READ ERROR: {e}")
        return set()


def save_processed(processed):
    """
    保存 processed.json

    使用排序后的 ID，避免 set 顺序随机。
    """

    data = {
        "messages": sorted(
            list(processed),
            key=lambda x: int(x) if str(x).isdigit() else str(x)
        )[-10000:]
    }

    temp_file = PROCESSED_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    temp_file.replace(PROCESSED_FILE)


# ============================================================
# 时间
# ============================================================

def to_beijing(dt):
    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(BEIJING_TZ)


def is_today(dt):
    bj = to_beijing(dt)

    if bj is None:
        return False

    today = datetime.now(BEIJING_TZ).date()

    return bj.date() == today


# ============================================================
# SCAN LIMIT
# ============================================================

def get_scan_limit():
    """
    0 / 空 = 不限制
    """

    if not SCAN_LIMIT_RAW:
        return None

    try:
        value = int(SCAN_LIMIT_RAW)

        if value <= 0:
            return None

        return value

    except Exception:
        return None


SCAN_LIMIT = get_scan_limit()


# ============================================================
# 文本联系方式检测
# ============================================================

def detect_contact_text(text):
    if not text:
        return None

    for pattern_type, pattern in CONTACT_PATTERNS:

        matches = pattern.findall(text)

        if matches:

            return {
                "type": pattern_type,
                "matches": matches
            }

    return None


# ============================================================
# OCR
# ============================================================

def ocr_image_variants(image):
    """
    生成多个 OCR 版本
    """

    variants = []

    original = image.convert("RGB")
    variants.append(("ORIGINAL", original))

    gray = ImageOps.grayscale(original)
    variants.append(("GRAYSCALE", gray))

    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    variants.append(("ENHANCED", enhanced))

    threshold = enhanced.point(
        lambda p: 255 if p > 160 else 0
    )

    variants.append(("THRESHOLD", threshold))

    return variants


def detect_contact_by_ocr(image):
    """
    OCR 联系方式检测。

    强联系方式：
        一次命中即可。

    手机号：
        至少两个 OCR 版本独立命中才确认。
    """

    strong_hits = []
    phone_hits = {}

    for variant_name, variant_image in ocr_image_variants(image):

        try:

            text = pytesseract.image_to_string(
                variant_image,
                lang=OCR_LANG,
                config="--psm 6"
            )

        except Exception as e:
            print(
                f"⚠️ OCR ERROR "
                f"{variant_name}: {e}"
            )
            continue

        if not text:
            continue

        for pattern_type, pattern in CONTACT_PATTERNS:

            matches = pattern.findall(text)

            if not matches:
                continue

            if pattern_type in STRONG_CONTACT_TYPES:

                print(
                    f"🚫 IMAGE CONTACT FOUND"
                )
                print(
                    f"PATTERN: {pattern_type}"
                )
                print(
                    f"OCR PASS: {variant_name} "
                    f"MATCH: {matches}"
                )

                return {
                    "type": pattern_type,
                    "matches": matches,
                    "variant": variant_name
                }

            if pattern_type in PHONE_CONTACT_TYPES:

                key = pattern_type

                if key not in phone_hits:
                    phone_hits[key] = []

                phone_hits[key].append({
                    "variant": variant_name,
                    "matches": matches
                })

    # 手机号必须两个独立 OCR 版本确认
    for pattern_type, hits in phone_hits.items():

        unique_variants = {
            x["variant"]
            for x in hits
        }

        if len(unique_variants) >= 2:

            matches = hits[0]["matches"]

            print(
                f"🚫 IMAGE CONTACT FOUND"
            )

            print(
                f"PATTERN: {pattern_type}"
            )

            print(
                f"OCR PASS COUNT: "
                f"{len(unique_variants)}"
            )

            print(
                f"MATCH: {matches}"
            )

            return {
                "type": pattern_type,
                "matches": matches,
                "variant_count": len(unique_variants)
            }

        else:

            print(
                f"⚠️ {pattern_type} "
                f"OCR HIT BUT NOT CONFIRMED"
            )

            print(
                f"OCR PASS COUNT: "
                f"{len(unique_variants)}"
            )

    return None


# ============================================================
# QR CODE
# ============================================================

def detect_qr_code(image):
    """
    检测二维码
    """

    try:

        cv_image = cv2.cvtColor(
            __import__("numpy").array(image.convert("RGB")),
            cv2.COLOR_RGB2BGR
        )

        detector = cv2.QRCodeDetector()

        # 单二维码
        try:

            data, points, _ = detector.detectAndDecode(
                cv_image
            )

            if points is not None:
                print("🚫 QR CODE FOUND")

                if data:
                    print(
                        f"QR DATA: {data[:100]}"
                    )

                return True

        except Exception:
            pass

        # 多二维码
        try:

            result = detector.detectAndDecodeMulti(
                cv_image
            )

            if len(result) >= 2:

                ok = result[0]

                if ok:
                    print("🚫 QR CODE FOUND")
                    return True

        except Exception:
            pass

    except Exception as e:

        print(
            f"⚠️ QR DETECTOR ERROR: {e}"
        )

    return False


# ============================================================
# 水印
# ============================================================

def add_watermark(image_path):
    """
    给图片增加 85H 官方频道水印
    """

    try:

        image = Image.open(image_path).convert("RGBA")

        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0)
        )

        draw = ImageDraw.Draw(overlay)

        font_paths = [

            "/usr/share/fonts/opentype/noto/"
            "NotoSansCJK-Regular.ttc",

            "/usr/share/fonts/opentype/noto/"
            "NotoSansCJK-Bold.ttc",

        ]

        font_path = None

        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break

        if font_path:

            font_size = max(
                24,
                min(
                    64,
                    int(image.width * 0.045)
                )
            )

            font = ImageFont.truetype(
                font_path,
                font_size
            )

        else:

            font = ImageFont.load_default()

        text = WATERMARK_TEXT

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        margin = max(
            20,
            int(image.width * 0.025)
        )

        x = (
            image.width
            - text_width
            - margin
        )

        y = (
            image.height
            - text_height
            - margin
        )

        # 半透明黑底
        padding = 10

        draw.rounded_rectangle(
            (
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding
            ),
            radius=8,
            fill=(0, 0, 0, 110)
        )

        # 白色文字 + 黑色描边
        draw.text(
            (x, y),
            text,
            font=font,
            fill=(255, 255, 255, 220),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220)
        )

        result = Image.alpha_composite(
            image,
            overlay
        ).convert("RGB")

        result.save(
            image_path,
            quality=95,
            optimize=True
        )

        return True

    except Exception as e:

        print(
            f"❌ WATERMARK ERROR: {e}"
        )

        return False


# ============================================================
# Bot 发送文本
# ============================================================

async def bot_send_text(
    bot,
    chat_id,
    text
):
    if not text:
        return True

    # Telegram 普通文本上限约 4096
    chunks = []

    max_length = 4000

    for i in range(
        0,
        len(text),
        max_length
    ):
        chunks.append(
            text[i:i + max_length]
        )

    for chunk in chunks:

        success = False

        for attempt in range(
            1,
            SEND_RETRIES + 1
        ):

            try:

                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    read_timeout=READ_TIMEOUT,
                    write_timeout=WRITE_TIMEOUT,
                    connect_timeout=CONNECT_TIMEOUT,
                    pool_timeout=POOL_TIMEOUT,
                )

                success = True
                break

            except Exception as e:

                print(
                    f"❌ BOT TEXT ERROR "
                    f"ATTEMPT {attempt}/"
                    f"{SEND_RETRIES}: {e}"
                )

                if attempt < SEND_RETRIES:

                    wait = attempt * 5

                    print(
                        f"⏳ WAIT {wait}s "
                        f"THEN RETRY..."
                    )

                    await asyncio.sleep(wait)

        if not success:
            return False

    return True


# ============================================================
# Bot 单图发送
# ============================================================

async def bot_send_photo(
    bot,
    chat_id,
    image_path,
    caption=None
):
    """
    单图发送，失败自动重试。
    """

    for attempt in range(
        1,
        SEND_RETRIES + 1
    ):

        try:

            print(
                f"📤 SEND IMAGE "
                f"ATTEMPT {attempt}/"
                f"{SEND_RETRIES}"
            )

            with open(
                image_path,
                "rb"
            ) as photo:

                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=(
                        caption[:1024]
                        if caption
                        else None
                    ),
                    read_timeout=READ_TIMEOUT,
                    write_timeout=WRITE_TIMEOUT,
                    connect_timeout=CONNECT_TIMEOUT,
                    pool_timeout=POOL_TIMEOUT,
                )

            print(
                "✅ IMAGE PUBLISHED"
            )

            # caption 超过 1024
            if caption and len(caption) > 1024:

                extra = caption[1024:]

                print(
                    "📝 CAPTION TOO LONG"
                    " → SEND REMAINING TEXT"
                )

                text_success = await bot_send_text(
                    bot,
                    chat_id,
                    extra
                )

                if not text_success:
                    return False

            return True

        except Exception as e:

            print(
                f"❌ BOT IMAGE ERROR"
            )

            print(
                f"ERROR: {e}"
            )

            if attempt < SEND_RETRIES:

                wait = attempt * 5

                print(
                    f"⏳ WAIT {wait}s "
                    f"THEN RETRY..."
                )

                await asyncio.sleep(wait)

    return False


# ============================================================
# Bot 相册发送
# ============================================================

async def bot_send_media_group(
    bot,
    chat_id,
    image_paths,
    caption=None
):
    """
    发送 Telegram 相册。

    Telegram Bot API 一次最多 10 张。
    """

    if not image_paths:
        return False

    # 分批
    groups = []

    for i in range(
        0,
        len(image_paths),
        MEDIA_GROUP_LIMIT
    ):

        groups.append(
            image_paths[
                i:i + MEDIA_GROUP_LIMIT
            ]
        )

    caption_sent = False

    for group_index, group in enumerate(groups, 1):

        group_caption = None

        # caption 只放第一组第一张
        if (
            caption
            and not caption_sent
        ):
            group_caption = caption[:1024]

        success = False

        for attempt in range(
            1,
            SEND_RETRIES + 1
        ):

            opened_files = []

            try:

                print(
                    f"📤 SEND ALBUM "
                    f"{group_index}/{len(groups)} "
                    f"({len(group)} IMAGES) "
                    f"ATTEMPT "
                    f"{attempt}/{SEND_RETRIES}"
                )

                media = []

                for path in group:

                    f = open(
                        path,
                        "rb"
                    )

                    opened_files.append(f)

                    media.append(
                        InputMediaPhoto(
                            media=f
                        )
                    )

                if group_caption:

                    media[0].caption = (
                        group_caption
                    )

                await bot.send_media_group(
                    chat_id=chat_id,
                    media=media,
                    read_timeout=READ_TIMEOUT,
                    write_timeout=WRITE_TIMEOUT,
                    connect_timeout=CONNECT_TIMEOUT,
                    pool_timeout=POOL_TIMEOUT,
                )

                print(
                    f"✅ ALBUM PUBLISHED "
                    f"({len(group)} IMAGES)"
                )

                success = True

                if group_caption:

                    caption_sent = True

                    # 超过 1024 字符
                    if len(caption) > 1024:

                        extra = caption[1024:]

                        print(
                            "📝 CAPTION TOO LONG"
                            " → SEND REMAINING TEXT"
                        )

                        text_success = (
                            await bot_send_text(
                                bot,
                                chat_id,
                                extra
                            )
                        )

                        if not text_success:
                            success = False

                break

            except Exception as e:

                print(
                    f"❌ BOT ALBUM ERROR"
                )

                print(
                    f"ERROR: {e}"
                )

                if attempt < SEND_RETRIES:

                    wait = attempt * 5

                    print(
                        f"⏳ WAIT {wait}s "
                        f"THEN RETRY..."
                    )

                    await asyncio.sleep(
                        wait
                    )

            finally:

                for f in opened_files:

                    try:
                        f.close()
                    except Exception:
                        pass

        if not success:
            return False

    return True


# ============================================================
# 下载图片
# ============================================================

async def download_photo(
    client,
    message,
    work_dir
):

    path = os.path.join(
        work_dir,
        f"{message.id}.jpg"
    )

    try:

        result = await client.download_media(
            message,
            file=path
        )

        if not result:
            return None

        return path

    except Exception as e:

        print(
            f"❌ IMAGE DOWNLOAD ERROR: {e}"
        )

        return None


# ============================================================
# 图片安全检查
# ============================================================

def inspect_image(
    image_path,
    caption=None
):
    """
    图片检查：
    1. caption 联系方式
    2. QR
    3. OCR
    """

    # --------------------------------------------------------
    # Caption
    # --------------------------------------------------------

    if caption:

        contact = detect_contact_text(
            caption
        )

        if contact:

            print(
                "🚫 CAPTION CONTACT FOUND"
            )

            print(
                f"PATTERN: "
                f"{contact['type']}"
            )

            return {
                "ok": False,
                "reason": (
                    "CAPTION_CONTACT"
                )
            }

    # --------------------------------------------------------
    # 打开图片
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        )

        image.load()

    except Exception as e:

        print(
            f"❌ IMAGE OPEN ERROR: {e}"
        )

        return {
            "ok": False,
            "reason": "IMAGE_OPEN_ERROR"
        }

    # --------------------------------------------------------
    # QR
    # --------------------------------------------------------

    if detect_qr_code(image):

        print(
            "🚫 IMAGE SKIPPED"
        )

        print(
            "REASON: QR_CODE"
        )

        return {
            "ok": False,
            "reason": "QR_CODE"
        }

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    contact = detect_contact_by_ocr(
        image
    )

    if contact:

        print(
            "🚫 IMAGE SKIPPED"
        )

        print(
            f"REASON: "
            f"{contact['type']}"
        )

        return {
            "ok": False,
            "reason": contact["type"]
        }

    print(
        "✅ IMAGE OCR: NO CONTACT FOUND"
    )

    return {
        "ok": True,
        "reason": None
    }


# ============================================================
# 处理单张图片
# ============================================================

async def prepare_image(
    client,
    message,
    work_dir
):
    """
    下载 → QR → OCR → 水印

    返回：
        {
            "success": True,
            "path": "...",
            "caption": "..."
        }

    或：

        {
            "success": False,
            "reason": "..."
        }
    """

    print(
        "🖼 IMAGE MESSAGE"
    )

    print(
        "⬇️ DOWNLOADING IMAGE..."
    )

    image_path = await download_photo(
        client,
        message,
        work_dir
    )

    if not image_path:

        return {
            "success": False,
            "reason": "DOWNLOAD_ERROR"
        }

    # --------------------------------------------------------
    # 图片检查
    # --------------------------------------------------------

    result = inspect_image(
        image_path,
        message.message or ""
    )

    if not result["ok"]:

        try:
            os.remove(image_path)
        except Exception:
            pass

        return {
            "success": False,
            "reason": result["reason"]
        }

    # --------------------------------------------------------
    # 水印
    # --------------------------------------------------------

    print(
        "💧 ADD WATERMARK..."
    )

    if not add_watermark(
        image_path
    ):

        try:
            os.remove(image_path)
        except Exception:
            pass

        return {
            "success": False,
            "reason": "WATERMARK_ERROR"
        }

    return {
        "success": True,
        "path": image_path,
        "caption": message.message or ""
    }


# ============================================================
# 普通文本消息
# ============================================================

async def process_text_message(
    bot,
    message,
    processed
):

    text = message.message or ""

    if not text.strip():

        processed.add(
            str(message.id)
        )

        save_processed(processed)

        print(
            "ℹ️ EMPTY MESSAGE → SKIPPED"
        )

        return True

    # --------------------------------------------------------
    # 联系方式
    # --------------------------------------------------------

    contact = detect_contact_text(
        text
    )

    if contact:

        print(
            "🚫 TEXT CONTACT FOUND"
        )

        print(
            f"PATTERN: "
            f"{contact['type']}"
        )

        print(
            "🚫 TEXT MESSAGE SKIPPED"
        )

        processed.add(
            str(message.id)
        )

        save_processed(processed)

        return True

    # --------------------------------------------------------
    # 发布
    # --------------------------------------------------------

    print(
        "📝 TEXT MESSAGE"
    )

    print(
        "📤 SEND TEXT..."
    )

    success = await bot_send_text(
        bot,
        TARGET_CHAT_ID,
        text
    )

    if success:

        print(
            "✅ TEXT PUBLISHED"
        )

        processed.add(
            str(message.id)
        )

        save_processed(processed)

        return True

    print(
        "❌ TEXT PUBLISH FAILED"
    )

    print(
        "MESSAGE WILL BE RETRIED NEXT RUN"
    )

    return False


# ============================================================
# 单图处理
# ============================================================

async def process_single_image(
    client,
    bot,
    message,
    processed,
    work_dir
):

    print()
    print(
        "=" * 70
    )

    print(
        f"PROCESS IMAGE MESSAGE: "
        f"{message.id}"
    )

    print(
        f"SOURCE: "
        f"{message.chat_id}"
    )

    print(
        f"TIME: "
        f"{to_beijing(message.date)}"
    )

    result = await prepare_image(
        client,
        message,
        work_dir
    )

    if not result["success"]:

        # 过滤掉的图片也必须记录
        if result["reason"] not in {
            "DOWNLOAD_ERROR",
            "WATERMARK_ERROR",
            "IMAGE_OPEN_ERROR",
        }:

            processed.add(
                str(message.id)
            )

            save_processed(processed)

        return (
            result["reason"]
            not in {
                "DOWNLOAD_ERROR",
                "WATERMARK_ERROR",
                "IMAGE_OPEN_ERROR",
            }
        )

    caption = result["caption"]

    print(
        "📤 SEND IMAGE..."
    )

    success = await bot_send_photo(
        bot,
        TARGET_CHAT_ID,
        result["path"],
        caption
    )

    try:
        os.remove(
            result["path"]
        )
    except Exception:
        pass

    if success:

        processed.add(
            str(message.id)
        )

        save_processed(
            processed
        )

        return True

    print(
        "❌ IMAGE PUBLISH FAILED"
    )

    print(
        "MESSAGE WILL BE RETRIED NEXT RUN"
    )

    return False


# ============================================================
# 多图相册处理
# ============================================================

async def process_album(
    client,
    bot,
    album_messages,
    processed,
    work_dir
):

    if not album_messages:
        return True

    print()
    print(
        "=" * 70
    )

    print(
        "📚 TELEGRAM ALBUM FOUND"
    )

    print(
        f"ALBUM SIZE: "
        f"{len(album_messages)}"
    )

    grouped_id = (
        album_messages[0].grouped_id
    )

    print(
        f"GROUPED ID: "
        f"{grouped_id}"
    )

    # --------------------------------------------------------
    # Caption
    #
    # Telegram 相册通常 caption 在第一张。
    # 如果其他图片有 caption，也尝试寻找。
    # --------------------------------------------------------

    album_caption = ""

    for item in album_messages:

        if item.message:

            album_caption = (
                item.message
            )

            break

    # --------------------------------------------------------
    # 相册 Caption 联系方式
    # --------------------------------------------------------

    if album_caption:

        contact = detect_contact_text(
            album_caption
        )

        if contact:

            print(
                "🚫 ALBUM CAPTION CONTACT FOUND"
            )

            print(
                f"PATTERN: "
                f"{contact['type']}"
            )

            print(
                "🚫 ENTIRE ALBUM SKIPPED"
            )

            for item in album_messages:

                processed.add(
                    str(item.id)
                )

            save_processed(
                processed
            )

            return True

    # --------------------------------------------------------
    # 已处理图片过滤
    # --------------------------------------------------------

    pending_messages = []

    for item in album_messages:

        if str(item.id) in processed:

            print(
                f"IMAGE MESSAGE "
                f"{item.id} already processed."
            )

        else:

            pending_messages.append(
                item
            )

    if not pending_messages:

        print(
            "✅ ENTIRE ALBUM ALREADY PROCESSED"
        )

        return True

    # --------------------------------------------------------
    # 每一张图片分别检测
    # --------------------------------------------------------

    valid_images = []

    skipped_count = 0

    for index, item in enumerate(
        pending_messages,
        1
    ):

        print()
        print(
            f"📷 ALBUM IMAGE "
            f"{index}/{len(pending_messages)}"
        )

        print(
            f"MESSAGE ID: "
            f"{item.id}"
        )

        result = await prepare_image(
            client,
            item,
            work_dir
        )

        if not result["success"]:

            reason = result["reason"]

            print(
                f"🚫 ALBUM IMAGE "
                f"{item.id} SKIPPED"
            )

            print(
                f"REASON: {reason}"
            )

            # 过滤类型 → 标记已处理
            if reason not in {
                "DOWNLOAD_ERROR",
                "WATERMARK_ERROR",
                "IMAGE_OPEN_ERROR",
            }:

                processed.add(
                    str(item.id)
                )

                save_processed(
                    processed
                )

            skipped_count += 1

            continue

        valid_images.append({
            "message": item,
            "path": result["path"],
        })

    # --------------------------------------------------------
    # 没有合格图片
    # --------------------------------------------------------

    if not valid_images:

        print(
            "ℹ️ NO VALID IMAGES IN ALBUM"
        )

        return True

    print()
    print(
        "📊 ALBUM RESULT"
    )

    print(
        f"ORIGINAL IMAGES: "
        f"{len(album_messages)}"
    )

    print(
        f"VALID IMAGES: "
        f"{len(valid_images)}"
    )

    print(
        f"SKIPPED IMAGES: "
        f"{skipped_count}"
    )

    # --------------------------------------------------------
    # 获取路径
    # --------------------------------------------------------

    paths = [
        x["path"]
        for x in valid_images
    ]

    # --------------------------------------------------------
    # 发布
    # --------------------------------------------------------

    if len(paths) == 1:

        print(
            "📤 ALBUM RESULT ONLY 1 IMAGE"
        )

        success = await bot_send_photo(
            bot,
            TARGET_CHAT_ID,
            paths[0],
            album_caption
        )

    else:

        print(
            f"📤 SEND ALBUM: "
            f"{len(paths)} IMAGES"
        )

        success = await bot_send_media_group(
            bot,
            TARGET_CHAT_ID,
            paths,
            album_caption
        )

    # --------------------------------------------------------
    # 清理临时文件
    # --------------------------------------------------------

    for path in paths:

        try:
            os.remove(path)
        except Exception:
            pass

    # --------------------------------------------------------
    # 发布成功
    # --------------------------------------------------------

    if success:

        print(
            "✅ ALBUM PUBLISHED"
        )

        for item in valid_images:

            processed.add(
                str(item["message"].id)
            )

        save_processed(
            processed
        )

        return True

    # --------------------------------------------------------
    # 发布失败
    # --------------------------------------------------------

    print(
        "❌ ALBUM PUBLISH FAILED"
    )

    print(
        "ALBUM IMAGES WILL BE RETRIED NEXT RUN"
    )

    return False


# ============================================================
# 获取今天全部消息
# ============================================================

async def collect_today_messages(
    client,
    source_entities
):

    all_messages = []

    for source_index, source_entity in enumerate(
        source_entities
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"SOURCE CHANNEL: "
            f"{SOURCE_CHANNELS[source_index]}"
        )

        print(
            f"SOURCE ENTITY: "
            f"{source_entity}"
        )

        today_messages = []

        count = 0

        async for message in client.iter_messages(
            source_entity,
            limit=SCAN_LIMIT
        ):

            # 今天以前
            if not is_today(
                message.date
            ):

                # Telethon 默认倒序：
                # 新 → 旧
                # 遇到昨天即可停止
                if to_beijing(
                    message.date
                ).date() < datetime.now(
                    BEIJING_TZ
                ).date():

                    break

                continue

            count += 1

            if message.photo:

                today_messages.append(
                    message
                )

            elif message.message:

                today_messages.append(
                    message
                )

        print(
            f"TODAY MESSAGES: "
            f"{count}"
        )

        all_messages.extend(
            [
                (
                    source_index,
                    source_entity,
                    m
                )
                for m in today_messages
            ]
        )

    return all_messages


# ============================================================
# 构建最终发布顺序
# ============================================================

def build_publish_groups(
    all_messages
):
    """
    关键逻辑：

    单图：
        SINGLE

    多图：
        ALBUM

    同一个 grouped_id 的图片，
    归到同一个 ALBUM。
    """

    # --------------------------------------------------------
    # 先按时间排序
    # --------------------------------------------------------

    all_messages.sort(
        key=lambda x: (
            x[2].date,
            x[0],
            x[2].id
        )
    )

    result = []

    album_map = {}

    for source_index, source_entity, message in all_messages:

        if message.grouped_id:

            key = (
                source_index,
                message.grouped_id
            )

            if key not in album_map:

                album_map[key] = {
                    "type": "album",
                    "source_index": source_index,
                    "source_entity": source_entity,
                    "grouped_id": message.grouped_id,
                    "messages": []
                }

                result.append(
                    album_map[key]
                )

            album_map[key][
                "messages"
            ].append(message)

        else:

            result.append({
                "type": "single",
                "source_index": source_index,
                "source_entity": source_entity,
                "message": message
            })

    # --------------------------------------------------------
    # 每个相册内部按 message ID 排序
    # --------------------------------------------------------

    for item in result:

        if item["type"] == "album":

            item["messages"].sort(
                key=lambda x: x.id
            )

    return result


# ============================================================
# 主程序
# ============================================================

async def main():

    print()
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
        f"{datetime.now(BEIJING_TZ)}"
    )

    print(
        f"SOURCE CHANNELS: "
        f"{SOURCE_CHANNELS}"
    )

    print(
        f"TARGET CHANNEL: "
        f"{TARGET_CHANNEL}"
    )

    print(
        f"SCAN LIMIT: "
        f"{SCAN_LIMIT if SCAN_LIMIT else 'UNLIMITED'}"
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
    # Telegram Client
    # --------------------------------------------------------

    print(
        "CONNECTING TELEGRAM..."
    )

    client = TelegramClient(
        TELEGRAM_SESSION,
        API_ID,
        API_HASH
    )

    await client.start()

    print(
        "✅ TELEGRAM CONNECTED"
    )

    # --------------------------------------------------------
    # Bot
    # --------------------------------------------------------

    bot = Bot(
        token=BOT_TOKEN
    )

    try:

        bot_me = await bot.get_me()

        print(
            f"BOT: "
            f"@{bot_me.username}"
        )

    except Exception as e:

        print(
            f"❌ BOT CONNECT ERROR: {e}"
        )

        await client.disconnect()

        return

    # --------------------------------------------------------
    # Target Entity
    #
    # 用普通 Telegram 账号验证目标频道。
    # 不再提前调用 Bot get_chat，
    # 避免之前 Chat not found 导致程序提前退出。
    # --------------------------------------------------------

    try:

        target_entity = await client.get_entity(
            TARGET_CHAT_ID
        )

        print(
            f"TARGET ENTITY: "
            f"{target_entity}"
        )

        print(
            f"TARGET TITLE: "
            f"{getattr(target_entity, 'title', '')}"
        )

        print(
            f"TARGET ID: "
            f"{getattr(target_entity, 'id', '')}"
        )

    except Exception as e:

        print()
        print(
            "❌ TARGET CHANNEL ERROR"
        )

        print(
            f"ERROR: {e}"
        )

        await client.disconnect()

        return

    print()
    print(
        "ℹ️ BOT TARGET PRE-CHECK SKIPPED"
    )

    print(
        "程序将在实际发布时验证 Bot 权限"
    )

    # --------------------------------------------------------
    # Source Entities
    # --------------------------------------------------------

    source_entities = []

    for source in SOURCE_CHANNELS:

        try:

            entity = await client.get_entity(
                source
            )

            source_entities.append(
                entity
            )

        except Exception as e:

            print()
            print(
                f"❌ SOURCE CHANNEL ERROR: "
                f"{source}"
            )

            print(
                f"ERROR: {e}"
            )

    if not source_entities:

        print(
            "❌ NO SOURCE CHANNELS AVAILABLE"
        )

        await client.disconnect()

        return

    # --------------------------------------------------------
    # 获取今日消息
    # --------------------------------------------------------

    all_messages = await collect_today_messages(
        client,
        source_entities
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"TOTAL TODAY MESSAGES: "
        f"{len(all_messages)}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 建立发布组
    # --------------------------------------------------------

    publish_groups = build_publish_groups(
        all_messages
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL PUBLISH ORDER:"
    )

    print(
        "=" * 70
    )

    display_index = 0

    for item in publish_groups:

        if item["type"] == "single":

            message = item["message"]

            display_index += 1

            print(
                f"{display_index}. "
                f"{SOURCE_CHANNELS[item['source_index']]} "
                f"MSG={message.id} "
                f"TIME={to_beijing(message.date)}"
            )

        else:

            messages = item["messages"]

            display_index += 1

            ids = [
                str(x.id)
                for x in messages
            ]

            print(
                f"{display_index}. "
                f"{SOURCE_CHANNELS[item['source_index']]} "
                f"ALBUM="
                f"{item['grouped_id']} "
                f"MESSAGES="
                f"{','.join(ids)} "
                f"TIME="
                f"{to_beijing(messages[0].date)}"
            )

    # --------------------------------------------------------
    # 开始发布
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "START PUBLISHING..."
    )

    print(
        "=" * 70
    )

    # 临时目录
    with tempfile.TemporaryDirectory(
        prefix="telegram_forwarder_"
    ) as work_dir:

        total = len(publish_groups)

        for index, item in enumerate(
            publish_groups,
            1
        ):

            print()
            print(
                f"[{index}/{total}]"
            )

            # =================================================
            # 单条消息
            # =================================================

            if item["type"] == "single":

                message = item["message"]

                if str(message.id) in processed:

                    print(
                        f"MESSAGE "
                        f"{message.id} "
                        f"already processed."
                    )

                    continue

                # 图片
                if message.photo:

                    await process_single_image(
                        client,
                        bot,
                        message,
                        processed,
                        work_dir
                    )

                # 文本
                else:

                    print()
                    print(
                        "=" * 70
                    )

                    print(
                        f"PROCESS MESSAGE: "
                        f"{message.id}"
                    )

                    print(
                        f"SOURCE: "
                        f"{message.chat_id}"
                    )

                    print(
                        f"TIME: "
                        f"{to_beijing(message.date)}"
                    )

                    await process_text_message(
                        bot,
                        message,
                        processed
                    )

            # =================================================
            # 相册
            # =================================================

            else:

                album_messages = item["messages"]

                # 如果相册全部已经处理
                if all(
                    str(x.id) in processed
                    for x in album_messages
                ):

                    print(
                        "ALBUM already processed."
                    )

                    print(
                        "MESSAGE IDS: "
                        + ",".join(
                            str(x.id)
                            for x in album_messages
                        )
                    )

                    continue

                await process_album(
                    client,
                    bot,
                    album_messages,
                    processed,
                    work_dir
                )

    # --------------------------------------------------------
    # 最终保存
    # --------------------------------------------------------

    save_processed(
        processed
    )

    await client.disconnect()

    print()
    print(
        "=" * 70
    )

    print(
        "✅ ALL DONE"
    )

    print(
        f"TODAY MESSAGES: "
        f"{len(all_messages)}"
    )

    print(
        f"PUBLISH GROUPS: "
        f"{len(publish_groups)}"
    )

    print(
        f"PROCESSED COUNT: "
        f"{len(processed)}"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "STOPPED BY USER"
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
            f"{type(e).__name__}: {e}"
        )

        print(
            "=" * 70
        )

        raise
