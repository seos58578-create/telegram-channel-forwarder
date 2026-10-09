# -*- coding: utf-8 -*-
"""
Telegram Channel Forwarder
适用于 GitHub Actions + Telethon + Telegram Bot API

功能：
1. 采集来源频道当天发布的文字和图片（北京时间）
2. 支持多来源频道、单个目标频道
3. 检测文字、二维码及图片中的疑似联系方式
4. 图片通过检测后添加水印
5. 支持单图及多图相册转发
6. 使用 httpx 直接上传相册，明确绑定每个 multipart 附件
7. 使用 processed.json 避免重复处理
"""

import asyncio
import io
import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytesseract

from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from telethon import TelegramClient, utils
from telethon.sessions import StringSession
from telegram import Bot, InputFile


# ============================================================
# 1. 基础配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_FILE = BASE_DIR / "processed.json"

BEIJING_TZ = timezone(timedelta(hours=8))

API_ID = int(os.environ.get("API_ID", "0"))
API_HASH = os.environ.get("API_HASH", "").strip()
TELEGRAM_SESSION = os.environ.get("TELEGRAM_SESSION", "").strip()
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()

SOURCE_CHANNELS_RAW = os.environ.get("SOURCE_CHANNELS", "").strip()
TARGET_CHANNEL_RAW = os.environ.get("TARGET_CHANNEL", "").strip()

# 0 表示不限制扫描数量，但只处理当天的消息
SCAN_LIMIT = int(os.environ.get("SCAN_LIMIT", "0"))

WATERMARK_TEXT = "85H官方频道"

# 单批相册最多 10 张
ALBUM_BATCH_SIZE = 10

# 发送失败时最多重试次数
SEND_RETRIES = 3

# 图片 OCR 语言
OCR_LANG = "eng+chi_sim+vie+por"

# 日志仅显示警告及以上级别，减少 Telethon 的下载日志
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
logging.getLogger("telethon").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


# ============================================================
# 2. 联系方式检测规则
# ============================================================

CONTACT_PATTERNS = [
    (
        "TELEGRAM_LINK",
        re.compile(
            r"(?:https?://)?(?:www\.)?"
            r"(?:t\.me|telegram\.me)/[A-Za-z0-9_+/=?-]{4,64}",
            re.IGNORECASE,
        ),
    ),
    (
        "TELEGRAM_USERNAME",
        re.compile(r"@[A-Za-z][A-Za-z0-9_]{4,31}"),
    ),
    (
        "WHATSAPP_LINK",
        re.compile(
            r"(?:https?://)?(?:www\.)?wa\.me/\d{7,15}",
            re.IGNORECASE,
        ),
    ),
    (
        "WHATSAPP_TEXT",
        re.compile(r"\bwhatsapp\b", re.IGNORECASE),
    ),
    (
        "WECHAT",
        re.compile(r"\bwechat\b", re.IGNORECASE),
    ),
    (
        "WECHAT_CN",
        re.compile(r"微信号|微信|微[信號号]"),
    ),
    (
        "CN_PHONE",
        re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    ),
    (
        "VN_PHONE",
        re.compile(
            r"(?<!\d)(?:\+?84[- ]?)?"
            r"(?:0?3|0?5|0?7|0?8|0?9)\d{8}(?!\d)"
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

PHONE_CONTACT_TYPES = {"CN_PHONE", "VN_PHONE"}


def detect_contact_text(text):
    """检测文本中的疑似联系方式。"""
    if not text:
        return None

    for pattern_name, pattern in CONTACT_PATTERNS:
        if pattern.search(text):
            return pattern_name

    return None


# ============================================================
# 3. processed.json 去重管理
# ============================================================

processed_ids = set()


def load_processed():
    """加载已经处理过的消息记录。"""
    global processed_ids

    if not PROCESSED_FILE.exists():
        processed_ids = set()
        return

    try:
        data = json.loads(PROCESSED_FILE.read_text(encoding="utf-8"))
        messages = data.get("messages", [])

        # 兼容旧版 processed.json 中保存的整数 ID
        processed_ids = {str(item) for item in messages}

    except Exception as exc:
        print(f"⚠️ 读取 processed.json 失败：{exc}")
        processed_ids = set()


def save_processed():
    """立即保存处理记录，避免任务中断后丢失进度。"""
    data = {
        "updated_at": datetime.now(BEIJING_TZ).isoformat(),
        "messages": sorted(processed_ids),
    }

    temp_file = PROCESSED_FILE.with_suffix(".json.tmp")

    temp_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temp_file.replace(PROCESSED_FILE)


def message_key(source_id, message_id):
    return f"{source_id}:{message_id}"


def is_processed(source_id, message_id):
    """
    优先使用来源频道 ID + 消息 ID 去重。
    同时兼容旧版仅保存消息 ID 的记录。
    """
    return (
        message_key(source_id, message_id) in processed_ids
        or str(message_id) in processed_ids
    )


def mark_processed(source_id, message_id):
    processed_ids.add(message_key(source_id, message_id))
    save_processed()


def mark_messages_processed(source_id, messages):
    changed = False

    for message in messages:
        key = message_key(source_id, message.id)

        if key not in processed_ids:
            processed_ids.add(key)
            changed = True

    if changed:
        save_processed()


# ============================================================
# 4. 图片识别：二维码及 OCR
# ============================================================

def decode_image(image_bytes):
    """将图片二进制内容解码为 OpenCV 图片。"""
    if not image_bytes:
        return None

    try:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def detect_qr_code(image_bytes):
    """检测图片中是否存在二维码。"""
    image = decode_image(image_bytes)

    if image is None:
        return False

    try:
        detector = cv2.QRCodeDetector()

        # 先尝试识别多个二维码
        try:
            result = detector.detectAndDecodeMulti(image)

            if result and len(result) >= 2:
                decoded_values = result[1]

                if decoded_values and any(decoded_values):
                    return True

        except Exception:
            pass

        # 再尝试识别单个二维码
        decoded_text, points, _ = detector.detectAndDecode(image)

        if points is not None:
            return True

        if decoded_text:
            return True

    except Exception as exc:
        print(f"⚠️ 二维码检测异常：{exc}")

    return False


def make_ocr_variants(image_bytes):
    """生成适合 OCR 的不同图片版本。"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 原图缩放，避免超大图片拖慢 OCR
    image.thumbnail((1800, 1800))

    variants = [image]

    gray = image.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
    variants.append(enhanced)

    threshold = enhanced.point(lambda value: 255 if value > 165 else 0)
    variants.append(threshold)

    return variants


def detect_contact_in_image(image_bytes):
    """
    OCR 检测图片中的疑似联系方式。

    强联系方式命中一次即判定；
    手机号需要在至少两个 OCR 版本中被识别到，
    以降低单次 OCR 误判。
    """
    try:
        variants = make_ocr_variants(image_bytes)
    except Exception as exc:
        print(f"⚠️ 图片预处理失败：{exc}")
        return None

    phone_hits = set()

    for index, image in enumerate(variants):
        try:
            text = pytesseract.image_to_string(
                image,
                lang=OCR_LANG,
                config="--psm 6",
            )

            # 检测 Telegram、WhatsApp、微信等强联系方式
            for pattern_name, pattern in CONTACT_PATTERNS:
                if pattern_name not in STRONG_CONTACT_TYPES:
                    continue

                if pattern.search(text):
                    return pattern_name

            # 手机号在多个 OCR 版本中出现才判定
            for pattern_name, pattern in CONTACT_PATTERNS:
                if pattern_name not in PHONE_CONTACT_TYPES:
                    continue

                if pattern.search(text):
                    phone_hits.add(pattern_name)

            # 对手机号码的第二次识别使用另一种页面分割模式
            if index == 0:
                extra_text = pytesseract.image_to_string(
                    image,
                    lang=OCR_LANG,
                    config="--psm 11",
                )

                for pattern_name, pattern in CONTACT_PATTERNS:
                    if pattern_name in STRONG_CONTACT_TYPES:
                        if pattern.search(extra_text):
                            return pattern_name

                    elif pattern_name in PHONE_CONTACT_TYPES:
                        if pattern.search(extra_text):
                            phone_hits.add(pattern_name)

        except Exception as exc:
            print(f"⚠️ OCR 识别异常：{exc}")

    if phone_hits:
        return next(iter(phone_hits))

    return None


# ============================================================
# 5. 图片水印
# ============================================================

def find_watermark_font(size):
    """尝试寻找 GitHub Ubuntu 环境中的中文字体。"""
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue

    return ImageFont.load_default()


def add_watermark(image_bytes):
    """添加 85H官方频道 水印并输出 JPEG 字节。"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 限制图片最长边，减少上传体积及处理耗时
    image.thumbnail((2000, 2000))

    draw = ImageDraw.Draw(image)

    font_size = max(20, min(image.width, image.height) // 24)
    font = find_watermark_font(font_size)

    bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = max(12, image.width // 70)

    x = max(margin, image.width - text_width - margin)
    y = max(margin, image.height - text_height - margin)

    padding = max(6, font_size // 4)

    draw.rectangle(
        [
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding,
        ],
        fill=(0, 0, 0),
    )

    draw.text(
        (x, y),
        WATERMARK_TEXT,
        font=font,
        fill=(255, 255, 255),
    )

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=88,
        optimize=True,
    )

    return output.getvalue()


# ============================================================
# 6. 下载图片并执行过滤
# ============================================================

def is_image_message(message):
    """判断消息是否包含可处理的图片。"""
    if getattr(message, "photo", None):
        return True

    document = getattr(message, "document", None)

    if document:
        mime_type = getattr(document, "mime_type", "") or ""

        if mime_type.lower().startswith("image/"):
            return True

    return False


async def download_image_bytes(client, message):
    """将 Telegram 图片下载到临时文件，再读取为 bytes。"""
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="tg_image_",
            suffix=".img",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        result = await client.download_media(
            message,
            file=temp_path,
        )

        if not result or not os.path.exists(temp_path):
            return None

        with open(temp_path, "rb") as file:
            return file.read()

    except Exception as exc:
        print(f"⚠️ 图片下载失败 message={message.id}: {exc}")
        return None

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def prepare_image(client, source_id, message):
    """
    检查图片文字、二维码、图片说明。
    通过检测后添加水印。
    返回处理后的图片字节；过滤或下载失败则返回 None。
    """
    caption = (getattr(message, "message", None) or "").strip()

    # 图片说明中有联系方式，跳过该图片
    contact_type = detect_contact_text(caption)

    if contact_type:
        print(
            f"   ❌ SKIP IMAGE message={message.id} "
            f"CAPTION CONTACT={contact_type}"
        )
        mark_processed(source_id, message.id)
        return None

    image_bytes = await download_image_bytes(client, message)

    if not image_bytes:
        print(f"   ❌ IMAGE DOWNLOAD FAILED message={message.id}")
        return None

    # 先检测二维码
    if detect_qr_code(image_bytes):
        print(f"   ❌ SKIP IMAGE message={message.id} QR CODE DETECTED")
        mark_processed(source_id, message.id)
        return None

    # 再进行 OCR 联系方式识别
    contact_type = detect_contact_in_image(image_bytes)

    if contact_type:
        print(
            f"   ❌ SKIP IMAGE message={message.id} "
            f"CONTACT DETECTED={contact_type}"
        )
        mark_processed(source_id, message.id)
        return None

    try:
        watermarked = add_watermark(image_bytes)
    except Exception as exc:
        print(f"   ❌ WATERMARK FAILED message={message.id}: {exc}")
        return None

    print(f"   ✅ IMAGE PASSED message={message.id}")

    return {
        "message": message,
        "image_bytes": watermarked,
        "caption": caption,
    }


# ============================================================
# 7. Bot 发送文字和单张图片
# ============================================================

async def bot_send_text(bot, target, text):
    """发送纯文字消息，超过单条长度时分段发送。"""
    text = (text or "").strip()

    if not text:
        return True

    if detect_contact_text(text):
        print("   ❌ SKIP TEXT: CONTACT INFORMATION DETECTED")
        return False

    chunks = [
        text[i:i + 4000]
        for i in range(0, len(text), 4000)
    ]

    for chunk in chunks:
        sent = False

        for attempt in range(1, SEND_RETRIES + 1):
            try:
                await bot.send_message(
                    chat_id=target,
                    text=chunk,
                )
                sent = True
                break

            except Exception as exc:
                print(
                    f"   ⚠️ BOT TEXT RETRY {attempt}/{SEND_RETRIES}: {exc}"
                )

                if attempt < SEND_RETRIES:
                    await asyncio.sleep(attempt * 2)

        if not sent:
            return False

    return True


async def bot_send_photo(bot, target, image_bytes, caption=""):
    """使用 Bot API 发送单张图片。"""
    for attempt in range(1, SEND_RETRIES + 1):
        try:
            file_obj = InputFile(
                io.BytesIO(image_bytes),
                filename="85h_image.jpg",
            )

            await bot.send_photo(
                chat_id=target,
                photo=file_obj,
                caption=(caption[:1024] if caption else None),
            )

            return True

        except Exception as exc:
            print(
                f"   ⚠️ BOT PHOTO RETRY {attempt}/{SEND_RETRIES}: {exc}"
            )

            if attempt < SEND_RETRIES:
                await asyncio.sleep(attempt * 2)

    return False


# ============================================================
# 8. 相册上传：直接使用 httpx multipart
# ============================================================

async def bot_send_album_httpx(http_client, target, items):
    """
    使用 Telegram Bot API 的 sendMediaGroup 发送 2-10 张图片。

    每个 media 的 media 字段都使用 attach://<独立字段名>，
    并在 multipart 文件列表中提供同名字段。
    """
    if not 2 <= len(items) <= 10:
        raise ValueError("sendMediaGroup 每批必须包含 2 到 10 张图片")

    media = []
    files = []

    for index, item in enumerate(items):
        # 每张图片使用独立附件名称
        attach_name = f"album_image_{index}"

        filename = f"85h_album_{index}.jpg"

        media_item = {
            "type": "photo",
            "media": f"attach://{attach_name}",
        }

        caption = (item.get("caption") or "").strip()

        if caption:
            media_item["caption"] = caption[:1024]

        media.append(media_item)

        files.append(
            (
                attach_name,
                (
                    filename,
                    item["image_bytes"],
                    "image/jpeg",
                ),
            )
        )

    data = {
        "chat_id": str(target),
        "media": json.dumps(
            media,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMediaGroup"

    response = await http_client.post(
        url,
        data=data,
        files=files,
        timeout=120.0,
    )

    try:
        result = response.json()
    except Exception:
        raise RuntimeError(
            f"Telegram API 返回非 JSON 响应：HTTP {response.status_code}"
        )

    if response.status_code != 200 or not result.get("ok"):
        description = result.get("description", "未知错误")

        raise RuntimeError(
            f"HTTP {response.status_code}: {description}"
        )

    return result.get("result", [])


async def bot_send_album(http_client, bot, target, items):
    """
    分批发送相册。
    每批成功后立即记录已发送的消息，避免下一次重复发送成功批次。
    """
    if not items:
        return True

    total = len(items)

    # Telegram 的相册接口每批支持 2-10 张；
    # 只有 1 张时改用 send_photo。
    batches = [
        items[i:i + ALBUM_BATCH_SIZE]
        for i in range(0, total, ALBUM_BATCH_SIZE)
    ]

    for batch_index, batch in enumerate(batches, start=1):
        print(
            f"   📦 SEND ALBUM {batch_index}/{len(batches)} "
            f"({len(batch)} images)"
        )

        if len(batch) == 1:
            item = batch[0]

            success = await bot_send_photo(
                bot,
                target,
                item["image_bytes"],
                item.get("caption", ""),
            )

        else:
            success = False

            for attempt in range(1, SEND_RETRIES + 1):
                try:
                    await bot_send_album_httpx(
                        http_client,
                        target,
                        batch,
                    )

                    success = True
                    break

                except Exception as exc:
                    print(
                        f"   ⚠️ BOT ALBUM RETRY "
                        f"{attempt}/{SEND_RETRIES}: {exc}"
                    )

                    if attempt < SEND_RETRIES:
                        await asyncio.sleep(attempt * 2)

        if not success:
            print("   ❌ ALBUM SEND FAILED")
            return False

        print("   ✅ ALBUM BATCH SENT")

        # 该批发送成功后，立即记录这些图片对应的源消息
        source_id = batch[0]["source_id"]

        mark_messages_processed(
            source_id,
            [item["message"] for item in batch],
        )

    return True


# ============================================================
# 9. 来源频道消息扫描
# ============================================================

def get_today_beijing():
    return datetime.now(BEIJING_TZ).date()


async def collect_today_messages(client, source_entity, source_label):
    """
    采集某个来源频道当天发布的消息。
    Telegram 消息按发布时间从新到旧读取。
    """
    today = get_today_beijing()
    source_id = int(source_entity.id)

    collected = []

    limit = SCAN_LIMIT if SCAN_LIMIT > 0 else None

    print()
    print("=" * 60)
    print(f"🔎 SCANNING SOURCE: {source_label}")
    print("=" * 60)

    async for message in client.iter_messages(
        source_entity,
        limit=limit,
    ):
        if not message.date:
            continue

        message_date = message.date.astimezone(BEIJING_TZ).date()

        # 只采集北京时间当天的消息
        if message_date < today:
            break

        if message_date > today:
            continue

        collected.append(message)

    print(f"   📥 TODAY MESSAGES FOUND: {len(collected)}")

    return {
