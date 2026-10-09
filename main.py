import asyncio
import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytesseract

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from telethon import TelegramClient, utils
from telethon.sessions import StringSession
from telegram import Bot, InputFile
from telegram.error import TelegramError


# ============================================================
# 1. 基础配置
# ============================================================

BEIJING_TZ = timezone(timedelta(hours=8))

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
TELEGRAM_SESSION = os.environ["TELEGRAM_SESSION"]
BOT_TOKEN = os.environ["BOT_TOKEN"]

SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ["SOURCE_CHANNELS"].split(",")
    if x.strip()
]

TARGET_CHANNEL = os.environ["TARGET_CHANNEL"].strip()

SCAN_LIMIT = int(os.environ.get("SCAN_LIMIT", "0"))

PROCESSED_FILE = Path("processed.json")

WATERMARK_TEXT = "85H官方频道"

OCR_LANG = "eng+chi_sim+vie+por"

MAX_RETRIES = 3
RETRY_DELAY = 3

# 图片压缩设置
MAX_IMAGE_SIDE = 2000
JPEG_QUALITY = 88

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)

for logger_name in [
    "telethon",
    "telethon.network",
    "telethon.extensions",
    "httpx",
    "httpcore",
    "telegram",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


# ============================================================
# 2. 联系方式检测规则
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
        re.compile(r"微信|微信号|微[信號号]"),
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

PHONE_CONTACT_TYPES = {
    "CN_PHONE",
    "VN_PHONE",
}


def detect_text_contact(text):
    """检测文字中的联系方式。"""
    if not text:
        return None

    for name, pattern in CONTACT_PATTERNS:
        if pattern.search(text):
            return name

    return None


# ============================================================
# 3. processed.json 去重记录
# ============================================================

def load_processed():
    if not PROCESSED_FILE.exists():
        return set()

    try:
        data = json.loads(
            PROCESSED_FILE.read_text(encoding="utf-8")
        )

        messages = data.get("messages", [])

        if isinstance(messages, dict):
            messages = list(messages.keys())

        return {str(x) for x in messages}

    except Exception as exc:
        print(f"⚠️ processed.json 读取失败: {exc}")
        return set()


PROCESSED = load_processed()


def save_processed():
    """原子写入，避免文件写到一半中断。"""
    data = {
        "messages": sorted(PROCESSED),
        "updated_at": datetime.now(BEIJING_TZ).isoformat(),
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
    兼容当前格式和常见的旧版单消息 ID 格式。
    """
    return (
        message_key(source_id, message_id) in PROCESSED
        or str(message_id) in PROCESSED
    )


def mark_processed(source_id, message_id):
    PROCESSED.add(message_key(source_id, message_id))


# ============================================================
# 4. 图片下载与二维码检测
# ============================================================

async def download_image(client, message):
    try:
        data = await client.download_media(
            message,
            file=bytes,
        )

        if not data:
            print(f"   ❌ 图片下载失败 message={message.id}")
            return None

        return data

    except Exception as exc:
        print(
            f"   ❌ 图片下载异常 message={message.id}: {exc}"
        )
        return None


def image_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def detect_qr_code(image_bytes):
    """
    检测二维码。
    如果检测到二维码，即使二维码内容未能解码，也按有二维码处理。
    """
    try:
        image = image_from_bytes(image_bytes)

        image_array = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR,
        )

        detector = cv2.QRCodeDetector()

        try:
            detected, decoded_info, points, _ = (
                detector.detectAndDecodeMulti(image_array)
            )

            if detected and points is not None:
                return True

        except Exception:
            pass

        try:
            decoded, points, _ = detector.detectAndDecode(
                image_array
            )

            if points is not None:
                return True

            if decoded:
                return True

        except Exception:
            pass

        return False

    except Exception as exc:
        print(f"   ⚠️ 二维码检测异常: {exc}")
        return False


# ============================================================
# 5. OCR 联系方式检测
# ============================================================

def build_ocr_images(image):
    """生成少量 OCR 变体，提高识别率并控制运行时间。"""
    gray = ImageOps.grayscale(image)

    enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)

    threshold = enhanced.point(
        lambda p: 255 if p > 155 else 0
    )

    return [
        ("ORIGINAL", image),
        ("ENHANCED", enhanced),
        ("THRESHOLD", threshold),
    ]


def detect_image_contact(image_bytes):
    """
    对图片进行 OCR 联系方式检测。

    Telegram 用户名、微信等强联系方式匹配一次即拦截。
    电话号码需要在两次 OCR 结果中被识别，降低误判。
    """
    try:
        image = image_from_bytes(image_bytes)
        variants = build_ocr_images(image)

        phone_hits = set()

        for variant_name, variant in variants:
            try:
                text = pytesseract.image_to_string(
                    variant,
                    lang=OCR_LANG,
                    config="--psm 6",
                    timeout=20,
                )

            except Exception:
                try:
                    text = pytesseract.image_to_string(
                        variant,
                        lang="eng",
                        config="--psm 11",
                        timeout=20,
                    )
                except Exception:
                    continue

            for name, pattern in CONTACT_PATTERNS:
                if not pattern.search(text):
                    continue

                if name in STRONG_CONTACT_TYPES:
                    return name

                if name in PHONE_CONTACT_TYPES:
                    phone_hits.add(name)

            # 同时识别常见的联系方式提示词
            lower_text = text.lower()

            if (
                "whatsapp" in lower_text
                or "wechat" in lower_text
                or "微信号" in text
                or "微信" in text
            ):
                return "CONTACT_KEYWORD"

        if phone_hits:
            # 需要两个 OCR 结果确认，减少单次误识别
            # 由于这里按类别累计，只有不同电话号码类型均出现时
            # 才会达到两类；下面另外执行一次合并检查。
            all_text = " ".join(
                pytesseract.image_to_string(
                    variant,
                    lang="eng",
                    config="--psm 11",
                    timeout=20,
                )
                for _, variant in variants[:2]
            )

            for name, pattern in CONTACT_PATTERNS:
                if (
                    name in PHONE_CONTACT_TYPES
                    and pattern.search(all_text)
                ):
                    return name

        return None

    except Exception as exc:
        print(f"   ⚠️ OCR 检测异常: {exc}")
        return None


# ============================================================
# 6. 图片添加水印
# ============================================================

def find_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


FONT_PATH = find_font()


def add_watermark(image_bytes):
    image = image_from_bytes(image_bytes)

    # 控制图片尺寸，降低 Bot API 上传失败概率
    image.thumbnail(
        (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
        Image.Resampling.LANCZOS,
    )

    width, height = image.size

    font_size = max(18, int(min(width, height) * 0.035))

    try:
        if FONT_PATH:
            font = ImageFont.truetype(
                FONT_PATH,
                font_size,
            )
        else:
            font = ImageFont.load_default()

    except Exception:
        font = ImageFont.load_default()

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(overlay)

    bbox = draw.textbbox(
        (0, 0),
        WATERMARK_TEXT,
        font=font,
        stroke_width=1,
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = max(12, int(min(width, height) * 0.025))

    x = max(margin, width - text_width - margin)
    y = max(margin, height - text_height - margin)

    draw.text(
        (x, y),
        WATERMARK_TEXT,
        font=font,
        fill=(255, 255, 255, 185),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 145),
    )

    image = Image.alpha_composite(
        image.convert("RGBA"),
        overlay,
    ).convert("RGB")

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
    )

    return output.getvalue()


# ============================================================
# 7. Bot 发送文字与单图
# ============================================================

BOT_TARGET_CHANNEL = None


async def bot_send_text(bot, text):
    if not text:
        return False

    try:
        await bot.send_message(
            chat_id=BOT_TARGET_CHANNEL,
            text=text[:4096],
        )

        print("   ✅ 文字发送成功")
        return True

    except Exception as exc:
        print(f"   ❌ 文字发送失败: {exc}")
        return False


async def bot_send_photo(bot, image_bytes, caption=None):
    try:
        photo_file = InputFile(
            image_bytes,
            filename=f"photo_{int(time.time() * 1000)}.jpg",
        )

        kwargs = {
            "chat_id": BOT_TARGET_CHANNEL,
            "photo": photo_file,
        }

        if caption:
            kwargs["caption"] = caption[:1024]

        await bot.send_photo(**kwargs)

        print("   ✅ 单图发送成功")
        return True

    except Exception as exc:
        print(f"   ❌ 单图发送失败: {exc}")
        return False


# ============================================================
# 8. 核心修复：直接使用 HTTPX 上传相册
# ============================================================

async def send_album_httpx(items):
    """
    items 格式：

    [
        {
            "image_bytes": b"...",
            "caption": "..."
        },
        ...
    ]

    通过明确的 multipart 字段名绑定 attach://引用，
    避免出现 media not found。
    """

    if not items:
        return False

    if len(items) == 1:
        return await bot_send_photo(
            bot,
            items[0]["image_bytes"],
            items[0].get("caption"),
        )

    if len(items) > 10:
        raise ValueError(
            "单次 sendMediaGroup 最多发送 10 张图片"
        )

    media = []
    files = {}

    for index, item in enumerate(items):
        field_name = f"photo_{index}_{int(time.time() * 1000)}"
        filename = f"{field_name}.jpg"

        files[field_name] = (
            filename,
            item["image_bytes"],
            "image/jpeg",
        )

        media_item = {
            "type": "photo",
            "media": f"attach://{field_name}",
        }

        caption = item.get("caption")

        # 相册仅将第一张图片的文字作为相册说明
        if index == 0 and caption:
            media_item["caption"] = caption[:1024]

        media.append(media_item)

    request_data = {
        "chat_id": str(BOT_TARGET_CHANNEL),
        "media": json.dumps(
            media,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }

    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMediaGroup"
    )

    timeout = httpx.Timeout(
        120.0,
        connect=20.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
    ) as session:

        response = await session.post(
            url,
            data=request_data,
            files=files,
        )

    try:
        result = response.json()
    except Exception:
        result = {}

    if response.status_code != 200 or not result.get("ok"):
        description = result.get(
            "description",
            response.text[:1000],
        )

        raise RuntimeError(
            f"Telegram API {response.status_code}: {description}"
        )

    return True


async def bot_send_album(items):
    """
    相册最多 10 张一批。
    失败重试 3 次。
    如果剩余批次只有 1 张，则使用单图接口发送。
    """

    if not items:
        return False

    chunks = [
        items[index:index + 10]
        for index in range(0, len(items), 10)
    ]

    for chunk_index, chunk in enumerate(chunks, start=1):
        print(
            f"   📦 SEND ALBUM "
            f"{chunk_index}/{len(chunks)} "
            f"({len(chunk)} images)"
        )

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if len(chunk) == 1:
                    success = await bot_send_photo(
                        bot,
                        chunk[0]["image_bytes"],
                        chunk[0].get("caption"),
                    )

                    if not success:
                        raise RuntimeError("单图发送失败")

                else:
                    await send_album_httpx(chunk)
                    success = True

                break

            except Exception as exc:
                print(
                    f"   ⚠️ BOT ALBUM RETRY "
                    f"{attempt}/{MAX_RETRIES}: {exc}"
                )

                if attempt < MAX_RETRIES:
                    await asyncio.sleep(
                        RETRY_DELAY * attempt
                    )

        if not success:
            print("   ❌ ALBUM SEND FAILED")
            return False

    print("   ✅ ALBUM PUBLISH SUCCESS")
    return True


# ============================================================
# 9. 处理单条消息
# ============================================================

async def process_single_message(
    client,
    bot,
    source_entity,
    message,
):
    source_id = source_entity.id

    if is_processed(source_id, message.id):
        print(
            f"   ⏭️ SKIP PROCESSED message={message.id}"
        )
        return

    text = (message.message or "").strip()

    # 纯文字消息
    if not message.photo:
        if not text:
            mark_processed(source_id, message.id)
            save_processed()
            return

        contact = detect_text_contact(text)

        if contact:
            print(
                f"   ❌ SKIP TEXT: CONTACT DETECTED "
                f"{contact}"
            )

            mark_processed(source_id, message.id)
            save_processed()
            return

        success = await bot_send_text(bot, text)

        if success:
            mark_processed(source_id, message.id)
            save_processed()

        return

    # 单图消息
    if text and detect_text_contact(text):
        print(
            f"   ❌ SKIP IMAGE: CAPTION CONTACT DETECTED "
            f"message={message.id}"
        )

        mark_processed(source_id, message.id)
        save_processed()
        return

    image_bytes = await download_image(
        client,
        message,
    )

    if not image_bytes:
        return

    if detect_qr_code(image_bytes):
        print(
            f"   ❌ SKIP IMAGE: QR CODE DETECTED "
            f"message={message.id}"
        )

        mark_processed(source_id, message.id)
        save_processed()
        return

    contact = detect_image_contact(image_bytes)

    if contact:
        print(
            f"   ❌ SKIP IMAGE: OCR CONTACT DETECTED "
            f"{contact} message={message.id}"
        )

        mark_processed(source_id, message.id)
        save_processed()
        return

    try:
        processed_image = add_watermark(image_bytes)
    except Exception as exc:
        print(f"   ❌ 水印处理失败: {exc}")
        return

    success = await bot_send_photo(
        bot,
        processed_image,
        text or None,
    )

    if success:
        mark_processed(source_id, message.id)
        save_processed()


# ============================================================
# 10. 处理相册
# ============================================================

async def process_album(
    client,
    source_entity,
    album_messages,
):
    source_id = source_entity.id

    album_messages.sort(
        key=lambda message: (
            message.date,
            message.id,
        )
    )

    print(
        f"\n📚 ALBUM "
        f"MESSAGES={len(album_messages)}"
    )

    send_items = []
    rejected_ids = []
    pending_ids = []

    for message in album_messages:
        if is_processed(source_id, message.id):
            print(
                f"   ⏭️ SKIP PROCESSED "
                f"message={message.id}"
            )
            continue

        caption = (message.message or "").strip()

        # 相册中每张图片独立检查
        if caption and detect_text_contact(caption):
            print(
                f"   ❌ SKIP ALBUM IMAGE "
                f"CAPTION CONTACT message={message.id}"
            )

            rejected_ids.append(message.id)
            continue

        if not message.photo:
            # 相册中没有图片的项目不作为图片上传
            rejected_ids.append(message.id)
            continue

        image_bytes = await download_image(
            client,
            message,
        )

        if not image_bytes:
            print(
                f"   ⚠️ 图片下载失败，暂不标记 "
                f"message={message.id}"
            )
            continue

        if detect_qr_code(image_bytes):
            print(
                f"   ❌ SKIP ALBUM IMAGE "
                f"QR DETECTED message={message.id}"
            )

            rejected_ids.append(message.id)
            continue

        contact = detect_image_contact(image_bytes)

        if contact:
            print(
                f"   ❌ SKIP ALBUM IMAGE "
                f"OCR CONTACT {contact} "
                f"message={message.id}"
            )

            rejected_ids.append(message.id)
            continue

        try:
            watermarked = add_watermark(image_bytes)
        except Exception as exc:
            print(
                f"   ❌ 水印失败 message={message.id}: {exc}"
            )
            continue

        send_items.append(
            {
                "image_bytes": watermarked,
                "caption": caption or None,
                "message_id": message.id,
            }
        )

        pending_ids.append(message.id)

    # 所有图片都被过滤时，记录过滤结果
    if not send_items:
        for message_id in rejected_ids:
            mark_processed(source_id, message_id)

        save_processed()

        print("   ℹ️ 相册没有可发布图片")
        return

    print(
        f"   📦 SEND ALBUM ({len(send_items)} images)"
    )

    # sendMediaGroup 只能每批发送 2–10 张图片
    # 若只有 1 张，bot_send_album 会自动使用 send_photo
    success = await bot_send_album(send_items)

    if success:
        for message_id in pending_ids:
            mark_processed(source_id, message_id)

        for message_id in rejected_ids:
            mark_processed(source_id, message_id)

        save_processed()

        print("   ✅ ALBUM PUBLISH SUCCESS")

    else:
        # 只保存明确过滤掉的消息。
        # 上传失败的有效图片不标记为已发布。
        for message_id in rejected_ids:
            mark_processed(source_id, message_id)

        save_processed()

        print("   ❌ ALBUM PUBLISH FAILED")
        print("   ⚠️ 有效图片不会写入 processed.json")


# ============================================================
# 11. 获取北京时间当天消息
# ============================================================

async def collect_today_messages(
    client,
    source_entity,
    source_name,
):
    now = datetime.now(BEIJING_TZ)

    today_start = now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    today_end = today_start + timedelta(days=1)

    messages = []
    scanned = 0

    print(f"\n🔎 SCANNING SOURCE: {source_name}")
    print(f"   BEIJING DATE: {now:%Y-%m-%d}")
    print(f"   SCAN LIMIT: {SCAN_LIMIT or 'ALL TODAY'}")

    limit = SCAN_LIMIT if SCAN_LIMIT > 0 else None

    async for message in client.iter_messages(
        source_entity,
        limit=limit,
    ):
        if not message.date:
            continue

        message_time = message.date.astimezone(
            BEIJING_TZ
        )

        # iter_messages 默认从新到旧读取
        if message_time < today_start:
            break

        if message_time >= today_end:
            continue

        scanned += 1

        if not message.photo and not (message.message or "").strip():
            continue

        messages.append(message)

    messages.sort(
        key=lambda message: (
            message.date,
            message.id,
        )
    )

    print(f"   📥 TODAY MESSAGES: {len(messages)}")
    print(f"   🔍 SCANNED: {scanned}")

    return messages


# ============================================================
# 12. 解析目标频道
# ============================================================

async def resolve_target(client):
    target_entity = await client.get_entity(
        TARGET_CHANNEL
    )

    print("\n🎯 TARGET ENTITY")
    print(f"   TITLE: {getattr(target_entity, 'title', '')}")
    print(f"   ID: {target_entity.id}")
    print(f"   USERNAME: {getattr(target_entity, 'username', None)}")

    username = getattr(
        target_entity,
        "username",
        None,
    )

    if username:
        return target_entity, f"@{username}"

    # 私有频道必须使用 Telethon 解析出的完整 peer ID
    peer_id = utils.get_peer_id(target_entity)

    return target_entity, peer_id


# ============================================================
# 13. 检查 Bot 目标频道
# ============================================================

async def diagnose_bot_target(bot):
    global BOT_TARGET_CHANNEL

    print("\n🤖 BOT TARGET DIAGNOSTIC")

    me = await bot.get_me()

    print("   ✅ BOT TOKEN VALID")
    print(f"   BOT ID: {me.id}")
    print(f"   BOT USERNAME: @{me.username}")

    print(
        f"   BOT TARGET: {BOT_TARGET_CHANNEL}"
    )

    chat = await bot.get_chat(
        chat_id=BOT_TARGET_CHANNEL
    )

    print("   ✅ BOT CAN ACCESS TARGET")
    print(f"   CHAT ID: {chat.id}")
    print(f"   CHAT TYPE: {chat.type}")
    print(f"   CHAT TITLE: {chat.title}")

    return True


# ============================================================
# 14. 主运行逻辑
# ============================================================

async def main():
    global bot
    global BOT_TARGET_CHANNEL

    print("=" * 65)
    print("Telegram Channel Forwarder")
    print("=" * 65)

    print(f"SOURCE CHANNELS: {SOURCE_CHANNELS}")
    print(f"TARGET CHANNEL: {TARGET_CHANNEL}")
    print(
        "BEIJING DATE:",
        datetime.now(BEIJING_TZ).isoformat(),
    )

    if not SOURCE_CHANNELS:
        raise RuntimeError(
            "SOURCE_CHANNELS 未配置"
        )

    # StringSession 是 GitHub Actions 无本地 session 文件时的登录方式
    client = TelegramClient(
        StringSession(TELEGRAM_SESSION),
        API_ID,
        API_HASH,
    )

    print("\n🔌 CONNECTING TELEGRAM...")

    await client.connect()

    if not await client.is_user_authorized():
        await client.disconnect()

        raise RuntimeError(
            "Telegram Session 无效或已过期，请重新生成 TELEGRAM_SESSION"
        )

    print("✅ TELEGRAM CONNECTED")

    bot = Bot(token=BOT_TOKEN)
    await bot.initialize()

    try:
        target_entity, BOT_TARGET_CHANNEL = await resolve_target(
            client
        )

        await diagnose_bot_target(bot)

        # 保存源频道消息，随后按时间排序发送
        all_events = []

        for source_name in SOURCE_CHANNELS:
            try:
                source_entity = await client.get_entity(
                    source_name
                )

                print(
                    f"\n📡 SOURCE ENTITY: "
                    f"{getattr(source_entity, 'title', source_name)}"
                )

                messages = await collect_today_messages(
                    client,
                    source_entity,
                    source_name,
                )

                # 先按 grouped_id 归类相册
                album_groups = defaultdict(list)
                standalone = []

                for message in messages:
                    if getattr(message, "grouped_id", None):
                        album_groups[
                            message.grouped_id
                        ].append(message)
                    else:
                        standalone.append(message)

                for message in standalone:
                    all_events.append(
                        {
                            "date": message.date,
                            "source_name": source_name,
                            "source_entity": source_entity,
                            "type": "single",
                            "messages": [message],
                        }
                    )

                for grouped_id, group_messages in album_groups.items():
                    group_messages.sort(
                        key=lambda message: (
                            message.date,
                            message.id,
                        )
                    )

                    all_events.append(
                        {
                            "date": group_messages[0].date,
                            "source_name": source_name,
                            "source_entity": source_entity,
                            "type": "album",
                            "messages": group_messages,
                            "grouped_id": grouped_id,
                        }
                    )

            except Exception as exc:
                print(
                    f"❌ SOURCE SCAN FAILED "
                    f"{source_name}: {exc}"
                )

        # 所有来源按实际发布时间排序
        all_events.sort(
            key=lambda event: (
                event["date"],
                SOURCE_CHANNELS.index(
                    event["source_name"]
                ),
                event["messages"][0].id,
            )
        )

        print("\n" + "=" * 65)
        print("📤 PROCESSING MESSAGES")
        print("=" * 65)

        seen_albums = set()

        for event in all_events:
            source_entity = event["source_entity"]
            messages = event["messages"]

            if event["type"] == "album":
                album_key = (
                    source_entity.id,
                    event["grouped_id"],
                )

                if album_key in seen_albums:
                    continue

                seen_albums.add(album_key)

                await process_album(
                    client,
                    source_entity,
                    messages,
                )

            else:
                await process_single_message(
                    client,
                    bot,
                    source_entity,
                    messages[0],
                )

        save_processed()

        print("\n" + "=" * 65)
        print("✅ FORWARDING RUN FINISHED")
        print(f"PROCESSED RECORDS: {len(PROCESSED)}")
        print("=" * 65)

    finally:
        await bot.shutdown()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
