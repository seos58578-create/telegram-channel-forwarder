import os
import re
import json
import time
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont

from telethon import TelegramClient
from telethon.sessions import StringSession

from telegram import Bot
from telegram.error import TimedOut, NetworkError


# ============================================================
# 基础配置
# ============================================================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]

# 这里必须是 Telethon StringSession
TELEGRAM_SESSION = os.environ["TELEGRAM_SESSION"]

BOT_TOKEN = os.environ["BOT_TOKEN"]

SOURCE_CHANNELS = [
    x.strip().lstrip("@")
    for x in os.environ["SOURCE_CHANNELS"].split(",")
    if x.strip()
]

TARGET_CHANNEL = os.environ["TARGET_CHANNEL"].strip()

SCAN_LIMIT_RAW = os.environ.get("SCAN_LIMIT", "0").strip()

try:
    SCAN_LIMIT = int(SCAN_LIMIT_RAW)
except Exception:
    SCAN_LIMIT = 0


# ============================================================
# 时区：北京时间
# ============================================================

BEIJING_TZ = timezone(timedelta(hours=8))


# ============================================================
# 文件
# ============================================================

PROCESSED_FILE = Path("processed.json")


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
        "WECHAT",
        re.compile(
            r"\bwechat\b",
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
    "WECHAT",
}


PHONE_CONTACT_TYPES = {
    "CN_PHONE",
    "VN_PHONE",
}


# ============================================================
# 工具函数
# ============================================================

def print_line(char="=", length=70):
    print(char * length)


def now_beijing():
    return datetime.now(BEIJING_TZ)


def message_date_beijing(message):
    """
    将 Telegram message.date 转换为北京时间
    """
    dt = message.date

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(BEIJING_TZ)


def is_today_beijing(message):
    """
    判断 Telegram 消息是否为北京时间今天
    """
    return message_date_beijing(message).date() == now_beijing().date()


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

        return set(str(x) for x in messages)

    except Exception as e:
        print(f"⚠️ processed.json 读取失败：{e}")
        return set()


def save_processed(processed):
    data = {
        "messages": sorted(
            processed,
            key=lambda x: int(x) if str(x).isdigit() else str(x)
        )[-10000:]
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


# ============================================================
# 联系方式检测
# ============================================================

def detect_contact_text(text):
    """
    检测文字中的联系方式

    返回：
    None
    或：
    {
        "type": "...",
        "match": "..."
    }
    """

    if not text:
        return None

    text = str(text)

    for contact_type, pattern in CONTACT_PATTERNS:

        match = pattern.search(text)

        if match:
            return {
                "type": contact_type,
                "match": match.group(0),
            }

    return None


# ============================================================
# OCR
# ============================================================

def normalize_ocr_text(text):
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    return text


def ocr_image_variants(image_path):
    """
    对图片进行多种 OCR
    """

    results = []

    try:
        image = cv2.imread(str(image_path))

        if image is None:
            return results

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        original = image

        text = pytesseract.image_to_string(
            original,
            lang="eng+chi_sim+vie+por",
        )

        results.append(
            ("ORIGINAL", normalize_ocr_text(text))
        )

        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        text = pytesseract.image_to_string(
            gray,
            lang="eng+chi_sim+vie+por",
        )

        results.append(
            ("GRAYSCALE", normalize_ocr_text(text))
        )

        # ----------------------------------------------------
        # ENHANCED
        # ----------------------------------------------------

        enhanced = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        enhanced = cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0,
        )

        text = pytesseract.image_to_string(
            enhanced,
            lang="eng+chi_sim+vie+por",
        )

        results.append(
            ("ENHANCED", normalize_ocr_text(text))
        )

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        threshold = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        text = pytesseract.image_to_string(
            threshold,
            lang="eng+chi_sim+vie+por",
        )

        results.append(
            ("THRESHOLD", normalize_ocr_text(text))
        )

    except Exception as e:
        print(f"⚠️ OCR ERROR: {e}")

    return results


def detect_contact_from_ocr(image_path):
    """
    OCR 联系方式识别策略：

    强联系方式：
    任意一次 OCR 识别出来就跳过

    手机号：
    至少两种 OCR 结果识别到才跳过
    """

    ocr_results = ocr_image_variants(image_path)

    if not ocr_results:
        return None

    phone_matches = []

    # --------------------------------------------------------
    # 强联系方式
    # --------------------------------------------------------

    for variant, text in ocr_results:

        contact = detect_contact_text(text)

        if not contact:
            continue

        contact_type = contact["type"]

        print(
            f"   OCR {variant}: "
            f"{contact_type} -> {contact['match']}"
        )

        if contact_type in STRONG_CONTACT_TYPES:

            return contact

        if contact_type in PHONE_CONTACT_TYPES:

            phone_matches.append(
                (
                    variant,
                    contact,
                )
            )

    # --------------------------------------------------------
    # 手机号至少两次识别
    # --------------------------------------------------------

    unique_types = set()

    for variant, contact in phone_matches:

        unique_types.add(
            contact["match"]
        )

    if len(unique_types) >= 1 and len(phone_matches) >= 2:

        return phone_matches[0][1]

    return None


# ============================================================
# QR CODE
# ============================================================

def detect_qr_code(image_path):
    """
    OpenCV QRCodeDetector

    同时支持单个 / 多个二维码
    """

    try:

        image = cv2.imread(str(image_path))

        if image is None:
            return None

        detector = cv2.QRCodeDetector()

        # ----------------------------------------------------
        # 多二维码
        # ----------------------------------------------------

        try:

            result = detector.detectAndDecodeMulti(image)

            if result and len(result) == 4:

                success, decoded_info, points, _ = result

                if success:

                    for value in decoded_info:

                        if value:
                            return value

        except Exception:
            pass

        # ----------------------------------------------------
        # 单二维码
        # ----------------------------------------------------

        try:

            data, points, _ = detector.detectAndDecode(
                image
            )

            if data:
                return data

        except Exception:
            pass

    except Exception as e:

        print(
            f"⚠️ QR DETECTION ERROR: {e}"
        )

    return None


# ============================================================
# 图片联系方式检测
# ============================================================

def inspect_image(image_path):
    """
    检测顺序：

    1. QR
    2. OCR
    """

    print("   🔍 CHECK QR...")

    qr_result = detect_qr_code(
        image_path
    )

    if qr_result:

        print(
            f"   ❌ QR CODE FOUND: {qr_result}"
        )

        return {
            "blocked": True,
            "type": "QR_CODE",
            "match": qr_result,
        }

    print("   🔍 CHECK OCR...")

    contact = detect_contact_from_ocr(
        image_path
    )

    if contact:

        print(
            f"   ❌ CONTACT FOUND: "
            f"{contact['type']} -> "
            f"{contact['match']}"
        )

        return {
            "blocked": True,
            "type": contact["type"],
            "match": contact["match"],
        }

    print("   ✅ NO CONTACT FOUND")

    return {
        "blocked": False,
        "type": None,
        "match": None,
    }


# ============================================================
# 水印
# ============================================================

def get_watermark_font(size=36):

    possible_fonts = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]

    for font_path in possible_fonts:

        if os.path.exists(font_path):

            try:

                return ImageFont.truetype(
                    font_path,
                    size=size,
                )

            except Exception:
                pass

    return ImageFont.load_default()


def add_watermark(
    input_path,
    output_path,
    text="",
):
    """
    给图片右下角增加水印
    """

    try:

        image = Image.open(
            input_path
        ).convert("RGBA")

        width, height = image.size

        font_size = max(
            24,
            min(width, height) // 18
        )

        font = get_watermark_font(
            font_size
        )

        draw = ImageDraw.Draw(
            image
        )

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        margin = max(
            15,
            min(width, height) // 50
        )

        x = width - text_width - margin
        y = height - text_height - margin

        # 半透明背景
        padding = 10

        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0),
        )

        overlay_draw = ImageDraw.Draw(
            overlay
        )

        overlay_draw.rounded_rectangle(
            [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding,
            ],
            radius=8,
            fill=(0, 0, 0, 120),
        )

        image = Image.alpha_composite(
            image,
            overlay,
        )

        draw = ImageDraw.Draw(
            image
        )

        draw.text(
            (x, y),
            text,
            font=font,
            fill=(255, 255, 255, 230),
        )

        image.convert("RGB").save(
            output_path,
            quality=95,
        )

        return True

    except Exception as e:

        print(
            f"❌ WATERMARK ERROR: {e}"
        )

        return False


# ============================================================
# Telethon Client
# ============================================================

client = TelegramClient(
    StringSession(TELEGRAM_SESSION),
    API_ID,
    API_HASH,
)


# ============================================================
# Bot
# ============================================================

bot = Bot(
    token=BOT_TOKEN
)


# ============================================================
# Bot 发送文字
# ============================================================

async def bot_send_text(text):
    """
    Bot 发送文字
    """

    for attempt in range(1, 4):

        try:

            await bot.send_message(
                chat_id=TARGET_CHANNEL,
                text=text,
            )

            return True

        except (
            TimedOut,
            NetworkError,
        ) as e:

            print(
                f"⚠️ BOT TEXT RETRY "
                f"{attempt}/3: {e}"
            )

            await asyncio.sleep(
                attempt * 3
            )

        except Exception as e:

            print(
                f"❌ BOT TEXT ERROR: {e}"
            )

            return False

    return False


# ============================================================
# Bot 发送单图片
# ============================================================

async def bot_send_photo(
    image_path,
    caption=None,
):
    """
    Bot 发送单张图片
    """

    for attempt in range(1, 4):

        try:

            with open(
                image_path,
                "rb",
            ) as photo:

                await bot.send_photo(
                    chat_id=TARGET_CHANNEL,
                    photo=photo,
                    caption=caption or None,
                )

            return True

        except (
            TimedOut,
            NetworkError,
        ) as e:

            print(
                f"⚠️ BOT PHOTO RETRY "
                f"{attempt}/3: {e}"
            )

            await asyncio.sleep(
                attempt * 3
            )

        except Exception as e:

            print(
                f"❌ BOT PHOTO ERROR: {e}"
            )

            return False

    return False


# ============================================================
# Bot 发送相册
# ============================================================

async def bot_send_media_group(
    image_items,
):
    """
    image_items:

    [
        {
            "path": "...",
            "caption": "..."
        }
    ]

    Telegram media group 一次最多 10 张
    """

    if not image_items:
        return False

    # Telegram 一组最多 10 个媒体
    chunks = [
        image_items[i:i + 10]
        for i in range(
            0,
            len(image_items),
            10
        )
    ]

    for chunk_index, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"   📦 SEND ALBUM "
            f"{chunk_index}/{len(chunks)} "
            f"({len(chunk)} images)"
        )

        for attempt in range(1, 4):

            files = []

            try:

                from telegram import (
                    InputMediaPhoto
                )

                media = []

                for index, item in enumerate(
                    chunk
                ):

                    f = open(
                        item["path"],
                        "rb",
                    )

                    files.append(f)

                    caption = None

                    # Telegram 相册建议只给第一张图 caption
                    if index == 0:

                        caption = (
                            item.get("caption")
                            or None
                        )

                        # Bot API caption 最多 1024
                        if caption:
                            caption = caption[:1024]

                    media.append(
                        InputMediaPhoto(
                            media=f,
                            caption=caption,
                        )
                    )

                await bot.send_media_group(
                    chat_id=TARGET_CHANNEL,
                    media=media,
                )

                for f in files:
                    try:
                        f.close()
                    except Exception:
                        pass

                return True

            except (
                TimedOut,
                NetworkError,
            ) as e:

                print(
                    f"⚠️ BOT ALBUM RETRY "
                    f"{attempt}/3: {e}"
                )

                for f in files:
                    try:
                        f.close()
                    except Exception:
                        pass

                await asyncio.sleep(
                    attempt * 3
                )

            except Exception as e:

                print(
                    f"❌ BOT ALBUM ERROR: {e}"
                )

                for f in files:
                    try:
                        f.close()
                    except Exception:
                        pass

                return False

    return False


# ============================================================
# 获取 Telegram Entity
# ============================================================

async def resolve_source(channel):
    try:

        entity = await client.get_entity(
            channel
        )

        return entity

    except Exception as e:

        print(
            f"❌ SOURCE RESOLVE ERROR "
            f"{channel}: {e}"
        )

        return None


# ============================================================
# 获取今天的消息
# ============================================================

async def collect_today_messages(
    processed,
):
    """
    从所有源频道读取北京时间今天消息

    返回：

    [
        {
            source_index,
            source_name,
            entity,
            message
        }
    ]
    """

    all_messages = []

    print_line()
    print("COLLECT TODAY MESSAGES")
    print_line()

    for source_index, source in enumerate(
        SOURCE_CHANNELS
    ):

        print(
            f"\n📡 SOURCE "
            f"{source_index + 1}: "
            f"{source}"
        )

        entity = await resolve_source(
            source
        )

        if entity is None:
            continue

        count = 0

        async for message in client.iter_messages(
            entity,
            limit=None if SCAN_LIMIT <= 0 else SCAN_LIMIT,
        ):

            # Telegram 消息按倒序读取
            if not is_today_beijing(
                message
            ):

                # 如果已经进入更早日期，
                # 可以停止当前频道继续读取
                message_time = (
                    message_date_beijing(
                        message
                    )
                )

                if (
                    message_time.date()
                    < now_beijing().date()
                ):
                    break

                continue

            # 只处理文字、图片
            if not (
                message.message
                or message.photo
            ):
                continue

            all_messages.append(
                {
                    "source_index": source_index,
                    "source_name": source,
                    "entity": entity,
                    "message": message,
                }
            )

            count += 1

        print(
            f"   TODAY MESSAGES: {count}"
        )

    # --------------------------------------------------------
    # 统一按发布时间排序
    # --------------------------------------------------------

    all_messages.sort(
        key=lambda item: (
            message_date_beijing(
                item["message"]
            ),
            item["source_index"],
            item["message"].id,
        )
    )

    print_line()
    print(
        f"TOTAL TODAY MESSAGES: "
        f"{len(all_messages)}"
    )
    print_line()

    return all_messages


# ============================================================
# 构建发布组
# ============================================================

def build_publish_groups(
    messages
):
    """
    相同 source + grouped_id 的消息
    视为一个 Telegram 相册

    返回：

    [
        {
            "type": "album",
            "items": [...]
        },

        {
            "type": "single",
            "item": {...}
        }
    ]
    """

    groups = []

    album_map = {}

    for item in messages:

        message = item["message"]

        grouped_id = getattr(
            message,
            "grouped_id",
            None,
        )

        # ----------------------------------------------------
        # 相册
        # ----------------------------------------------------

        if grouped_id:

            key = (
                item["source_index"],
                grouped_id,
            )

            if key not in album_map:

                album_map[key] = {
                    "type": "album",
                    "items": [],
                    "first_date": message_date_beijing(
                        message
                    ),
                    "first_id": message.id,
                }

                groups.append(
                    album_map[key]
                )

            album_map[key]["items"].append(
                item
            )

        else:

            groups.append(
                {
                    "type": "single",
                    "item": item,
                    "first_date": message_date_beijing(
                        message
                    ),
                    "first_id": message.id,
                }
            )

    # --------------------------------------------------------
    # 每个相册内部按 Telegram message ID 排序
    # --------------------------------------------------------

    for group in groups:

        if group["type"] == "album":

            group["items"].sort(
                key=lambda x: x["message"].id
            )

    # --------------------------------------------------------
    # 相册 / 单图整体重新按第一条消息时间排序
    # --------------------------------------------------------

    groups.sort(
        key=lambda x: (
            x["first_date"],
            x["first_id"],
        )
    )

    return groups


# ============================================================
# 下载图片
# ============================================================

async def download_photo(
    message,
    output_path,
):
    try:

        await client.download_media(
            message,
            file=str(output_path),
        )

        if not os.path.exists(
            output_path
        ):
            return False

        if os.path.getsize(
            output_path
        ) <= 0:
            return False

        return True

    except Exception as e:

        print(
            f"❌ DOWNLOAD ERROR: {e}"
        )

        return False


# ============================================================
# 处理单张图片
# ============================================================

async def prepare_image(
    message,
    work_dir,
):
    """
    返回：

    {
        blocked: True/False,
        reason: ...,
        output_path: ...
    }
    """

    message_id = message.id

    original_path = (
        Path(work_dir)
        / f"original_{message_id}.jpg"
    )

    watermark_path = (
        Path(work_dir)
        / f"watermark_{message_id}.jpg"
    )

    print(
        f"\n🖼️ MESSAGE {message_id}"
    )

    # --------------------------------------------------------
    # 先检测 caption
    # --------------------------------------------------------

    caption = (
        message.message
        or ""
    ).strip()

    if caption:

        caption_contact = (
            detect_contact_text(
                caption
            )
        )

        if caption_contact:

            print(
                f"   ❌ CAPTION CONTACT: "
                f"{caption_contact['type']} "
                f"-> "
                f"{caption_contact['match']}"
            )

            return {
                "blocked": True,
                "reason": "CAPTION_CONTACT",
                "output_path": None,
                "caption": caption,
            }

    # --------------------------------------------------------
    # 下载
    # --------------------------------------------------------

    print("   ⬇️ DOWNLOAD IMAGE...")

    downloaded = await download_photo(
        message,
        original_path,
    )

    if not downloaded:

        print(
            "   ❌ DOWNLOAD FAILED"
        )

        return {
            "blocked": False,
            "failed": True,
            "reason": "DOWNLOAD_FAILED",
            "output_path": None,
            "caption": caption,
        }

    print(
        f"   ✅ DOWNLOADED "
        f"{os.path.getsize(original_path)} bytes"
    )

    # --------------------------------------------------------
    # QR + OCR
    # --------------------------------------------------------

    inspection = inspect_image(
        original_path
    )

    if inspection["blocked"]:

        print(
            f"   🚫 IMAGE SKIPPED: "
            f"{inspection['type']}"
        )

        return {
            "blocked": True,
            "reason": inspection["type"],
            "output_path": None,
            "caption": caption,
        }

    # --------------------------------------------------------
    # 水印
    # --------------------------------------------------------

    print(
        "   💧 ADD WATERMARK..."
    )

    success = add_watermark(
        original_path,
        watermark_path,
        "85H官方频道",
    )

    if not success:

        print(
            "   ❌ WATERMARK FAILED"
        )

        return {
            "blocked": False,
            "failed": True,
            "reason": "WATERMARK_FAILED",
            "output_path": None,
            "caption": caption,
        }

    print(
        "   ✅ WATERMARK DONE"
    )

    return {
        "blocked": False,
        "failed": False,
        "reason": None,
        "output_path": str(
            watermark_path
        ),
        "caption": caption,
    }


# ============================================================
# 处理单条文字
# ============================================================

async def process_text(
    item,
    processed,
):
    message = item["message"]

    message_id = str(
        message.id
    )

    text = (
        message.message
        or ""
    ).strip()

    if not text:

        processed.add(
            message_id
        )

        return

    print_line("-")
    print(
        f"📝 TEXT MSG {message.id}"
    )

    contact = detect_contact_text(
        text
    )

    if contact:

        print(
            f"🚫 TEXT CONTACT FOUND: "
            f"{contact['type']} "
            f"-> "
            f"{contact['match']}"
        )

        processed.add(
            message_id
        )

        return

    print(
        "📤 BOT SEND TEXT..."
    )

    success = await bot_send_text(
        text
    )

    if success:

        print(
            "   ✅ TEXT PUBLISHED"
        )

        processed.add(
            message_id
        )

    else:

        print(
            "   ❌ TEXT PUBLISH FAILED"
        )


# ============================================================
# 处理单图
# ============================================================

async def process_single_image(
    item,
    processed,
    work_dir,
):
    message = item["message"]

    message_id = str(
        message.id
    )

    print_line("-")

    result = await prepare_image(
        message,
        work_dir,
    )

    # --------------------------------------------------------
    # 图片被过滤
    # --------------------------------------------------------

    if result["blocked"]:

        processed.add(
            message_id
        )

        return

    # --------------------------------------------------------
    # 图片处理失败
    # --------------------------------------------------------

    if result.get("failed"):

        print(
            "❌ IMAGE PROCESS FAILED"
        )

        return

    output_path = result[
        "output_path"
    ]

    caption = result.get(
        "caption"
    )

    print(
        "📤 BOT SEND PHOTO..."
    )

    success = await bot_send_photo(
        output_path,
        caption=caption,
    )

    if success:

        print(
            "   ✅ PHOTO PUBLISHED"
        )

        processed.add(
            message_id
        )

    else:

        print(
            "   ❌ PHOTO PUBLISH FAILED"
        )


# ============================================================
# 处理相册
# ============================================================

async def process_album(
    group,
    processed,
    work_dir,
):
    """
    一个 Telegram album：

    逐张图片检查。

    有联系方式：
        过滤该图片

    没联系方式：
        保留

    最终将所有合格图片一起发布。
    """

    items = group["items"]

    print_line("=")

    print(
        f"📚 TELEGRAM ALBUM"
    )

    print(
        f"   TOTAL IMAGES: "
        f"{len(items)}"
    )

    # --------------------------------------------------------
    # 相册准备
    # --------------------------------------------------------

    prepared = []

    for index, item in enumerate(
        items,
        start=1
    ):

        message = item["message"]

        print(
            f"\n   [{index}/{len(items)}] "
            f"MSG {message.id}"
        )

        result = await prepare_image(
            message,
            work_dir,
        )

        if result["blocked"]:

            print(
                f"   🚫 FILTERED "
                f"MSG {message.id}"
            )

            # 被过滤也要记录 processed
            processed.add(
                str(message.id)
            )

            continue

        if result.get("failed"):

            print(
                f"   ⚠️ FAILED "
                f"MSG {message.id}"
            )

            # 失败不记录 processed
            # 下次运行继续重试

            continue

        prepared.append(
            {
                "message_id": message.id,
                "path": result[
                    "output_path"
                ],
                "caption": result.get(
                    "caption"
                ),
            }
        )

    # --------------------------------------------------------
    # 没有合格图片
    # --------------------------------------------------------

    if not prepared:

        print(
            "\n🚫 ALBUM: NO VALID IMAGES"
        )

        return

    # --------------------------------------------------------
    # 如果只剩一张
    # --------------------------------------------------------

    if len(prepared) == 1:

        item = prepared[0]

        print(
            "\n📤 ALBUM -> SINGLE PHOTO"
        )

        success = await bot_send_photo(
            item["path"],
            caption=item.get(
                "caption"
            ),
        )

        if success:

            print(
                "   ✅ PHOTO PUBLISHED"
            )

            processed.add(
                str(
                    item["message_id"]
                )
            )

        else:

            print(
                "   ❌ PHOTO PUBLISH FAILED"
            )

        return

    # --------------------------------------------------------
    # 多张图片 → Telegram Album
    # --------------------------------------------------------

    print(
        f"\n📤 SEND "
        f"{len(prepared)} IMAGES "
        f"AS ALBUM..."
    )

    success = await bot_send_media_group(
        prepared
    )

    if success:

        print(
            "   ✅ ALBUM PUBLISHED"
        )

        for item in prepared:

            processed.add(
                str(
                    item["message_id"]
                )
            )

    else:

        print(
            "   ❌ ALBUM PUBLISH FAILED"
        )


# ============================================================
# 主程序
# ============================================================

async def main():

    print_line()
    print(
        "TELEGRAM AUTO FORWARDER"
    )
    print_line()

    print(
        "BEIJING TIME:",
        now_beijing()
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
        "SCAN LIMIT:",
        SCAN_LIMIT
    )

    processed = load_processed()

    print(
        "PROCESSED COUNT:",
        len(processed)
    )

    # ========================================================
    # 连接 Telegram
    # ========================================================

    print(
        "\nCONNECTING TELEGRAM..."
    )

    await client.start()

    print(
        "✅ TELEGRAM CONNECTED"
    )

    # ========================================================
    # 获取当前用户
    # ========================================================

    try:

        me = await client.get_me()

        print(
            f"TELEGRAM ACCOUNT: "
            f"{getattr(me, 'username', None) or me.id}"
        )

    except Exception as e:

        print(
            f"⚠️ GET ME ERROR: {e}"
        )

    # ========================================================
    # 获取目标 Entity
    # ========================================================

    try:

        target_entity = await client.get_entity(
            TARGET_CHANNEL
        )

        print(
            "\nTARGET ENTITY:",
            target_entity
        )

        print(
            "TARGET TITLE:",
            getattr(
                target_entity,
                "title",
                None,
            )
        )

        print(
            "TARGET ID:",
            getattr(
                target_entity,
                "id",
                None,
            )
        )

    except Exception as e:

        print(
            f"❌ TARGET ENTITY ERROR: {e}"
        )

        raise

    # --------------------------------------------------------
    # Bot 目标预检查不做 get_chat
    # --------------------------------------------------------

    print(
        "\nℹ️ BOT TARGET PRE-CHECK SKIPPED"
    )

    print(
        "程序将在实际发布时验证 Bot 权限"
    )

    # ========================================================
    # 获取今天消息
    # ========================================================

    messages = await collect_today_messages(
        processed
    )

    # ========================================================
    # 建立发布组
    # ========================================================

    groups = build_publish_groups(
        messages
    )

    print_line()

    print(
        f"PUBLISH GROUPS: "
        f"{len(groups)}"
    )

    print_line()

    # ========================================================
    # 创建临时目录
    # ========================================================

    with tempfile.TemporaryDirectory(
        prefix="telegram_forward_"
    ) as work_dir:

        print(
            "WORK DIR:",
            work_dir
        )

        # ====================================================
        # 按时间顺序发布
        # ====================================================

        for group_index, group in enumerate(
            groups,
            start=1
        ):

            print_line()

            print(
                f"PROCESS "
                f"{group_index}/"
                f"{len(groups)}"
            )

            # ------------------------------------------------
            # 单条消息
            # ------------------------------------------------

            if group["type"] == "single":

                item = group["item"]

                message = item[
                    "message"
                ]

                message_id = str(
                    message.id
                )

                # 已处理
                if message_id in processed:

                    print(
                        f"⏭️ MSG "
                        f"{message.id} "
                        f"ALREADY PROCESSED"
                    )

                    continue

                # 图片
                if message.photo:

                    await process_single_image(
                        item,
                        processed,
                        work_dir,
                    )

                # 文字
                else:

                    await process_text(
                        item,
                        processed,
                    )

            # ------------------------------------------------
            # 相册
            # ------------------------------------------------

            elif group["type"] == "album":

                # 如果相册中的所有消息
                # 都已经处理，则跳过

                unprocessed_items = [
                    item
                    for item in group["items"]
                    if str(
                        item["message"].id
                    ) not in processed
                ]

                if not unprocessed_items:

                    print(
                        "⏭️ ALBUM ALREADY PROCESSED"
                    )

                    continue

                # 注意：
                # 使用整个相册，而不是只处理第一张
                #
                # 这样可以确保一个 Telegram
                # 多图消息的每一张图片都被识别。

                await process_album(
                    group,
                    processed,
                    work_dir,
                )

            # ------------------------------------------------
            # 每处理一个组就保存一次
            # ------------------------------------------------

            save_processed(
                processed
            )

            # 防止过快请求
            await asyncio.sleep(
                0.5
            )

    # ========================================================
    # 最终保存
    # ========================================================

    save_processed(
        processed
    )

    # ========================================================
    # 完成
    # ========================================================

    print_line()
    print(
        "✅ ALL DONE"
    )

    print(
        "TODAY MESSAGES:",
        len(messages)
    )

    print(
        "PUBLISH GROUPS:",
        len(groups)
    )

    print(
        "PROCESSED COUNT:",
        len(processed)
    )

    print_line()

    # ========================================================
    # 关闭连接
    # ========================================================

    await client.disconnect()

    try:
        await bot.shutdown()
    except Exception:
        pass


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\n⚠️ PROGRAM STOPPED"
        )

    except Exception as e:

        print_line()

        print(
            "❌ FATAL ERROR"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        print_line()

        raise
