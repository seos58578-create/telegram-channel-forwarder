import os
import re
import json
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont

from telethon import TelegramClient
from telethon.sessions import StringSession

from telegram import Bot, InputMediaPhoto
from telegram.error import TimedOut, NetworkError


# ============================================================
# 基础配置
# ============================================================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]

# 必须是 Telethon StringSession
TELEGRAM_SESSION = os.environ["TELEGRAM_SESSION"]

BOT_TOKEN = os.environ["BOT_TOKEN"]

SOURCE_CHANNELS = [
    x.strip().lstrip("@")
    for x in os.environ["SOURCE_CHANNELS"].split(",")
    if x.strip()
]

TARGET_CHANNEL_RAW = os.environ["TARGET_CHANNEL"].strip()

SCAN_LIMIT_RAW = os.environ.get("SCAN_LIMIT", "0").strip()

try:
    SCAN_LIMIT = int(SCAN_LIMIT_RAW)
except Exception:
    SCAN_LIMIT = 0


# ============================================================
# 北京时间
# ============================================================

BEIJING_TZ = timezone(timedelta(hours=8))


# ============================================================
# 文件
# ============================================================

PROCESSED_FILE = Path("processed.json")


# ============================================================
# TARGET_CHANNEL 规范化
# ============================================================

def normalize_target_channel(value):
    """
    支持：

    @username
    username
    https://t.me/username

    私有频道：

    -100xxxxxxxxxx

    自动清理空格。
    """

    value = str(value).strip()

    if value.startswith("https://t.me/"):
        value = value.replace(
            "https://t.me/",
            "",
            1
        )

    if value.startswith("http://t.me/"):
        value = value.replace(
            "http://t.me/",
            "",
            1
        )

    value = value.strip("/")

    return value


TARGET_CHANNEL = normalize_target_channel(
    TARGET_CHANNEL_RAW
)


# ============================================================
# 联系方式
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
# 打印
# ============================================================

def print_line(char="=", length=70):
    print(char * length)


# ============================================================
# 北京时间
# ============================================================

def now_beijing():
    return datetime.now(
        BEIJING_TZ
    )


def message_date_beijing(message):
    dt = message.date

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt.astimezone(
        BEIJING_TZ
    )


def is_today_beijing(message):
    return (
        message_date_beijing(
            message
        ).date()
        == now_beijing().date()
    )


# ============================================================
# processed.json
# ============================================================

def load_processed():

    if not PROCESSED_FILE.exists():

        return set()

    try:

        with open(
            PROCESSED_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        messages = data.get(
            "messages",
            []
        )

        return {
            str(x)
            for x in messages
        }

    except Exception as e:

        print(
            f"⚠️ processed.json 读取失败：{e}"
        )

        return set()


def save_processed(processed):
    """
    修复：
    TypeError: '<' not supported
    between instances of 'int' and 'str'

    统一转字符串，再按照数字 ID 排序。
    """

    normalized = {
        str(x)
        for x in processed
    }

    def sort_key(value):

        try:
            return (
                0,
                int(value)
            )

        except Exception:

            return (
                1,
                str(value)
            )

    sorted_messages = sorted(
        normalized,
        key=sort_key
    )

    data = {
        "messages": sorted_messages[-10000:]
    }

    tmp_file = PROCESSED_FILE.with_suffix(
        ".tmp"
    )

    with open(
        tmp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    tmp_file.replace(
        PROCESSED_FILE
    )


# ============================================================
# 文字联系方式检测
# ============================================================

def detect_contact_text(text):

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

    text = text.replace(
        "\n",
        " "
    )

    text = text.replace(
        "\r",
        " "
    )

    return text


def ocr_image_variants(
    image_path
):

    results = []

    try:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            return results

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        text = pytesseract.image_to_string(
            image,
            lang="eng+chi_sim+vie+por",
        )

        results.append(
            (
                "ORIGINAL",
                normalize_ocr_text(text)
            )
        )

        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        text = pytesseract.image_to_string(
            gray,
            lang="eng+chi_sim+vie+por",
        )

        results.append(
            (
                "GRAYSCALE",
                normalize_ocr_text(text)
            )
        )

        # ----------------------------------------------------
        # ENHANCED
        # ----------------------------------------------------

        enhanced = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        enhanced = cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0
        )

        text = pytesseract.image_to_string(
            enhanced,
            lang="eng+chi_sim+vie+por",
        )

        results.append(
            (
                "ENHANCED",
                normalize_ocr_text(text)
            )
        )

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        threshold = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )[1]

        text = pytesseract.image_to_string(
            threshold,
            lang="eng+chi_sim+vie+por",
        )

        results.append(
            (
                "THRESHOLD",
                normalize_ocr_text(text)
            )
        )

    except Exception as e:

        print(
            f"⚠️ OCR ERROR: {e}"
        )

    return results


def detect_contact_from_ocr(
    image_path
):

    ocr_results = ocr_image_variants(
        image_path
    )

    if not ocr_results:

        return None

    phone_matches = []

    # --------------------------------------------------------
    # 强联系方式
    # --------------------------------------------------------

    for variant, text in ocr_results:

        contact = detect_contact_text(
            text
        )

        if not contact:

            continue

        contact_type = contact[
            "type"
        ]

        print(
            f"   OCR {variant}: "
            f"{contact_type} -> "
            f"{contact['match']}"
        )

        if contact_type in STRONG_CONTACT_TYPES:

            return contact

        if contact_type in PHONE_CONTACT_TYPES:

            phone_matches.append(
                (
                    variant,
                    contact
                )
            )

    # --------------------------------------------------------
    # 手机号至少识别 2 次
    # --------------------------------------------------------

    if len(phone_matches) >= 2:

        return phone_matches[0][1]

    return None


# ============================================================
# QR CODE
# ============================================================

def detect_qr_code(
    image_path
):

    try:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            return None

        detector = cv2.QRCodeDetector()

        # ----------------------------------------------------
        # 多二维码
        # ----------------------------------------------------

        try:

            result = (
                detector.detectAndDecodeMulti(
                    image
                )
            )

            if result and len(result) == 4:

                success = result[0]
                decoded_info = result[1]

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

            data, points, _ = (
                detector.detectAndDecode(
                    image
                )
            )

            if data:

                return data

        except Exception:
            pass

    except Exception as e:

        print(
            f"⚠️ QR ERROR: {e}"
        )

    return None


# ============================================================
# 图片检查
# ============================================================

def inspect_image(
    image_path
):

    print(
        "   🔍 CHECK QR..."
    )

    qr = detect_qr_code(
        image_path
    )

    if qr:

        print(
            f"   ❌ QR CODE FOUND: "
            f"{qr}"
        )

        return {
            "blocked": True,
            "type": "QR_CODE",
            "match": qr,
        }

    print(
        "   🔍 CHECK OCR..."
    )

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

    print(
        "   ✅ NO CONTACT FOUND"
    )

    return {
        "blocked": False,
        "type": None,
        "match": None,
    }


# ============================================================
# 水印字体
# ============================================================

def get_watermark_font(
    size=36
):

    fonts = [

        "/usr/share/fonts/opentype/noto/"
        "NotoSansCJK-Regular.ttc",

        "/usr/share/fonts/opentype/noto/"
        "NotoSansCJK-Bold.ttc",

        "/usr/share/fonts/truetype/noto/"
        "NotoSansCJK-Regular.ttc",

        "/usr/share/fonts/noto-cjk/"
        "NotoSansCJK-Regular.ttc",
    ]

    for path in fonts:

        if os.path.exists(path):

            try:

                return ImageFont.truetype(
                    path,
                    size=size
                )

            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# 水印
# ============================================================

def add_watermark(
    input_path,
    output_path,
    text=""
):

    try:

        image = Image.open(
            input_path
        ).convert(
            "RGBA"
        )

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
            font=font
        )

        text_width = (
            bbox[2] - bbox[0]
        )

        text_height = (
            bbox[3] - bbox[1]
        )

        margin = max(
            15,
            min(width, height) // 50
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

        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0)
        )

        overlay_draw = ImageDraw.Draw(
            overlay
        )

        padding = 10

        overlay_draw.rounded_rectangle(
            [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding,
            ],
            radius=8,
            fill=(0, 0, 0, 120)
        )

        image = Image.alpha_composite(
            image,
            overlay
        )

        draw = ImageDraw.Draw(
            image
        )

        draw.text(
            (x, y),
            text,
            font=font,
            fill=(255, 255, 255, 230)
        )

        image.convert(
            "RGB"
        ).save(
            output_path,
            quality=95
        )

        return True

    except Exception as e:

        print(
            f"❌ WATERMARK ERROR: {e}"
        )

        return False


# ============================================================
# Telethon
# ============================================================

client = TelegramClient(
    StringSession(
        TELEGRAM_SESSION
    ),
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

async def bot_send_text(
    text
):

    for attempt in range(
        1,
        4
    ):

        try:

            await bot.send_message(
                chat_id=TARGET_CHANNEL,
                text=text
            )

            return True

        except (
            TimedOut,
            NetworkError
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
                f"❌ BOT TEXT ERROR: "
                f"{e}"
            )

            # ------------------------------------------------
            # Chat not found
            # ------------------------------------------------

            if "Chat not found" in str(e):

                print()
                print(
                    "🚨 BOT 找不到目标频道"
                )

                print(
                    "请检查："
                )

                print(
                    "1. Bot 是否已经加入目标频道"
                )

                print(
                    "2. Bot 是否为管理员"
                )

                print(
                    "3. TARGET_CHANNEL 是否正确"
                )

                print(
                    "4. 私有频道请使用 -100xxxxxxxxxx"
                )

                print()

            return False

    return False


# ============================================================
# Bot 发送单图
# ============================================================

async def bot_send_photo(
    image_path,
    caption=None
):

    for attempt in range(
        1,
        4
    ):

        try:

            with open(
                image_path,
                "rb"
            ) as photo:

                await bot.send_photo(
                    chat_id=TARGET_CHANNEL,
                    photo=photo,
                    caption=caption or None
                )

            return True

        except (
            TimedOut,
            NetworkError
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
                f"❌ BOT PHOTO ERROR: "
                f"{e}"
            )

            if "Chat not found" in str(e):

                print()
                print(
                    "🚨 BOT 找不到目标频道"
                )

                print(
                    "请检查 TARGET_CHANNEL"
                )

                print(
                    "公开频道：@频道用户名"
                )

                print(
                    "私有频道：-100xxxxxxxxxx"
                )

                print(
                    "并确认 Bot 是频道管理员"
                )

                print()

            return False

    return False


# ============================================================
# Bot 发送相册
# ============================================================

async def bot_send_media_group(
    image_items
):

    if not image_items:

        return False

    # Telegram Bot API 每组最多 10 张
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
            f"{chunk_index}/"
            f"{len(chunks)} "
            f"({len(chunk)} images)"
        )

        for attempt in range(
            1,
            4
        ):

            files = []

            try:

                media = []

                for index, item in enumerate(
                    chunk
                ):

                    f = open(
                        item["path"],
                        "rb"
                    )

                    files.append(
                        f
                    )

                    caption = None

                    if index == 0:

                        caption = (
                            item.get(
                                "caption"
                            )
                            or None
                        )

                        if caption:

                            caption = (
                                caption[:1024]
                            )

                    media.append(
                        InputMediaPhoto(
                            media=f,
                            caption=caption
                        )
                    )

                await bot.send_media_group(
                    chat_id=TARGET_CHANNEL,
                    media=media
                )

                for f in files:

                    try:
                        f.close()
                    except Exception:
                        pass

                return True

            except (
                TimedOut,
                NetworkError
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
                    f"❌ BOT ALBUM ERROR: "
                    f"{e}"
                )

                for f in files:

                    try:
                        f.close()
                    except Exception:
                        pass

                if "Chat not found" in str(e):

                    print()
                    print(
                        "🚨 BOT 找不到目标频道"
                    )

                    print(
                        "请检查 TARGET_CHANNEL"
                    )

                    print(
                        "公开频道：@频道用户名"
                    )

                    print(
                        "私有频道：-100xxxxxxxxxx"
                    )

                    print(
                        "并确认 Bot 是管理员"
                    )

                    print()

                return False

    return False


# ============================================================
# 解析源频道
# ============================================================

async def resolve_source(
    channel
):

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
# 获取今天消息
# ============================================================

async def collect_today_messages(
    processed
):

    all_messages = []

    print_line()
    print(
        "COLLECT TODAY MESSAGES"
    )
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
            limit=(
                None
                if SCAN_LIMIT <= 0
                else SCAN_LIMIT
            )
        ):

            if not is_today_beijing(
                message
            ):

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

            if not (
                message.message
                or message.photo
            ):

                continue

            all_messages.append(
                {
                    "source_index":
                        source_index,

                    "source_name":
                        source,

                    "entity":
                        entity,

                    "message":
                        message,
                }
            )

            count += 1

        print(
            f"   TODAY MESSAGES: "
            f"{count}"
        )

    # --------------------------------------------------------
    # 多频道统一时间排序
    # --------------------------------------------------------

    all_messages.sort(
        key=lambda item: (
            message_date_beijing(
                item["message"]
            ),
            item["source_index"],
            item["message"].id
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
# 相册分组
# ============================================================

def build_publish_groups(
    messages
):

    groups = []

    album_map = {}

    for item in messages:

        message = item["message"]

        grouped_id = getattr(
            message,
            "grouped_id",
            None
        )

        if grouped_id:

            key = (
                item["source_index"],
                grouped_id
            )

            if key not in album_map:

                album_map[key] = {
                    "type":
                        "album",

                    "items":
                        [],

                    "first_date":
                        message_date_beijing(
                            message
                        ),

                    "first_id":
                        message.id,
                }

                groups.append(
                    album_map[key]
                )

            album_map[key][
                "items"
            ].append(
                item
            )

        else:

            groups.append(
                {
                    "type":
                        "single",

                    "item":
                        item,

                    "first_date":
                        message_date_beijing(
                            message
                        ),

                    "first_id":
                        message.id,
                }
            )

    # --------------------------------------------------------
    # 相册内部排序
    # --------------------------------------------------------

    for group in groups:

        if group["type"] == "album":

            group["items"].sort(
                key=lambda x:
                    x["message"].id
            )

    # --------------------------------------------------------
    # 整体排序
    # --------------------------------------------------------

    groups.sort(
        key=lambda x: (
            x["first_date"],
            x["first_id"]
        )
    )

    return groups


# ============================================================
# 下载图片
# ============================================================

async def download_photo(
    message,
    output_path
):

    try:

        await client.download_media(
            message,
            file=str(
                output_path
            )
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
            f"❌ DOWNLOAD ERROR: "
            f"{e}"
        )

        return False


# ============================================================
# 准备图片
# ============================================================

async def prepare_image(
    message,
    work_dir
):

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
        f"\n🖼 MESSAGE {message_id}"
    )

    # --------------------------------------------------------
    # Caption 检测
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
                "failed": False,
                "reason":
                    "CAPTION_CONTACT",
                "output_path": None,
                "caption": caption
            }

    # --------------------------------------------------------
    # 下载
    # --------------------------------------------------------

    print(
        "   ⬇️ DOWNLOAD IMAGE..."
    )

    downloaded = await download_photo(
        message,
        original_path
    )

    if not downloaded:

        print(
            "   ❌ DOWNLOAD FAILED"
        )

        return {
            "blocked": False,
            "failed": True,
            "reason":
                "DOWNLOAD_FAILED",
            "output_path": None,
            "caption": caption
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
            "failed": False,
            "reason":
                inspection["type"],
            "output_path": None,
            "caption": caption
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
        "85H官方频道"
    )

    if not success:

        print(
            "   ❌ WATERMARK FAILED"
        )

        return {
            "blocked": False,
            "failed": True,
            "reason":
                "WATERMARK_FAILED",
            "output_path": None,
            "caption": caption
        }

    print(
        "   ✅ WATERMARK DONE"
    )

    return {
        "blocked": False,
        "failed": False,
        "reason": None,
        "output_path":
            str(watermark_path),
        "caption": caption
    }


# ============================================================
# 处理文字
# ============================================================

async def process_text(
    item,
    processed
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
        f"📝 TEXT MSG "
        f"{message.id}"
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
    work_dir
):

    message = item["message"]

    message_id = str(
        message.id
    )

    print_line("-")

    result = await prepare_image(
        message,
        work_dir
    )

    # --------------------------------------------------------
    # 被过滤
    # --------------------------------------------------------

    if result["blocked"]:

        processed.add(
            message_id
        )

        return

    # --------------------------------------------------------
    # 处理失败
    # --------------------------------------------------------

    if result.get(
        "failed"
    ):

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
        caption
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
    work_dir
):

    items = group["items"]

    print_line("=")

    print(
        "📚 TELEGRAM ALBUM"
    )

    print(
        f"   TOTAL IMAGES: "
        f"{len(items)}"
    )

    prepared = []

    # --------------------------------------------------------
    # 每张图片单独识别
    # --------------------------------------------------------

    for index, item in enumerate(
        items,
        start=1
    ):

        message = item["message"]

        print(
            f"\n   [{index}/"
            f"{len(items)}] "
            f"MSG {message.id}"
        )

        # 已处理的单张图片不重复处理
        if str(
            message.id
        ) in processed:

            print(
                "   ⏭️ ALREADY PROCESSED"
            )

            continue

        result = await prepare_image(
            message,
            work_dir
        )

        # ----------------------------------------------------
        # 过滤
        # ----------------------------------------------------

        if result["blocked"]:

            print(
                f"   🚫 FILTERED "
                f"MSG {message.id}"
            )

            processed.add(
                str(message.id)
            )

            save_processed(
                processed
            )

            continue

        # ----------------------------------------------------
        # 失败
        # ----------------------------------------------------

        if result.get(
            "failed"
        ):

            print(
                f"   ⚠️ FAILED "
                f"MSG {message.id}"
            )

            continue

        # ----------------------------------------------------
        # 保留
        # ----------------------------------------------------

        prepared.append(
            {
                "message_id":
                    message.id,

                "path":
                    result[
                        "output_path"
                    ],

                "caption":
                    result.get(
                        "caption"
                    )
            }
        )

    # --------------------------------------------------------
    # 没有合格图片
    # --------------------------------------------------------

    if not prepared:

        print(
            "\n🚫 ALBUM: "
            "NO VALID IMAGES"
        )

        return

    # --------------------------------------------------------
    # 只剩一张
    # --------------------------------------------------------

    if len(prepared) == 1:

        item = prepared[0]

        print(
            "\n📤 ALBUM -> "
            "SINGLE PHOTO"
        )

        success = await bot_send_photo(
            item["path"],
            item.get("caption")
        )

        if success:

            print(
                "   ✅ PHOTO PUBLISHED"
            )

            processed.add(
                str(
                    item[
                        "message_id"
                    ]
                )
            )

            save_processed(
                processed
            )

        else:

            print(
                "   ❌ PHOTO PUBLISH FAILED"
            )

        return

    # --------------------------------------------------------
    # 多张图片
    # --------------------------------------------------------

    print(
        f"\n📤 SEND "
        f"{len(prepared)} "
        f"IMAGES AS ALBUM..."
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
                    item[
                        "message_id"
                    ]
                )
            )

        save_processed(
            processed
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

    print(
        "PROCESSED COUNT:",
        len(
            load_processed()
        )
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
    # 当前账号
    # ========================================================

    try:

        me = await client.get_me()

        print(
            "TELEGRAM ACCOUNT:",
            getattr(
                me,
                "username",
                None
            )
            or me.id
        )

    except Exception as e:

        print(
            f"⚠️ GET ME ERROR: {e}"
        )

    # ========================================================
    # 目标频道
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
                None
            )
        )

        print(
            "TARGET ID:",
            getattr(
                target_entity,
                "id",
                None
            )
        )

    except Exception as e:

        print(
            "\n❌ TARGET ENTITY ERROR:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise

    # ========================================================
    # Bot 目标预检查
    # ========================================================

    print(
        "\nℹ️ BOT TARGET PRE-CHECK SKIPPED"
    )

    print(
        "程序将在实际发布时验证 Bot 权限"
    )

    # ========================================================
    # 收集今天消息
    # ========================================================

    processed = load_processed()

    messages = await collect_today_messages(
        processed
    )

    # ========================================================
    # 构建发布组
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
    # 临时目录
    # ========================================================

    with tempfile.TemporaryDirectory(
        prefix="telegram_forward_"
    ) as work_dir:

        print(
            "WORK DIR:",
            work_dir
        )

        # ====================================================
        # 按时间顺序处理
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
            # 单条
            # ------------------------------------------------

            if group["type"] == "single":

                item = group["item"]

                message = item[
                    "message"
                ]

                message_id = str(
                    message.id
                )

                if message_id in processed:

                    print(
                        f"⏭️ MSG "
                        f"{message.id} "
                        f"ALREADY PROCESSED"
                    )

                    continue

                if message.photo:

                    await process_single_image(
                        item,
                        processed,
                        work_dir
                    )

                else:

                    await process_text(
                        item,
                        processed
                    )

            # ------------------------------------------------
            # 相册
            # ------------------------------------------------

            elif group["type"] == "album":

                unprocessed = [
                    item
                    for item
                    in group["items"]
                    if str(
                        item["message"].id
                    ) not in processed
                ]

                if not unprocessed:

                    print(
                        "⏭️ ALBUM "
                        "ALREADY PROCESSED"
                    )

                    continue

                await process_album(
                    group,
                    processed,
                    work_dir
                )

            # ------------------------------------------------
            # 每个组结束立即保存
            # ------------------------------------------------

            save_processed(
                processed
            )

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

    await client.disconnect()

    try:

        await bot.shutdown()

    except Exception:

        pass


# ============================================================
# 程序入口
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
