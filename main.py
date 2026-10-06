import os
import re
import json
import asyncio
import tempfile
import shutil

from datetime import datetime, timezone, timedelta

from PIL import (
    Image,
    ImageOps,
    ImageEnhance,
    ImageDraw,
    ImageFont
)

import pytesseract
import cv2

from telethon import TelegramClient
from telethon.sessions import StringSession

from telegram import Bot


# ============================================================
# 基础配置
# ============================================================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]

# 普通账号 Telethon Session
SESSION = os.environ["TELEGRAM_SESSION"]

# Bot Token
BOT_TOKEN = os.environ["BOT_TOKEN"]


# ============================================================
# 源频道
# ============================================================

SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ.get(
        "SOURCE_CHANNELS",
        ""
    ).split(",")
    if x.strip()
]


# ============================================================
# 目标频道
# ============================================================

TARGET_CHANNEL = os.environ.get(
    "TARGET_CHANNEL",
    ""
).strip()


if TARGET_CHANNEL:
    if not TARGET_CHANNEL.startswith("@") and not TARGET_CHANNEL.lstrip("-").isdigit():
        TARGET_CHANNEL = "@" + TARGET_CHANNEL


# ============================================================
# 北京时间
# ============================================================

BEIJING_TZ = timezone(
    timedelta(hours=8)
)


# ============================================================
# 扫描数量
# ============================================================

SCAN_LIMIT = int(
    os.environ.get(
        "SCAN_LIMIT",
        "500"
    )
)


# ============================================================
# processed.json
# ============================================================

PROCESSED_FILE = "processed.json"


# ============================================================
# 水印
# ============================================================

WATERMARK_TEXT = "85H官方频道"


# ============================================================
# 基础检查
# ============================================================

if not SOURCE_CHANNELS:
    raise RuntimeError(
        "SOURCE_CHANNELS is empty."
    )


if not TARGET_CHANNEL:
    raise RuntimeError(
        "TARGET_CHANNEL is empty."
    )


if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is empty."
    )


# ============================================================
# processed.json
# ============================================================

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
            repr(e)
        )

        return {
            "messages": []
        }


def save_processed(data):

    messages = list(
        dict.fromkeys(
            data.get(
                "messages",
                []
            )
        )
    )

    # 只保留最近 5000 条
    messages = messages[-5000:]

    data["messages"] = messages

    temp_file = PROCESSED_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temp_file,
        PROCESSED_FILE
    )


processed = load_processed()


# ============================================================
# 联系方式检测
# ============================================================

CONTACT_PATTERNS = [

    # Telegram 链接
    r"(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me)/[A-Za-z0-9_+/=?-]+",

    # Telegram 用户名
    r"@[A-Za-z][A-Za-z0-9_]{4,31}",

    # WhatsApp
    r"(?:https?://)?(?:www\.)?wa\.me/\d+",
    r"\bwhatsapp\b",

    # 微信
    r"微信",
    r"\bwechat\b",

    # 联系方式关键词
    r"客服",
    r"联系我",
    r"加我",
    r"私聊",
    r"添加好友",
    r"扫码联系",
    r"二维码",
    r"联系方式",

    # 英文联系方式
    r"\bcontact\s*(?:me|us)?\b",
    r"\bcustomer\s*service\b",
    r"\btelegram\b",

    # 中国大陆手机号
    r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)",

    # 越南手机号
    r"(?<!\d)(?:\+?84[- ]?)?(?:0?3|0?5|0?7|0?8|0?9)\d{8}(?!\d)",

    # 香港电话号码
    r"(?<!\d)(?:\+?852[- ]?)?[2-9]\d{3}[- ]?\d{4}(?!\d)",
]


def contains_contact(
    text,
    print_result=True
):

    if not text:
        return False

    text = str(text)

    for pattern in CONTACT_PATTERNS:

        try:

            if re.search(
                pattern,
                text,
                re.IGNORECASE
            ):

                if print_result:

                    print(
                        "CONTACT DETECTED:",
                        pattern
                    )

                return True

        except Exception as e:

            print(
                "REGEX ERROR:",
                repr(e)
            )

    return False


# ============================================================
# 二维码检测
# ============================================================

def image_contains_qrcode(
    image_path
):

    try:

        print(
            "========================================"
        )

        print(
            "QR CODE CHECK"
        )

        print(
            "========================================"
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                "QR CHECK: image load failed"
            )

            return False

        detector = cv2.QRCodeDetector()

        # 原图
        data, points, _ = detector.detectAndDecode(
            image
        )

        if data:

            print(
                "QR CODE DETECTED:",
                data[:200]
            )

            return True

        # 灰度图
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        data, points, _ = detector.detectAndDecode(
            gray
        )

        if data:

            print(
                "QR CODE DETECTED:",
                data[:200]
            )

            return True

        # 尝试多二维码
        try:

            result = detector.detectAndDecodeMulti(
                image
            )

            if len(result) >= 2:

                retval = result[0]
                decoded_info = result[1]

                if retval:

                    for item in decoded_info:

                        if item:

                            print(
                                "QR CODE DETECTED:",
                                item[:200]
                            )

                            return True

        except Exception as e:

            print(
                "MULTI QR CHECK ERROR:",
                repr(e)
            )

        print(
            "QR CODE: NOT DETECTED"
        )

        return False

    except Exception as e:

        print(
            "QR CHECK ERROR:",
            repr(e)
        )

        return False


# ============================================================
# OCR
# ============================================================

def run_ocr(
    image,
    config=""
):

    try:

        return pytesseract.image_to_string(
            image,
            lang="eng+chi_sim+vie+por",
            config=config
        )

    except Exception as e:

        print(
            "OCR ERROR:",
            repr(e)
        )

        return ""


# ============================================================
# 图片联系方式检测
# ============================================================

def image_contains_contact(
    image_path
):

    try:

        print(
            "========================================"
        )

        print(
            "IMAGE CONTACT CHECK"
        )

        print(
            "========================================"
        )

        image = Image.open(
            image_path
        )

        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        print(
            "ORIGINAL IMAGE SIZE:",
            width,
            "x",
            height
        )

        # ----------------------------------------------------
        # QR
        # ----------------------------------------------------

        if image_contains_qrcode(
            image_path
        ):

            print(
                "❌ CONTACT CHECK: QR CODE FOUND"
            )

            return True

        # ----------------------------------------------------
        # 放大
        # ----------------------------------------------------

        if width < 1800:

            ratio = 1800 / width

            image = image.resize(
                (
                    int(width * ratio),
                    int(height * ratio)
                ),
                Image.Resampling.LANCZOS
            )

        # ----------------------------------------------------
        # OCR 1：原图
        # ----------------------------------------------------

        ocr_text_1 = run_ocr(
            image
        )

        print(
            "OCR ORIGINAL:"
        )

        print(
            repr(
                ocr_text_1[:1500]
            )
        )

        if contains_contact(
            ocr_text_1
        ):

            print(
                "❌ IMAGE CONTACT FOUND"
            )

            return True

        # ----------------------------------------------------
        # OCR 2：灰度
        # ----------------------------------------------------

        gray = ImageOps.grayscale(
            image
        )

        ocr_text_2 = run_ocr(
            gray
        )

        print(
            "OCR GRAYSCALE:"
        )

        print(
            repr(
                ocr_text_2[:1500]
            )
        )

        if contains_contact(
            ocr_text_2
        ):

            print(
                "❌ IMAGE CONTACT FOUND"
            )

            return True

        # ----------------------------------------------------
        # OCR 3：增强
        # ----------------------------------------------------

        enhanced = ImageEnhance.Contrast(
            gray
        ).enhance(2.0)

        enhanced = ImageEnhance.Sharpness(
            enhanced
        ).enhance(2.0)

        ocr_text_3 = run_ocr(
            enhanced
        )

        print(
            "OCR ENHANCED:"
        )

        print(
            repr(
                ocr_text_3[:1500]
            )
        )

        if contains_contact(
            ocr_text_3
        ):

            print(
                "❌ IMAGE CONTACT FOUND"
            )

            return True

        # ----------------------------------------------------
        # OCR 4：二值化
        # ----------------------------------------------------

        threshold = enhanced.point(
            lambda p:
            255
            if p > 150
            else 0
        )

        ocr_text_4 = run_ocr(
            threshold
        )

        print(
            "OCR THRESHOLD:"
        )

        print(
            repr(
                ocr_text_4[:1500]
            )
        )

        if contains_contact(
            ocr_text_4
        ):

            print(
                "❌ IMAGE CONTACT FOUND"
            )

            return True

        print(
            "✅ IMAGE CONTACT NOT FOUND"
        )

        return False

    except Exception as e:

        print(
            "IMAGE OCR ERROR:",
            repr(e)
        )

        # OCR 出错时默认不发送
        return True


# ============================================================
# 水印字体
# ============================================================

def get_watermark_font(
    font_size
):

    font_paths = [

        # Ubuntu / GitHub Actions
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",

        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",

        # 备用
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",

        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for font_path in font_paths:

        if os.path.exists(
            font_path
        ):

            try:

                return ImageFont.truetype(
                    font_path,
                    font_size
                )

            except Exception:
                pass

    return ImageFont.load_default()


# ============================================================
# 添加水印
# ============================================================

def add_watermark(
    image_path
):

    try:

        print(
            "========================================"
        )

        print(
            "ADDING WATERMARK"
        )

        print(
            "TEXT:",
            WATERMARK_TEXT
        )

        print(
            "========================================"
        )

        image = Image.open(
            image_path
        )

        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert(
            "RGBA"
        )

        width, height = image.size

        # ----------------------------------------------------
        # 根据图片宽度自动调整字体
        # ----------------------------------------------------

        font_size = max(
            24,
            int(width * 0.035)
        )

        font = get_watermark_font(
            font_size
        )

        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0)
        )

        draw = ImageDraw.Draw(
            overlay
        )

        bbox = draw.textbbox(
            (0, 0),
            WATERMARK_TEXT,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        margin = max(
            20,
            int(width * 0.02)
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

        # ----------------------------------------------------
        # 黑色描边
        # ----------------------------------------------------

        stroke_width = max(
            2,
            int(font_size * 0.08)
        )

        draw.text(
            (x, y),
            WATERMARK_TEXT,
            font=font,
            fill=(0, 0, 0, 150),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 160)
        )

        # ----------------------------------------------------
        # 白色文字
        # ----------------------------------------------------

        draw.text(
            (x, y),
            WATERMARK_TEXT,
            font=font,
            fill=(255, 255, 255, 190)
        )

        result = Image.alpha_composite(
            image,
            overlay
        )

        result = result.convert(
            "RGB"
        )

        output_path = (
            image_path.rsplit(
                ".",
                1
            )[0]
            + "_watermark.jpg"
        )

        # JPEG
        result.save(
            output_path,
            "JPEG",
            quality=92,
            optimize=True
        )

        print(
            "✅ WATERMARK ADDED:",
            output_path
        )

        return output_path

    except Exception as e:

        print(
            "❌ WATERMARK ERROR:",
            repr(e)
        )

        return None


# ============================================================
# 北京时间：是否今天
# ============================================================

def is_today(
    message
):

    if not message.date:
        return False

    message_time = message.date.astimezone(
        BEIJING_TZ
    )

    today = datetime.now(
        BEIJING_TZ
    ).date()

    return (
        message_time.date()
        == today
    )


# ============================================================
# 标记已处理
# ============================================================

def mark_processed(
    key
):

    processed.setdefault(
        "messages",
        []
    ).append(
        key
    )

    save_processed(
        processed
    )


# ============================================================
# Bot 检查
# ============================================================

async def check_bot(
    bot
):

    print(
        "========================================"
    )

    print(
        "CHECKING BOT"
    )

    print(
        "========================================"
    )

    try:

        bot_info = await bot.get_me()

        print(
            "BOT USERNAME:",
            getattr(
                bot_info,
                "username",
                None
            )
        )

        print(
            "BOT ID:",
            getattr(
                bot_info,
                "id",
                None
            )
        )

    except Exception as e:

        print(
            "❌ BOT TOKEN ERROR:",
            repr(e)
        )

        raise


    # --------------------------------------------------------
    # 检查目标频道
    # --------------------------------------------------------

    try:

        chat = await bot.get_chat(
            TARGET_CHANNEL
        )

        print(
            "TARGET CHAT:",
            getattr(
                chat,
                "title",
                None
            )
        )

        print(
            "TARGET CHAT ID:",
            getattr(
                chat,
                "id",
                None
            )
        )

    except Exception as e:

        print(
            "❌ BOT TARGET CHANNEL ERROR:",
            repr(e)
        )

        print(
            "请检查："
        )

        print(
            "1. Bot 是否加入目标频道"
        )

        print(
            "2. Bot 是否是管理员"
        )

        print(
            "3. Bot 是否拥有发布消息权限"
        )

        raise


# ============================================================
# Bot 发送文字
# ============================================================

async def bot_send_text(
    bot,
    text
):

    if not text.strip():
        return

    print(
        "BOT SENDING TEXT..."
    )

    # Telegram 普通文本限制
    # 这里按 4000 字切割，留出安全空间
    max_length = 4000

    chunks = [
        text[i:i + max_length]
        for i in range(
            0,
            len(text),
            max_length
        )
    ]

    for chunk in chunks:

        await bot.send_message(
            chat_id=TARGET_CHANNEL,
            text=chunk,
            parse_mode=None
        )

        # 避免过快
        await asyncio.sleep(
            0.5
        )

    print(
        "✅ BOT TEXT SENT"
    )


# ============================================================
# Bot 发送图片
# ============================================================

async def bot_send_photo(
    bot,
    image_path,
    caption=""
):

    print(
        "BOT SENDING PHOTO..."
    )

    caption = caption or ""

    # Telegram Photo Caption 最好控制在 1024 字以内
    if len(caption) <= 1024:

        with open(
            image_path,
            "rb"
        ) as photo:

            await bot.send_photo(
                chat_id=TARGET_CHANNEL,
                photo=photo,
                caption=(
                    caption
                    if caption.strip()
                    else None
                ),
                parse_mode=None
            )

    else:

        # Caption 太长：
        # 图片先发送
        with open(
            image_path,
            "rb"
        ) as photo:

            await bot.send_photo(
                chat_id=TARGET_CHANNEL,
                photo=photo
            )

        await asyncio.sleep(
            0.5
        )

        # 然后发送完整文字
        await bot_send_text(
            bot,
            caption
        )

    print(
        "✅ BOT PHOTO SENT"
    )


# ============================================================
# 处理图片消息
# ============================================================

async def process_photo_message(
    client,
    bot,
    message,
    source_name,
    key,
    text
):

    image_path = None
    watermarked_path = None
    temp_dir = None

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
                "❌ IMAGE DOWNLOAD FAILED"
            )

            return

        print(
            "IMAGE DOWNLOADED:",
            image_path
        )

        # ----------------------------------------------------
        # 图片检测
        # ----------------------------------------------------

        has_contact = image_contains_contact(
            image_path
        )

        if has_contact:

            print(
                "❌ SKIP IMAGE:"
                " CONTACT INFORMATION / QR CODE DETECTED"
            )

            mark_processed(
                key
            )

            return

        # ----------------------------------------------------
        # Caption 检测
        # ----------------------------------------------------

        if contains_contact(
            text
        ):

            print(
                "❌ SKIP IMAGE:"
                " CONTACT IN CAPTION"
            )

            mark_processed(
                key
            )

            return

        # ----------------------------------------------------
        # 添加水印
        # ----------------------------------------------------

        watermarked_path = add_watermark(
            image_path
        )

        if not watermarked_path:

            print(
                "❌ WATERMARK FAILED"
            )

            return

        # ----------------------------------------------------
        # Bot 发送
        # ----------------------------------------------------

        await bot_send_photo(
            bot,
            watermarked_path,
            text
        )

        print(
            "✅ IMAGE FORWARDED BY BOT:",
            key
        )

        mark_processed(
            key
        )

    except Exception as e:

        print(
            "❌ IMAGE PROCESS ERROR:",
            repr(e)
        )

    finally:

        # 删除下载图片
        if image_path:

            try:
                os.remove(
                    image_path
                )
            except Exception:
                pass

        # 删除水印图片
        if watermarked_path:

            try:
                os.remove(
                    watermarked_path
                )
            except Exception:
                pass

        if temp_dir:

            try:

                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True
                )

            except Exception:
                pass


# ============================================================
# 处理文字消息
# ============================================================

async def process_text_message(
    bot,
    message,
    source_name,
    key,
    text
):

    if not text.strip():

        print(
            "SKIP: EMPTY OR UNSUPPORTED MESSAGE"
        )

        return

    # --------------------------------------------------------
    # 联系方式检测
    # --------------------------------------------------------

    if contains_contact(
        text
    ):

        print(
            "❌ SKIP TEXT:"
            " CONTACT INFORMATION DETECTED"
        )

        mark_processed(
            key
        )

        return

    # --------------------------------------------------------
    # Bot 发送
    # --------------------------------------------------------

    try:

        await bot_send_text(
            bot,
            text
        )

        print(
            "✅ TEXT FORWARDED BY BOT:",
            key
        )

        mark_processed(
            key
        )

    except Exception as e:

        print(
            "❌ BOT TEXT SEND ERROR:",
            repr(e)
        )


# ============================================================
# 处理单条消息
# ============================================================

async def process_message(
    client,
    bot,
    message,
    source_name
):

    key = (
        f"{source_name}:"
        f"{message.id}"
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "PROCESS:",
        key
    )

    print(
        "MESSAGE DATE:",
        message.date
    )

    print(
        "BEIJING DATE:",
        (
            message.date.astimezone(
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

    text = message.text or ""

    print(
        "TEXT:",
        repr(text[:500])
    )

    # --------------------------------------------------------
    # 去重
    # --------------------------------------------------------

    if key in processed.get(
        "messages",
        []
    ):

        print(
            "SKIP: ALREADY PROCESSED"
        )

        return

    # --------------------------------------------------------
    # 只处理北京时间今天
    # --------------------------------------------------------

    if not is_today(
        message
    ):

        print(
            "SKIP: NOT TODAY"
        )

        return

    # --------------------------------------------------------
    # 图片
    # --------------------------------------------------------

    if message.photo:

        await process_photo_message(
            client,
            bot,
            message,
            source_name,
            key,
            text
        )

        return

    # --------------------------------------------------------
    # 文字
    # --------------------------------------------------------

    if text.strip():

        await process_text_message(
            bot,
            message,
            source_name,
            key,
            text
        )

        return

    print(
        "SKIP: EMPTY OR UNSUPPORTED MESSAGE"
    )


# ============================================================
# 主程序
# ============================================================

async def main():

    print(
        "========================================"
    )

    print(
        "Telegram Channel Forwarder"
    )

    print(
        "B VERSION"
    )

    print(
        "普通账号读取 + Bot发布"
    )

    print(
        "========================================"
    )

    print(
        "BEIJING TIME:",
        datetime.now(
            BEIJING_TZ
        )
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
            processed.get(
                "messages",
                []
            )
        )
    )


    # ========================================================
    # 创建普通账号客户端
    # ========================================================

    client = TelegramClient(
        StringSession(
            SESSION
        ),
        API_ID,
        API_HASH
    )


    # ========================================================
    # 创建 Bot
    # ========================================================

    bot = Bot(
        token=BOT_TOKEN
    )


    # ========================================================
    # 连接普通账号
    # ========================================================

    print(
        "CONNECTING TELEGRAM ACCOUNT..."
    )

    await client.start()

    print(
        "✅ TELEGRAM ACCOUNT CONNECTED"
    )


    try:

        # ====================================================
        # 检查 Bot
        # ====================================================

        await check_bot(
            bot
        )


        # ====================================================
        # 检查普通账号目标频道
        # ====================================================

        try:

            target_entity = await client.get_entity(
                TARGET_CHANNEL
            )

            print(
                "ACCOUNT TARGET ENTITY:",
                target_entity
            )

            print(
                "ACCOUNT TARGET TITLE:",
                getattr(
                    target_entity,
                    "title",
                    None
                )
            )

            print(
                "ACCOUNT TARGET ID:",
                getattr(
                    target_entity,
                    "id",
                    None
                )
            )

        except Exception as e:

            print(
                "ACCOUNT TARGET CHECK WARNING:",
                repr(e)
            )


        # ====================================================
        # 遍历源频道
        # ====================================================

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


            # ------------------------------------------------
            # 获取源频道
            # ------------------------------------------------

            try:

                entity = await client.get_entity(
                    source
                )

                print(
                    "SOURCE ENTITY:",
                    entity
                )

                print(
                    "SOURCE TITLE:",
                    getattr(
                        entity,
                        "title",
                        None
                    )
                )

                print(
                    "SOURCE ID:",
                    getattr(
                        entity,
                        "id",
                        None
                    )
                )

            except Exception as e:

                print(
                    "❌ SOURCE CHANNEL ERROR:",
                    source,
                    repr(e)
                )

                continue


            count = 0
            today_count = 0
            processed_count = 0


            # ------------------------------------------------
            # 扫描消息
            # ------------------------------------------------

            async for message in client.iter_messages(
                entity,
                limit=SCAN_LIMIT
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
                    "PHOTO:",
                    bool(message.photo)
                )

                print(
                    "TEXT:",
                    repr(
                        (
                            message.text
                            or ""
                        )[:100]
                    )
                )


                # --------------------------------------------
                # 不是今天
                # --------------------------------------------

                if not is_today(
                    message
                ):

                    print(
                        "NOT TODAY - SKIP"
                    )

                    continue


                today_count += 1


                # --------------------------------------------
                # 处理
                # --------------------------------------------

                before = len(
                    processed.get(
                        "messages",
                        []
                    )
                )


                await process_message(
                    client,
                    bot,
                    message,
                    source
                )


                after = len(
                    processed.get(
                        "messages",
                        []
                    )
                )


                if after > before:

                    processed_count += 1


                # --------------------------------------------
                # 稍微等待
                # --------------------------------------------

                await asyncio.sleep(
                    0.3
                )


            # ------------------------------------------------
            # 当前源频道结束
            # ------------------------------------------------

            print(
                "\n========================================"
            )

            print(
                "SOURCE SCAN COMPLETE"
            )

            print(
                "SOURCE:",
                source
            )

            print(
                "TOTAL READ:",
                count
            )

            print(
                "TODAY:",
                today_count
            )

            print(
                "PROCESSED:",
                processed_count
            )

            print(
                "========================================"
            )


    finally:

        # ====================================================
        # 关闭
        # ====================================================

        await client.disconnect()

        await bot.shutdown()

        print(
            "\n✅ TELEGRAM ACCOUNT DISCONNECTED"
        )

        print(
            "✅ BOT DISCONNECTED"
        )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
