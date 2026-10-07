import os
import re
import json
import asyncio
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
import pytesseract
import cv2

from telethon import TelegramClient
from telethon.sessions import StringSession

from telegram import Bot


# =========================================================
# 基础配置
# =========================================================

BEIJING_TZ = timezone(timedelta(hours=8))

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]

# 普通 Telegram 账号 StringSession
TELEGRAM_SESSION = os.environ["TELEGRAM_SESSION"]

# Telegram Bot Token
BOT_TOKEN = os.environ["BOT_TOKEN"]

# 源频道
# 例如：
# SOURCE_CHANNELS=channel1,channel2,channel3
SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ.get(
        "SOURCE_CHANNELS",
        ""
    ).split(",")
    if x.strip()
]

# 目标频道
# 可以填写：
# abc123
# @abc123
# -1001234567890
TARGET_CHANNEL = os.environ[
    "TARGET_CHANNEL"
].strip()

# 每个源频道最多扫描多少条
SCAN_LIMIT = int(
    os.environ.get(
        "SCAN_LIMIT",
        "500"
    )
)

# processed 文件
PROCESSED_FILE = "processed.json"

# 图片水印
WATERMARK_TEXT = "85H官方频道"


# =========================================================
# 联系方式识别规则
# =========================================================
#
# 重点：
#
# 不再识别：
#
# ❌ 任意 8 位数字
# ❌ Telegram 单词
# ❌ 客服
# ❌ 联系我
# ❌ 私聊
# ❌ 添加好友
# ❌ 二维码文字
#
# 因为 OCR 很容易把普通图片文字识别成这些内容。
#


CONTACT_PATTERNS = [

    # =====================================================
    # Telegram 链接
    #
    # 示例：
    #
    # https://t.me/username
    # http://t.me/username
    # t.me/username
    # telegram.me/username
    #
    # 这是强联系方式
    # 一次 OCR 命中即可拦截
    # =====================================================

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

    # =====================================================
    # Telegram 用户名
    #
    # 示例：
    #
    # @abcde
    # @username123
    #
    # 必须英文开头
    # 至少 5 个字符
    # =====================================================

    (
        "TELEGRAM_USERNAME",
        re.compile(
            r"@[A-Za-z][A-Za-z0-9_]{4,31}"
        )
    ),

    # =====================================================
    # WhatsApp
    #
    # 示例：
    #
    # https://wa.me/84912345678
    # wa.me/84912345678
    #
    # 强联系方式
    # 一次 OCR 命中即可拦截
    # =====================================================

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

    # =====================================================
    # 微信
    #
    # 这里只识别英文 wechat
    #
    # 不直接识别：
    #
    # 微信
    # 客服
    # 联系
    #
    # 防止 OCR 中文误识别
    # =====================================================

    (
        "WECHAT",
        re.compile(
            r"\bwechat\b",
            re.IGNORECASE
        )
    ),

    # =====================================================
    # 中国大陆手机号
    #
    # 示例：
    #
    # 13812345678
    # +8613812345678
    # 8613812345678
    #
    # 注意：
    # 图片 OCR 手机号需要二次确认
    # =====================================================

    (
        "CN_PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?86[- ]?)?"
            r"1[3-9]\d{9}"
            r"(?!\d)"
        )
    ),

    # =====================================================
    # 越南手机号
    #
    # 示例：
    #
    # 0912345678
    # 0912 345 678
    # +84912345678
    # 84912345678
    #
    # 图片 OCR 手机号需要二次确认
    # =====================================================

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


# =========================================================
# 联系方式类型
# =========================================================

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


# =========================================================
# 北京时间
# =========================================================

def beijing_now():

    return datetime.now(
        BEIJING_TZ
    )


# =========================================================
# 判断消息是否为北京时间今天
# =========================================================

def is_today(message):

    if not message.date:
        return False

    msg_time = message.date

    # Telethon 通常返回 UTC datetime
    if msg_time.tzinfo is None:

        msg_time = msg_time.replace(
            tzinfo=timezone.utc
        )

    msg_time = msg_time.astimezone(
        BEIJING_TZ
    )

    return (
        msg_time.date()
        ==
        beijing_now().date()
    )


# =========================================================
# processed.json
# =========================================================

def load_processed():

    if not os.path.exists(
        PROCESSED_FILE
    ):

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

        return set(
            str(x)
            for x in messages
        )

    except Exception as e:

        print(
            "⚠️ processed.json 读取失败:",
            e
        )

        return set()


# =========================================================
# 保存 processed
# =========================================================

def save_processed(
    processed
):

    data = {
        "messages": list(
            processed
        )[-5000:]
    }

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


# =========================================================
# 文本联系方式检测
# =========================================================

def detect_contact(text):

    if not text:

        return None

    text = str(text)

    for name, pattern in CONTACT_PATTERNS:

        match = pattern.search(
            text
        )

        if match:

            return {
                "type": name,
                "pattern": pattern.pattern,
                "match": match.group(0)
            }

    return None


# =========================================================
# QR 二维码检测
# =========================================================

def detect_qr(
    image_path
):

    try:

        image = cv2.imread(
            image_path
        )

        if image is None:

            return False, []

        detector = cv2.QRCodeDetector()

        detected_data = []

        # =================================================
        # 普通 QR 检测
        # =================================================

        try:

            data, points, _ = (
                detector.detectAndDecode(
                    image
                )
            )

            if data:

                detected_data.append(
                    data
                )

        except Exception:
            pass

        # =================================================
        # 灰度 QR 检测
        # =================================================

        try:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            data, points, _ = (
                detector.detectAndDecode(
                    gray
                )
            )

            if data:

                detected_data.append(
                    data
                )

        except Exception:
            pass

        # =================================================
        # 多二维码检测
        # =================================================

        try:

            result = (
                detector.detectAndDecodeMulti(
                    image
                )
            )

            if result and len(result) >= 2:

                ok = result[0]
                decoded_info = result[1]

                if (
                    ok
                    and decoded_info
                ):

                    for item in decoded_info:

                        if item:

                            detected_data.append(
                                item
                            )

        except Exception:
            pass

        # =================================================
        # 去重
        # =================================================

        detected_data = list(
            dict.fromkeys(
                x
                for x in detected_data
                if x
            )
        )

        if detected_data:

            return True, detected_data

        return False, []

    except Exception as e:

        print(
            "⚠️ QR 检测失败:",
            e
        )

        return False, []


# =========================================================
# OCR 多版本识别
# =========================================================

def run_ocr_variants(
    image_path
):

    results = []

    try:

        original = Image.open(
            image_path
        ).convert(
            "RGB"
        )

    except Exception as e:

        print(
            "❌ 图片打开失败:",
            e
        )

        return results

    # =====================================================
    # OCR 1：原图
    # =====================================================

    try:

        text = pytesseract.image_to_string(
            original,
            lang="eng+chi_sim+vie"
        )

        results.append(
            (
                "ORIGINAL",
                text
            )
        )

    except Exception as e:

        print(
            "OCR ORIGINAL ERROR:",
            e
        )

    # =====================================================
    # OCR 2：灰度
    # =====================================================

    try:

        gray = ImageOps.grayscale(
            original
        )

        text = pytesseract.image_to_string(
            gray,
            lang="eng+chi_sim+vie"
        )

        results.append(
            (
                "GRAYSCALE",
                text
            )
        )

    except Exception as e:

        print(
            "OCR GRAYSCALE ERROR:",
            e
        )

    # =====================================================
    # OCR 3：增强
    # =====================================================

    try:

        enhanced = ImageEnhance.Contrast(
            original
        ).enhance(
            2.0
        )

        enhanced = ImageEnhance.Sharpness(
            enhanced
        ).enhance(
            2.0
        )

        text = pytesseract.image_to_string(
            enhanced,
            lang="eng+chi_sim+vie"
        )

        results.append(
            (
                "ENHANCED",
                text
            )
        )

    except Exception as e:

        print(
            "OCR ENHANCED ERROR:",
            e
        )

    # =====================================================
    # OCR 4：阈值
    # =====================================================

    try:

        gray = ImageOps.grayscale(
            original
        )

        threshold = gray.point(
            lambda p:
            255 if p > 160 else 0
        )

        text = pytesseract.image_to_string(
            threshold,
            lang="eng+chi_sim+vie"
        )

        results.append(
            (
                "THRESHOLD",
                text
            )
        )

    except Exception as e:

        print(
            "OCR THRESHOLD ERROR:",
            e
        )

    return results


# =========================================================
# 图片联系方式检测
# =========================================================

def detect_image_contact(
    image_path
):

    ocr_results = run_ocr_variants(
        image_path
    )

    if not ocr_results:

        print(
            "⚠️ OCR 没有返回结果"
        )

        return None

    # =====================================================
    # 所有检测结果
    # =====================================================

    detections = {}

    # =====================================================
    # 遍历 OCR
    # =====================================================

    for pass_name, text in ocr_results:

        print()
        print(
            f"OCR {pass_name}:"
        )

        # 防止日志过长
        print(
            repr(
                text[:2000]
            )
        )

        result = detect_contact(
            text
        )

        if result:

            contact_type = result[
                "type"
            ]

            if (
                contact_type
                not in detections
            ):

                detections[
                    contact_type
                ] = []

            detections[
                contact_type
            ].append(
                {
                    "pass": pass_name,
                    "match": result[
                        "match"
                    ],
                    "pattern": result[
                        "pattern"
                    ]
                }
            )

            print(
                "CONTACT DETECTED:",
                contact_type
            )

            print(
                "MATCH:",
                repr(
                    result["match"]
                )
            )

    # =====================================================
    # 完全没有联系方式
    # =====================================================

    if not detections:

        print()
        print(
            "✅ IMAGE OCR: NO CONTACT FOUND"
        )

        return None

    # =====================================================
    # 强联系方式
    #
    # Telegram URL
    # Telegram username
    # WhatsApp
    # WeChat
    #
    # 一次命中即可
    # =====================================================

    for contact_type in (
        STRONG_CONTACT_TYPES
    ):

        hits = detections.get(
            contact_type,
            []
        )

        if not hits:

            continue

        hit = hits[0]

        print()
        print(
            "========================================"
        )

        print(
            "❌ STRONG CONTACT FOUND"
        )

        print(
            "PATTERN:",
            contact_type
        )

        print(
            "MATCH:",
            repr(
                hit["match"]
            )
        )

        print(
            "OCR PASS:",
            hit["pass"]
        )

        print(
            "ACTION: SKIP IMAGE"
        )

        print(
            "========================================"
        )

        return hit

    # =====================================================
    # 手机号检测
    #
    # 注意：
    #
    # 如果只命中一次：
    #
    # ❌ 不跳过
    #
    # 如果至少两个 OCR 版本命中：
    #
    # ✅ 确认联系方式
    #
    # 这样可以减少 OCR 乱码误判。
    # =====================================================

    for contact_type in (
        PHONE_CONTACT_TYPES
    ):

        hits = detections.get(
            contact_type,
            []
        )

        # -------------------------------------------------
        # 这里非常重要
        #
        # 如果根本没有命中，
        # 直接 continue
        #
        # 不再出现：
        #
        # CN_PHONE
        # OCR PASS COUNT: 0
        #
        # 这种错误日志。
        # -------------------------------------------------

        if not hits:

            continue

        unique_passes = set(
            item["pass"]
            for item in hits
        )

        # =================================================
        # 两次或以上 OCR 命中
        # =================================================

        if len(unique_passes) >= 2:

            hit = hits[0]

            print()
            print(
                "========================================"
            )

            print(
                "❌ PHONE CONTACT CONFIRMED"
            )

            print(
                "PATTERN:",
                contact_type
            )

            print(
                "MATCH:",
                repr(
                    hit["match"]
                )
            )

            print(
                "OCR PASS COUNT:",
                len(unique_passes)
            )

            print(
                "OCR PASSES:",
                ", ".join(
                    sorted(
                        unique_passes
                    )
                )
            )

            print(
                "ACTION: SKIP IMAGE"
            )

            print(
                "========================================"
            )

            return hit

        # =================================================
        # 只有一次 OCR 命中
        #
        # 不拦截
        # =================================================

        print()
        print(
            "⚠️ PHONE OCR HIT BUT NOT CONFIRMED"
        )

        print(
            "PATTERN:",
            contact_type
        )

        print(
            "MATCH:",
            repr(
                hits[0]["match"]
            )
        )

        print(
            "OCR PASS COUNT:",
            len(unique_passes)
        )

        print(
            "OCR PASSES:",
            ", ".join(
                sorted(
                    unique_passes
                )
            )
        )

        print(
            "ACTION: IGNORE FALSE POSITIVE"
        )

    # =====================================================
    # 没有确认联系方式
    # =====================================================

    print()
    print(
        "✅ IMAGE CONTACT NOT CONFIRMED"
    )

    return None


# =========================================================
# 添加水印
# =========================================================

def add_watermark(
    input_path,
    output_path
):

    image = Image.open(
        input_path
    ).convert(
        "RGBA"
    )

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(
        overlay
    )

    # =====================================================
    # Noto CJK 字体
    # =====================================================

    font_path = (
        "/usr/share/fonts/opentype/"
        "noto/NotoSansCJK-Regular.ttc"
    )

    try:

        font_size = max(
            24,
            image.width // 30
        )

        font = ImageFont.truetype(
            font_path,
            font_size
        )

    except Exception:

        font = ImageFont.load_default()

    # =====================================================
    # 计算文字尺寸
    # =====================================================

    bbox = draw.textbbox(
        (0, 0),
        WATERMARK_TEXT,
        font=font
    )

    text_width = (
        bbox[2] - bbox[0]
    )

    text_height = (
        bbox[3] - bbox[1]
    )

    margin = max(
        20,
        image.width // 50
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

    padding = 10

    # =====================================================
    # 半透明背景
    # =====================================================

    draw.rounded_rectangle(
        (
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        ),
        radius=8,
        fill=(
            0,
            0,
            0,
            120
        )
    )

    # =====================================================
    # 水印文字
    # =====================================================

    draw.text(
        (x, y),
        WATERMARK_TEXT,
        font=font,
        fill=(
            255,
            255,
            255,
            220
        ),
        stroke_width=2,
        stroke_fill=(
            0,
            0,
            0,
            180
        )
    )

    # =====================================================
    # 合并
    # =====================================================

    result = Image.alpha_composite(
        image,
        overlay
    )

    result.convert(
        "RGB"
    ).save(
        output_path,
        quality=95
    )


# =========================================================
# Bot 发送文本
# =========================================================

async def send_text(
    bot,
    target,
    text
):

    if not text:

        return

    # Telegram 普通文本安全限制
    max_length = 4000

    # =====================================================
    # 不超过限制
    # =====================================================

    if len(text) <= max_length:

        await bot.send_message(
            chat_id=target,
            text=text
        )

        return

    # =====================================================
    # 超长拆分
    # =====================================================

    for i in range(
        0,
        len(text),
        max_length
    ):

        chunk = text[
            i:i + max_length
        ]

        await bot.send_message(
            chat_id=target,
            text=chunk
        )

        await asyncio.sleep(
            0.5
        )


# =========================================================
# Bot 发送图片
# =========================================================

async def send_photo(
    bot,
    target,
    image_path,
    caption=None
):

    # Telegram 图片 Caption 最大约 1024 字符
    if (
        caption
        and len(caption) > 1024
    ):

        # 先发图片
        with open(
            image_path,
            "rb"
        ) as photo:

            await bot.send_photo(
                chat_id=target,
                photo=photo
            )

        # 再发完整文字
        await send_text(
            bot,
            target,
            caption
        )

    else:

        with open(
            image_path,
            "rb"
        ) as photo:

            await bot.send_photo(
                chat_id=target,
                photo=photo,
                caption=caption or None
            )


# =========================================================
# 主程序
# =========================================================

async def main():

    print()
    print(
        "=========================================="
    )

    print(
        "Telegram Channel Forwarder"
    )

    print(
        "=========================================="
    )

    print(
        "BEIJING TIME:",
        beijing_now()
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
        "=========================================="
    )

    # =====================================================
    # 参数检查
    # =====================================================

    if not SOURCE_CHANNELS:

        print(
            "❌ SOURCE_CHANNELS 未配置"
        )

        return

    if not TARGET_CHANNEL:

        print(
            "❌ TARGET_CHANNEL 未配置"
        )

        return

    # =====================================================
    # 目标频道格式
    # =====================================================

    target = TARGET_CHANNEL

    # 如果不是数字 ID
    # 自动添加 @
    if not target.startswith("@"):

        if not target.lstrip(
            "-"
        ).isdigit():

            target = "@" + target

    print(
        "NORMALIZED TARGET:",
        target
    )

    # =====================================================
    # processed
    # =====================================================

    processed = load_processed()

    print(
        "PROCESSED COUNT:",
        len(processed)
    )

    # =====================================================
    # 创建普通 Telegram Client
    # =====================================================

    client = TelegramClient(
        StringSession(
            TELEGRAM_SESSION
        ),
        API_ID,
        API_HASH
    )

    print()
    print(
        "CONNECTING TELEGRAM..."
    )

    await client.connect()

    # =====================================================
    # 检查普通账号
    # =====================================================

    if not await client.is_user_authorized():

        print(
            "❌ TELEGRAM SESSION 未授权"
        )

        await client.disconnect()

        return

    print(
        "✅ TELEGRAM CONNECTED"
    )

    # =====================================================
    # 创建 Bot
    # =====================================================

    bot = Bot(
        token=BOT_TOKEN
    )

    # =====================================================
    # 检查 Bot
    # =====================================================

    try:

        bot_me = await bot.get_me()

        print()
        print(
            "BOT USERNAME:",
            bot_me.username
        )

        print(
            "BOT ID:",
            bot_me.id
        )

    except Exception as e:

        print(
            "❌ BOT TOKEN ERROR:",
            e
        )

        await client.disconnect()

        await bot.shutdown()

        return

    # =====================================================
    # 检查目标频道
    # =====================================================

    try:

        target_entity = await bot.get_chat(
            target
        )

        print()
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
            target_entity.id
        )

        print(
            "TARGET TYPE:",
            getattr(
                target_entity,
                "type",
                None
            )
        )

    except Exception as e:

        print()
        print(
            "❌ TARGET CHANNEL ERROR:"
        )

        print(
            repr(e)
        )

        print()
        print(
            "请检查："
        )

        print(
            "1. Bot 是否已经加入目标频道"
        )

        print(
            "2. Bot 是否是目标频道管理员"
        )

        print(
            "3. Bot 是否拥有发布消息权限"
        )

        print(
            "4. TARGET_CHANNEL 是否正确"
        )

        await client.disconnect()

        await bot.shutdown()

        return

    # =====================================================
    # 遍历源频道
    # =====================================================

    for source_name in SOURCE_CHANNELS:

        print()
        print(
            "=========================================="
        )

        print(
            "SOURCE:",
            source_name
        )

        print(
            "=========================================="
        )

        # =================================================
        # 获取源频道实体
        # =================================================

        try:

            source_entity = (
                await client.get_entity(
                    source_name
                )
            )

            print(
                "SOURCE ENTITY:",
                source_entity
            )

        except Exception as e:

            print()
            print(
                "❌ SOURCE ENTITY ERROR:",
                source_name
            )

            print(
                repr(e)
            )

            continue

        # =================================================
        # 扫描消息
        # =================================================

        try:

            messages = client.iter_messages(
                source_entity,
                limit=SCAN_LIMIT
            )

            async for message in messages:

                # =================================================
                # 只处理北京时间今天
                # =================================================

                if not is_today(message):

                    continue

                # =================================================
                # 消息唯一 ID
                # =================================================

                message_key = (
                    f"{source_name}:{message.id}"
                )

                # =================================================
                # 已处理
                # =================================================

                if (
                    message_key
                    in processed
                ):

                    print(
                        f"MESSAGE {message.id} "
                        f"already processed."
                    )

                    continue

                # =================================================
                # 空消息
                # =================================================

                if (
                    not message.message
                    and not message.media
                ):

                    print(
                        f"MESSAGE {message.id} "
                        f"empty, skipped."
                    )

                    processed.add(
                        message_key
                    )

                    continue

                # =================================================
                # 文本消息
                # =================================================

                if (
                    message.message
                    and not message.photo
                ):

                    print()
                    print(
                        "------------------------------------------"
                    )

                    print(
                        f"MESSAGE {message.id} "
                        f"TEXT"
                    )

                    # ---------------------------------------------
                    # 检查文本联系方式
                    # ---------------------------------------------

                    contact = detect_contact(
                        message.message
                    )

                    if contact:

                        print(
                            "❌ TEXT CONTACT FOUND"
                        )

                        print(
                            "PATTERN:",
                            contact["type"]
                        )

                        print(
                            "MATCH:",
                            repr(
                                contact["match"]
                            )
                        )

                        print(
                            "ACTION: SKIP TEXT"
                        )

                        processed.add(
                            message_key
                        )

                        continue

                    # ---------------------------------------------
                    # Bot 发送
                    # ---------------------------------------------

                    try:

                        await send_text(
                            bot,
                            target,
                            message.message
                        )

                        print(
                            "✅ TEXT FORWARDED"
                        )

                        processed.add(
                            message_key
                        )

                    except Exception as e:

                        print()
                        print(
                            "❌ TEXT SEND ERROR:"
                        )

                        print(
                            repr(e)
                        )

                    continue

                # =================================================
                # 图片消息
                # =================================================

                if message.photo:

                    print()
                    print(
                        "------------------------------------------"
                    )

                    print(
                        f"MESSAGE {message.id} "
                        f"IMAGE"
                    )

                    # ---------------------------------------------
                    # Caption
                    # ---------------------------------------------

                    caption = (
                        message.message
                        or ""
                    )

                    # ---------------------------------------------
                    # Caption 联系方式检测
                    # ---------------------------------------------

                    caption_contact = detect_contact(
                        caption
                    )

                    if caption_contact:

                        print(
                            "❌ CAPTION CONTACT FOUND"
                        )

                        print(
                            "PATTERN:",
                            caption_contact["type"]
                        )

                        print(
                            "MATCH:",
                            repr(
                                caption_contact[
                                    "match"
                                ]
                            )
                        )

                        print(
                            "ACTION: SKIP IMAGE"
                        )

                        processed.add(
                            message_key
                        )

                        continue

                    # ---------------------------------------------
                    # 创建临时目录
                    # ---------------------------------------------

                    temp_dir = tempfile.mkdtemp(
                        prefix="telegram_forward_"
                    )

                    try:

                        original_path = os.path.join(
                            temp_dir,
                            "original.jpg"
                        )

                        clean_path = os.path.join(
                            temp_dir,
                            "clean.jpg"
                        )

                        # =========================================
                        # 下载图片
                        # =========================================

                        print(
                            "DOWNLOADING IMAGE..."
                        )

                        downloaded = (
                            await client.download_media(
                                message,
                                file=original_path
                            )
                        )

                        if (
                            not downloaded
                            or not os.path.exists(
                                original_path
                            )
                        ):

                            print(
                                "❌ IMAGE DOWNLOAD FAILED"
                            )

                            continue

                        print(
                            "✅ IMAGE DOWNLOADED"
                        )

                        # =========================================
                        # QR 检测
                        # =========================================

                        qr_found, qr_data = detect_qr(
                            original_path
                        )

                        if qr_found:

                            print()
                            print(
                                "❌ QR CODE FOUND"
                            )

                            print(
                                "QR DATA:",
                                qr_data
                            )

                            print(
                                "ACTION: SKIP IMAGE"
                            )

                            processed.add(
                                message_key
                            )

                            continue

                        print(
                            "✅ QR CHECK PASSED"
                        )

                        # =========================================
                        # OCR 联系方式
                        # =========================================

                        image_contact = (
                            detect_image_contact(
                                original_path
                            )
                        )

                        if image_contact:

                            print()
                            print(
                                "❌ IMAGE CONTACT FOUND"
                            )

                            print(
                                "PATTERN:",
                                image_contact[
                                    "type"
                                ]
                            )

                            print(
                                "MATCH:",
                                repr(
                                    image_contact[
                                        "match"
                                    ]
                                )
                            )

                            print(
                                "ACTION: SKIP IMAGE"
                            )

                            processed.add(
                                message_key
                            )

                            continue

                        # =========================================
                        # 加水印
                        # =========================================

                        print()
                        print(
                            "ADDING WATERMARK..."
                        )

                        add_watermark(
                            original_path,
                            clean_path
                        )

                        print(
                            "✅ WATERMARK ADDED"
                        )

                        # =========================================
                        # Bot 上传
                        # =========================================

                        print(
                            "UPLOADING IMAGE..."
                        )

                        await send_photo(
                            bot,
                            target,
                            clean_path,
                            caption
                        )

                        print(
                            "✅ IMAGE FORWARDED"
                        )

                        # =========================================
                        # 只有发送成功之后才记录 processed
                        # =========================================

                        processed.add(
                            message_key
                        )

                    except Exception as e:

                        print()
                        print(
                            "❌ IMAGE PROCESS ERROR:"
                        )

                        print(
                            repr(e)
                        )

                        print(
                            "⚠️ 本条消息不会写入 processed.json"
                        )

                    finally:

                        shutil.rmtree(
                            temp_dir,
                            ignore_errors=True
                        )

                    continue

        except Exception as e:

            print()
            print(
                "❌ SOURCE SCAN ERROR:",
                source_name
            )

            print(
                repr(e)
            )

    # =====================================================
    # 保存 processed
    # =====================================================

    save_processed(
        processed
    )

    print()
    print(
        "=========================================="
    )

    print(
        "FINAL PROCESSED COUNT:",
        len(processed)
    )

    print(
        "=========================================="
    )

    # =====================================================
    # 关闭 Telegram Client
    # =====================================================

    try:

        await client.disconnect()

    except Exception:
        pass

    # =====================================================
    # 关闭 Bot
    # =====================================================

    try:

        await bot.shutdown()

    except Exception:
        pass

    print()
    print(
        "✅ ALL DONE"
    )


# =========================================================
# 程序入口
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
