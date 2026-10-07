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
TELEGRAM_SESSION = os.environ["TELEGRAM_SESSION"]

BOT_TOKEN = os.environ["BOT_TOKEN"]

SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ.get("SOURCE_CHANNELS", "").split(",")
    if x.strip()
]

TARGET_CHANNEL = os.environ["TARGET_CHANNEL"].strip()

SCAN_LIMIT = int(os.environ.get("SCAN_LIMIT", "500"))

PROCESSED_FILE = "processed.json"

WATERMARK_TEXT = ""


# =========================================================
# 联系方式识别规则
# =========================================================
#
# 注意：
# 这里故意不再识别：
#
# 8位普通数字
# Telegram 单词
# 客服
# 联系我
# 私聊
# 添加好友
# 二维码
#
# 因为 OCR 很容易产生误判。
#


CONTACT_PATTERNS = [

    # -----------------------------------------------------
    # Telegram 链接
    # 例如：
    # https://t.me/username
    # t.me/username
    # telegram.me/username
    # -----------------------------------------------------
    (
        "TELEGRAM_LINK",
        re.compile(
            r"(?:https?://)?(?:www\.)?"
            r"(?:t\.me|telegram\.me)/"
            r"[A-Za-z0-9_+/=?-]{4,64}",
            re.IGNORECASE
        )
    ),

    # -----------------------------------------------------
    # Telegram 用户名
    #
    # 必须：
    # @ + 英文字母开头
    # 至少 5 个字符
    #
    # 避免把普通 @ 符号误认为联系方式
    # -----------------------------------------------------
    (
        "TELEGRAM_USERNAME",
        re.compile(
            r"@[A-Za-z][A-Za-z0-9_]{4,31}"
        )
    ),

    # -----------------------------------------------------
    # WhatsApp 链接
    # -----------------------------------------------------
    (
        "WHATSAPP_LINK",
        re.compile(
            r"(?:https?://)?(?:www\.)?wa\.me/\d{7,15}",
            re.IGNORECASE
        )
    ),

    # -----------------------------------------------------
    # 微信
    #
    # 只识别英文 wechat
    # 不使用「微信」「客服」等 OCR 容易误判的中文词
    # -----------------------------------------------------
    (
        "WECHAT",
        re.compile(
            r"\bwechat\b",
            re.IGNORECASE
        )
    ),

    # -----------------------------------------------------
    # 中国大陆手机号
    #
    # 例如：
    # 13812345678
    # +8613812345678
    # 86 13812345678
    # -----------------------------------------------------
    (
        "CN_PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?86[- ]?)?"
            r"1[3-9]\d{9}"
            r"(?!\d)"
        )
    ),

    # -----------------------------------------------------
    # 越南手机号
    #
    # 例如：
    # 0912345678
    # +84912345678
    # 84912345678
    # -----------------------------------------------------
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
# 强联系方式
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
# 时间
# =========================================================

def beijing_now():
    return datetime.now(BEIJING_TZ)


def is_today(message):
    """
    判断 Telegram 消息是否为北京时间今天
    """

    if not message.date:
        return False

    msg_time = message.date

    if msg_time.tzinfo is None:
        msg_time = msg_time.replace(tzinfo=timezone.utc)

    msg_time = msg_time.astimezone(BEIJING_TZ)

    return msg_time.date() == beijing_now().date()


# =========================================================
# processed.json
# =========================================================

def load_processed():

    if not os.path.exists(PROCESSED_FILE):
        return set()

    try:
        with open(
            PROCESSED_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return set(
            str(x)
            for x in data.get("messages", [])
        )

    except Exception as e:

        print(
            "⚠️ processed.json 读取失败:",
            e
        )

        return set()


def save_processed(processed):

    data = {
        "messages": list(processed)[-5000:]
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

        match = pattern.search(text)

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

def detect_qr(image_path):

    try:

        image = cv2.imread(image_path)

        if image is None:
            return False, []

        detector = cv2.QRCodeDetector()

        detected_data = []

        # ---------------------------------------------
        # 普通检测
        # ---------------------------------------------

        try:

            data, points, _ = detector.detectAndDecode(
                image
            )

            if data:

                detected_data.append(data)

        except Exception:
            pass

        # ---------------------------------------------
        # 灰度检测
        # ---------------------------------------------

        try:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            data, points, _ = detector.detectAndDecode(
                gray
            )

            if data:

                detected_data.append(data)

        except Exception:
            pass

        # ---------------------------------------------
        # 多二维码检测
        # ---------------------------------------------

        try:

            result = detector.detectAndDecodeMulti(
                image
            )

            if result and len(result) >= 2:

                ok = result[0]
                decoded_info = result[1]

                if ok and decoded_info:

                    for item in decoded_info:

                        if item:
                            detected_data.append(item)

        except Exception:
            pass

        detected_data = list(
            dict.fromkeys(
                x for x in detected_data if x
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
# OCR
# =========================================================

def run_ocr_variants(image_path):

    results = []

    try:

        original = Image.open(
            image_path
        ).convert("RGB")

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
        ).enhance(2.0)

        enhanced = ImageEnhance.Sharpness(
            enhanced
        ).enhance(2.0)

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
            lambda p: 255 if p > 160 else 0
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

def detect_image_contact(image_path):

    ocr_results = run_ocr_variants(
        image_path
    )

    if not ocr_results:

        return None

    # =====================================================
    # 记录所有 OCR 命中
    # =====================================================

    detections = {}

    for pass_name, text in ocr_results:

        print()
        print(
            f"OCR {pass_name}:"
        )

        print(
            repr(text[:2000])
        )

        result = detect_contact(
            text
        )

        if result:

            key = result["type"]

            if key not in detections:

                detections[key] = []

            detections[key].append(
                {
                    "pass": pass_name,
                    "match": result["match"],
                    "pattern": result["pattern"]
                }
            )

            print(
                "CONTACT DETECTED:",
                result["type"]
            )

            print(
                "MATCH:",
                repr(result["match"])
            )

    # =====================================================
    # 没有任何联系方式
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
    # Telegram 用户名
    # WhatsApp URL
    # WeChat
    #
    # 一次 OCR 命中即可
    # =====================================================

    for contact_type in STRONG_CONTACT_TYPES:

        if contact_type in detections:

            hit = detections[
                contact_type
            ][0]

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
                repr(hit["match"])
            )

            print(
                "OCR PASS:",
                hit["pass"]
            )

            print(
                "========================================"
            )

            return hit

    # =====================================================
    # 手机号码
    #
    # 必须至少 2 个 OCR PASS 命中
    #
    # 这样可以防止：
    #
    # OCR：
    # 27711154
    #
    # 这种随机数字误判
    # =====================================================

    for contact_type in PHONE_CONTACT_TYPES:

        hits = detections.get(
            contact_type,
            []
        )

        unique_passes = set(
            x["pass"]
            for x in hits
        )

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
                repr(hit["match"])
            )

            print(
                "OCR PASS COUNT:",
                len(unique_passes)
            )

            print(
                "OCR PASSES:",
                ", ".join(
                    sorted(unique_passes)
                )
            )

            print(
                "========================================"
            )

            return hit

        else:

            print()
            print(
                "⚠️ PHONE OCR HIT BUT NOT CONFIRMED"
            )

            print(
                "PATTERN:",
                contact_type
            )

            print(
                "OCR PASS COUNT:",
                len(unique_passes)
            )

            print(
                "ACTION: IGNORE FALSE POSITIVE"
            )

    # =====================================================
    # 没有达到确认条件
    # =====================================================

    print()
    print(
        "✅ IMAGE CONTACT NOT CONFIRMED"
    )

    return None


# =========================================================
# 水印
# =========================================================

def add_watermark(
    input_path,
    output_path
):

    image = Image.open(
        input_path
    ).convert("RGBA")

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(
        overlay
    )

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

    bbox = draw.textbbox(
        (0, 0),
        WATERMARK_TEXT,
        font=font
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

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

    # 半透明黑色背景
    padding = 10

    draw.rounded_rectangle(
        (
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        ),
        radius=8,
        fill=(0, 0, 0, 120)
    )

    draw.text(
        (x, y),
        WATERMARK_TEXT,
        font=font,
        fill=(255, 255, 255, 220),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 180)
    )

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
# Telegram Bot 发送文本
# =========================================================

async def send_text(
    bot,
    target,
    text
):

    if not text:
        return

    # Telegram 单条文本限制
    max_length = 4000

    if len(text) <= max_length:

        await bot.send_message(
            chat_id=target,
            text=text
        )

        return

    # 超长文本切割
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

        await asyncio.sleep(0.5)


# =========================================================
# Telegram Bot 发送图片
# =========================================================

async def send_photo(
    bot,
    target,
    image_path,
    caption=None
):

    if caption and len(caption) > 1024:

        await bot.send_photo(
            chat_id=target,
            photo=open(
                image_path,
                "rb"
            )
        )

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

    # -----------------------------------------------------
    # 参数检查
    # -----------------------------------------------------

    if not SOURCE_CHANNELS:

        print(
            "❌ SOURCE_CHANNELS 未配置"
        )

        return

    # -----------------------------------------------------
    # Target 格式
    # -----------------------------------------------------

    target = TARGET_CHANNEL

    if not target.startswith("@"):

        if not target.lstrip("-").isdigit():

            target = "@" + target

    # -----------------------------------------------------
    # processed
    # -----------------------------------------------------

    processed = load_processed()

    print(
        "PROCESSED COUNT:",
        len(processed)
    )

    # -----------------------------------------------------
    # Telegram 普通账号
    # -----------------------------------------------------

    client = TelegramClient(
        StringSession(
            TELEGRAM_SESSION
        ),
        API_ID,
        API_HASH
    )

    print(
        "CONNECTING TELEGRAM..."
    )

    await client.connect()

    if not await client.is_user_authorized():

        print(
            "❌ TELEGRAM SESSION 未授权"
        )

        await client.disconnect()

        return

    print(
        "✅ TELEGRAM CONNECTED"
    )

    # -----------------------------------------------------
    # Bot
    # -----------------------------------------------------

    bot = Bot(
        token=BOT_TOKEN
    )

    try:

        bot_me = await bot.get_me()

        print(
            "BOT:",
            bot_me.username
        )

        target_entity = await bot.get_chat(
            target
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
            target_entity.id
        )

    except Exception as e:

        print(
            "❌ TARGET CHANNEL ERROR:",
            e
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

        try:

            source_entity = await client.get_entity(
                source_name
            )

            print(
                "SOURCE ENTITY:",
                source_entity
            )

        except Exception as e:

            print(
                "❌ SOURCE ENTITY ERROR:",
                source_name,
                e
            )

            continue

        # -------------------------------------------------
        # 获取消息
        # -------------------------------------------------

        try:

            messages = client.iter_messages(
                source_entity,
                limit=SCAN_LIMIT
            )

            async for message in messages:

                # -----------------------------------------
                # 只处理今天北京时间消息
                # -----------------------------------------

                if not is_today(message):

                    continue

                message_key = (
                    f"{source_name}:{message.id}"
                )

                # -----------------------------------------
                # 已处理
                # -----------------------------------------

                if message_key in processed:

                    print(
                        f"MESSAGE {message.id} "
                        f"already processed."
                    )

                    continue

                # -----------------------------------------
                # 空消息
                # -----------------------------------------

                if not message.message and not message.media:

                    print(
                        f"MESSAGE {message.id} "
                        f"empty, skipped."
                    )

                    processed.add(
                        message_key
                    )

                    continue

                # =========================================
                # 文本
                # =========================================

                if message.message and not message.photo:

                    print()
                    print(
                        f"MESSAGE {message.id} "
                        f"TEXT MESSAGE"
                    )

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
                            "❌ SKIP TEXT"
                        )

                        processed.add(
                            message_key
                        )

                        continue

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

                        print(
                            "❌ TEXT SEND ERROR:",
                            e
                        )

                    continue

                # =========================================
                # 图片
                # =========================================

                if message.photo:

                    print()
                    print(
                        f"MESSAGE {message.id} "
                        f"IMAGE"
                    )

                    caption = (
                        message.message or ""
                    )

                    # -------------------------------------
                    # Caption 联系方式
                    # -------------------------------------

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
                                caption_contact["match"]
                            )
                        )

                        print(
                            "❌ SKIP IMAGE"
                        )

                        processed.add(
                            message_key
                        )

                        continue

                    # -------------------------------------
                    # 临时目录
                    # -------------------------------------

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

                        # ---------------------------------
                        # 下载图片
                        # ---------------------------------

                        print(
                            "DOWNLOADING IMAGE..."
                        )

                        await client.download_media(
                            message,
                            file=original_path
                        )

                        if not os.path.exists(
                            original_path
                        ):

                            print(
                                "❌ IMAGE DOWNLOAD FAILED"
                            )

                            continue

                        # ---------------------------------
                        # QR 检测
                        # ---------------------------------

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
                                "❌ SKIP IMAGE"
                            )

                            processed.add(
                                message_key
                            )

                            continue

                        print(
                            "✅ QR CHECK PASSED"
                        )

                        # ---------------------------------
                        # OCR 联系方式
                        # ---------------------------------

                        image_contact = detect_image_contact(
                            original_path
                        )

                        if image_contact:

                            print()
                            print(
                                "❌ IMAGE CONTACT FOUND"
                            )

                            print(
                                "PATTERN:",
                                image_contact["type"]
                            )

                            print(
                                "MATCH:",
                                repr(
                                    image_contact["match"]
                                )
                            )

                            print(
                                "❌ SKIP IMAGE"
                            )

                            processed.add(
                                message_key
                            )

                            continue

                        # ---------------------------------
                        # 加水印
                        # ---------------------------------

                        print(
                            "ADDING WATERMARK..."
                        )

                        add_watermark(
                            original_path,
                            clean_path
                        )

                        # ---------------------------------
                        # Bot 上传
                        # ---------------------------------

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

                        processed.add(
                            message_key
                        )

                    except Exception as e:

                        print(
                            "❌ IMAGE PROCESS ERROR:",
                            e
                        )

                    finally:

                        shutil.rmtree(
                            temp_dir,
                            ignore_errors=True
                        )

                    continue

        except Exception as e:

            print(
                "❌ SOURCE SCAN ERROR:",
                source_name,
                e
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
    # 关闭
    # =====================================================

    await client.disconnect()

    await bot.shutdown()

    print(
        "✅ ALL DONE"
    )


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
