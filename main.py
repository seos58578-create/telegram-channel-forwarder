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
from telegram.error import TelegramError


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
    for x in os.environ["SOURCE_CHANNELS"].split(",")
    if x.strip()
]

TARGET_CHANNEL = os.environ["TARGET_CHANNEL"].strip()

WATERMARK_TEXT = ""

PROCESSED_FILE = "processed.json"

# 每个源频道最多检查多少条。
# 如果设置为空、0 或 unlimited，则当天消息全部扫描。
SCAN_LIMIT_RAW = os.environ.get("SCAN_LIMIT", "0").strip()

if SCAN_LIMIT_RAW.lower() in ("", "0", "unlimited", "none"):
    SCAN_LIMIT = None
else:
    SCAN_LIMIT = int(SCAN_LIMIT_RAW)


# =========================================================
# 联系方式识别规则
# =========================================================

CONTACT_PATTERNS = [

    # Telegram 链接
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

    # Telegram 用户名
    (
        "TELEGRAM_USERNAME",
        re.compile(
            r"@[A-Za-z][A-Za-z0-9_]{4,31}"
        )
    ),

    # WhatsApp
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

    # 微信
    (
        "WECHAT",
        re.compile(
            r"\bwechat\b",
            re.IGNORECASE
        )
    ),

    # 中国手机号
    (
        "CN_PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?86[- ]?)?"
            r"1[3-9]\d{9}"
            r"(?!\d)"
        )
    ),

    # 越南手机号
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


# =========================================================
# processed.json
# =========================================================

def load_processed():
    if not os.path.exists(PROCESSED_FILE):
        return set()

    try:
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data.get("messages", [])

        if not isinstance(messages, list):
            return set()

        return set(str(x) for x in messages)

    except Exception as e:
        print("⚠️ processed.json 读取失败:", e)
        return set()


def save_processed(processed):
    # 防止文件无限增长
    latest = list(processed)[-5000:]

    data = {
        "messages": latest
    }

    temp_file = PROCESSED_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(temp_file, PROCESSED_FILE)


# =========================================================
# 北京时间判断
# =========================================================

def to_beijing(dt):
    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(BEIJING_TZ)


def is_today(message):
    if not message.date:
        return False

    msg_time = to_beijing(message.date)
    now = datetime.now(BEIJING_TZ)

    return msg_time.date() == now.date()


# =========================================================
# 文本联系方式检测
# =========================================================

def detect_text_contact(text):
    if not text:
        return []

    results = []

    for contact_type, pattern in CONTACT_PATTERNS:
        matches = pattern.findall(text)

        if matches:
            results.append({
                "type": contact_type,
                "matches": matches
            })

    return results


def text_has_contact(text):
    return len(detect_text_contact(text)) > 0


# =========================================================
# OCR
# =========================================================

def ocr_image_variants(image_path):

    results = []

    try:
        image = Image.open(image_path).convert("RGB")

        # ORIGINAL
        results.append(
            (
                "ORIGINAL",
                pytesseract.image_to_string(
                    image,
                    lang="eng+chi_sim+vie+por"
                )
            )
        )

        # GRAYSCALE
        gray = ImageOps.grayscale(image)

        results.append(
            (
                "GRAYSCALE",
                pytesseract.image_to_string(
                    gray,
                    lang="eng+chi_sim+vie+por"
                )
            )
        )

        # ENHANCED
        enhanced = ImageEnhance.Contrast(gray).enhance(2.0)

        results.append(
            (
                "ENHANCED",
                pytesseract.image_to_string(
                    enhanced,
                    lang="eng+chi_sim+vie+por"
                )
            )
        )

        # THRESHOLD
        cv_image = cv2.imread(image_path)

        if cv_image is not None:

            gray_cv = cv2.cvtColor(
                cv_image,
                cv2.COLOR_BGR2GRAY
            )

            _, threshold = cv2.threshold(
                gray_cv,
                150,
                255,
                cv2.THRESH_BINARY
            )

            threshold_pil = Image.fromarray(threshold)

            results.append(
                (
                    "THRESHOLD",
                    pytesseract.image_to_string(
                        threshold_pil,
                        lang="eng+chi_sim+vie+por"
                    )
                )
            )

    except Exception as e:
        print("❌ OCR ERROR:", e)

    return results


# =========================================================
# 图片二维码检测
# =========================================================

def detect_qr_code(image_path):

    try:
        image = cv2.imread(image_path)

        if image is None:
            return False

        detector = cv2.QRCodeDetector()

        # 普通检测
        try:
            data, points, _ = detector.detectAndDecode(image)

            if points is not None:
                if data:
                    print("🚫 QR CODE FOUND:", data)
                else:
                    print("🚫 QR CODE FOUND")
                return True
        except Exception:
            pass

        # 灰度检测
        try:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            data, points, _ = detector.detectAndDecode(gray)

            if points is not None:
                if data:
                    print("🚫 QR CODE FOUND:", data)
                else:
                    print("🚫 QR CODE FOUND")
                return True

        except Exception:
            pass

        # 多二维码
        try:
            result = detector.detectAndDecodeMulti(image)

            if result is not None:

                retval = result[0]

                if retval:
                    print("🚫 MULTIPLE QR CODE FOUND")
                    return True

        except Exception:
            pass

    except Exception as e:
        print("⚠️ QR DETECTION ERROR:", e)

    return False


# =========================================================
# 图片联系方式检测
# =========================================================

def detect_image_contact(image_path):

    # -----------------------------------------------------
    # 第一层：二维码
    # -----------------------------------------------------

    if detect_qr_code(image_path):

        return {
            "found": True,
            "reason": "QR_CODE"
        }


    # -----------------------------------------------------
    # 第二层：OCR
    # -----------------------------------------------------

    ocr_results = ocr_image_variants(image_path)

    detections = {}

    for pass_name, text in ocr_results:

        if not text:
            continue

        for contact_type, pattern in CONTACT_PATTERNS:

            matches = pattern.findall(text)

            if matches:

                if contact_type not in detections:
                    detections[contact_type] = []

                detections[contact_type].append({
                    "pass": pass_name,
                    "matches": matches
                })


    # -----------------------------------------------------
    # 强联系方式：
    # Telegram / WhatsApp / WeChat
    # 一次识别即可判定
    # -----------------------------------------------------

    for contact_type in STRONG_CONTACT_TYPES:

        hits = detections.get(contact_type, [])

        if not hits:
            continue

        print("")
        print("🚫 IMAGE CONTACT FOUND")
        print("PATTERN:", contact_type)

        for item in hits:
            print(
                "OCR PASS:",
                item["pass"],
                "MATCH:",
                item["matches"]
            )

        return {
            "found": True,
            "reason": contact_type
        }


    # -----------------------------------------------------
    # 手机号：
    # 至少两个独立 OCR 结果确认
    # 防止 OCR 把普通数字识别成电话号码
    # -----------------------------------------------------

    for contact_type in PHONE_CONTACT_TYPES:

        hits = detections.get(contact_type, [])

        if not hits:
            continue

        pass_names = set(
            item["pass"]
            for item in hits
        )

        if len(pass_names) >= 2:

            print("")
            print("🚫 IMAGE PHONE CONTACT CONFIRMED")
            print("PATTERN:", contact_type)
            print("OCR PASS COUNT:", len(pass_names))
            print("OCR PASSES:", ", ".join(pass_names))

            for item in hits:
                print(
                    "MATCH:",
                    item["matches"]
                )

            return {
                "found": True,
                "reason": contact_type
            }

        else:

            print("")
            print("⚠️ PHONE OCR HIT BUT NOT CONFIRMED")
            print("PATTERN:", contact_type)

            for item in hits:
                print(
                    "MATCH:",
                    item["matches"]
                )

            print(
                "OCR PASS COUNT:",
                len(pass_names)
            )

            print(
                "OCR PASSES:",
                ", ".join(pass_names)
            )

            print(
                "ACTION: IGNORE FALSE POSITIVE"
            )


    print("")
    print("✅ IMAGE OCR: NO CONTACT FOUND")

    return {
        "found": False,
        "reason": None
    }


# =========================================================
# 水印字体
# =========================================================

def get_font(size):

    font_paths = [

        "/usr/share/fonts/opentype/noto/"
        "NotoSansCJK-Regular.ttc",

        "/usr/share/fonts/opentype/noto/"
        "NotoSansCJKSC-Regular.otf",

        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans.ttf",
    ]

    for path in font_paths:

        if os.path.exists(path):

            try:
                return ImageFont.truetype(
                    path,
                    size
                )
            except Exception:
                pass

    return ImageFont.load_default()


# =========================================================
# 图片加水印
# =========================================================

def add_watermark(
    image_path,
    output_path
):

    try:

        image = Image.open(
            image_path
        ).convert("RGBA")

        width, height = image.size

        # 根据图片大小自动计算字体
        font_size = max(
            24,
            int(min(width, height) * 0.045)
        )

        font = get_font(font_size)

        # 透明图层
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
            font=font,
            stroke_width=2
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        margin = max(
            15,
            int(min(width, height) * 0.02)
        )

        x = width - text_width - margin
        y = height - text_height - margin

        # 黑色描边 + 白字
        draw.text(
            (x, y),
            WATERMARK_TEXT,
            font=font,
            fill=(255, 255, 255, 210),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 180)
        )

        result = Image.alpha_composite(
            image,
            overlay
        )

        result.convert("RGB").save(
            output_path,
            "JPEG",
            quality=95
        )

        return True

    except Exception as e:

        print(
            "❌ WATERMARK ERROR:",
            e
        )

        return False


# =========================================================
# Telegram Bot
# =========================================================

async def bot_send_text(
    bot,
    target,
    text
):

    if not text:
        return False

    # Telegram 普通消息限制约 4096 字符
    max_length = 4000

    chunks = [
        text[i:i + max_length]
        for i in range(
            0,
            len(text),
            max_length
        )
    ]

    try:

        for chunk in chunks:

            await bot.send_message(
                chat_id=target,
                text=chunk
            )

        return True

    except TelegramError as e:

        print(
            "❌ BOT TEXT ERROR:",
            e
        )

        return False

    except Exception as e:

        print(
            "❌ TEXT SEND ERROR:",
            e
        )

        return False


async def bot_send_photo(
    bot,
    target,
    image_path,
    caption=None
):

    try:

        caption = caption or ""

        # Telegram 图片 caption 最大约 1024 字符
        if len(caption) <= 1024:

            with open(
                image_path,
                "rb"
            ) as photo:

                await bot.send_photo(
                    chat_id=target,
                    photo=photo,
                    caption=caption
                )

        else:

            # 图片先发
            with open(
                image_path,
                "rb"
            ) as photo:

                await bot.send_photo(
                    chat_id=target,
                    photo=photo
                )

            # 长文字单独发送
            await bot_send_text(
                bot,
                target,
                caption
            )

        return True

    except TelegramError as e:

        print(
            "❌ BOT IMAGE ERROR:",
            e
        )

        return False

    except Exception as e:

        print(
            "❌ IMAGE SEND ERROR:",
            e
        )

        return False


# =========================================================
# 处理一条消息
# =========================================================

async def process_message(
    client,
    bot,
    target_entity,
    source_name,
    message,
    processed
):

    message_key = (
        f"{source_name}:{message.id}"
    )

    # -----------------------------------------------------
    # 已经处理过
    # -----------------------------------------------------

    if message_key in processed:

        print(
            f"MESSAGE {message.id} "
            f"already processed."
        )

        return


    # -----------------------------------------------------
    # 空消息
    # -----------------------------------------------------

    text = (
        message.message
        or ""
    ).strip()

    if not message.media and not text:

        print(
            f"MESSAGE {message.id} "
            f"empty, skipped."
        )

        processed.add(
            message_key
        )

        return


    # -----------------------------------------------------
    # 消息时间
    # -----------------------------------------------------

    msg_time = to_beijing(
        message.date
    )

    print("")
    print("=" * 70)

    print(
        "PROCESS MESSAGE:",
        message.id
    )

    print(
        "SOURCE:",
        source_name
    )

    print(
        "TIME:",
        msg_time
    )

    # =====================================================
    # 文字消息
    # =====================================================

    if not message.photo:

        if text_has_contact(text):

            print(
                "🚫 TEXT CONTACT FOUND"
            )

            print(
                "ACTION: SKIP"
            )

            processed.add(
                message_key
            )

            return


        print(
            "📤 SEND TEXT"
        )

        success = await bot_send_text(
            bot,
            target_entity,
            text
        )

        if success:

            print(
                "✅ TEXT PUBLISHED"
            )

            processed.add(
                message_key
            )

        return


    # =====================================================
    # 图片消息
    # =====================================================

    print(
        "🖼 IMAGE MESSAGE"
    )

    temp_dir = tempfile.mkdtemp(
        prefix="telegram_forward_"
    )

    try:

        original_image = os.path.join(
            temp_dir,
            "original.jpg"
        )

        clean_image = os.path.join(
            temp_dir,
            "clean.jpg"
        )

        # -------------------------------------------------
        # 下载图片
        # -------------------------------------------------

        print(
            "⬇️ DOWNLOADING IMAGE..."
        )

        await client.download_media(
            message,
            file=original_image
        )

        if not os.path.exists(
            original_image
        ):

            print(
                "❌ IMAGE DOWNLOAD FAILED"
            )

            return


        # -------------------------------------------------
        # 图片 Caption 联系方式检测
        # -------------------------------------------------

        if text_has_contact(text):

            print(
                "🚫 IMAGE CAPTION CONTACT FOUND"
            )

            print(
                "ACTION: SKIP WHOLE IMAGE"
            )

            processed.add(
                message_key
            )

            return


        # -------------------------------------------------
        # QR + OCR
        # -------------------------------------------------

        contact_result = (
            detect_image_contact(
                original_image
            )
        )

        if contact_result["found"]:

            print(
                "🚫 IMAGE SKIPPED"
            )

            print(
                "REASON:",
                contact_result["reason"]
            )

            processed.add(
                message_key
            )

            return


        # -------------------------------------------------
        # 加水印
        # -------------------------------------------------

        print(
            "💧 ADD WATERMARK..."
        )

        watermark_success = (
            add_watermark(
                original_image,
                clean_image
            )
        )

        if not watermark_success:

            print(
                "❌ WATERMARK FAILED"
            )

            return


        # -------------------------------------------------
        # Bot 发布
        # -------------------------------------------------

        print(
            "📤 SEND IMAGE..."
        )

        success = await bot_send_photo(
            bot,
            target_entity,
            clean_image,
            text
        )

        if success:

            print(
                "✅ IMAGE PUBLISHED"
            )

            processed.add(
                message_key
            )

        else:

            print(
                "❌ IMAGE PUBLISH FAILED"
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


# =========================================================
# 主程序
# =========================================================

async def main():

    print("")
    print("=" * 70)
    print("TELEGRAM AUTO FORWARDER")
    print("=" * 70)

    now = datetime.now(
        BEIJING_TZ
    )

    print(
        "BEIJING TIME:",
        now
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

    print(
        "CONNECTING TELEGRAM..."
    )

    client = TelegramClient(
        StringSession(
            TELEGRAM_SESSION
        ),
        API_ID,
        API_HASH
    )

    await client.start()

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

        bot_info = await bot.get_me()

        print(
            "BOT:",
            bot_info.username
        )

        # -------------------------------------------------
        # 检查目标频道
        # -------------------------------------------------

        target_entity = await client.get_entity(
            TARGET_CHANNEL
        )

        print(
            "TARGET ENTITY:",
            target_entity
        )

        print(
            "TARGET TITLE:",
            getattr(
                target_entity,
                "title",
                ""
            )
        )

        print(
            "TARGET ID:",
            getattr(
                target_entity,
                "id",
                ""
            )
        )

        # -------------------------------------------------
        # Bot 获取目标频道
        # -------------------------------------------------

        try:

            bot_target = await bot.get_chat(
                chat_id=TARGET_CHANNEL
            )

            print(
                "BOT TARGET:",
                bot_target.title
            )

            print(
                "✅ BOT TARGET ACCESS OK"
            )

        except Exception as e:

            print("")
            print(
                "❌ BOT TARGET ACCESS ERROR"
            )

            print(
                str(e)
            )

            print("")
            print(
                "请检查："
            )

            print(
                "1. Bot 是否已经加入目标频道"
            )

            print(
                "2. Bot 是否设置为管理员"
            )

            print(
                "3. 管理员权限是否开启 Post Messages"
            )

            print("")

            return


        # =================================================
        # 第一步：
        # 收集所有源频道今天的消息
        # =================================================

        all_messages = []

        for source_index, source_name in enumerate(
            SOURCE_CHANNELS
        ):

            print("")
            print(
                "=" * 70
            )

            print(
                "SOURCE CHANNEL:",
                source_name
            )

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

                print(
                    "❌ SOURCE CHANNEL ERROR:",
                    source_name,
                    e
                )

                continue


            count = 0

            # -------------------------------------------------
            # Telegram 返回顺序是：
            # 最新 → 最旧
            #
            # 我们这里先全部收集
            # 后面统一排序
            # -------------------------------------------------

            async for message in client.iter_messages(
                source_entity,
                limit=SCAN_LIMIT
            ):

                # 今天
                if is_today(message):

                    all_messages.append({
                        "source_name": source_name,
                        "source_index": source_index,
                        "message": message
                    })

                    count += 1

                else:

                    # 因为 iter_messages 是
                    # 最新 → 最旧
                    #
                    # 一旦发现已经不是今天
                    # 后面的也不可能是今天
                    if message.date:

                        message_time = (
                            to_beijing(
                                message.date
                            )
                        )

                        today = datetime.now(
                            BEIJING_TZ
                        ).date()

                        if (
                            message_time.date()
                            < today
                        ):
                            break


            print(
                "TODAY MESSAGES:",
                count
            )


        # =================================================
        # 第二步：
        # 全部消息按照原发布时间排序
        # =================================================

        print("")
        print(
            "=" * 70
        )

        print(
            "TOTAL TODAY MESSAGES:",
            len(all_messages)
        )


        def sort_key(item):

            message = item["message"]

            message_time = (
                to_beijing(
                    message.date
                )
            )

            return (
                message_time,
                item["source_index"],
                message.id
            )


        all_messages.sort(
            key=sort_key
        )


        # =================================================
        # 输出最终发布顺序
        # =================================================

        print("")
        print(
            "FINAL PUBLISH ORDER:"
        )

        for index, item in enumerate(
            all_messages,
            start=1
        ):

            message = item["message"]

            print(
                f"{index}. "
                f"{item['source_name']} "
                f"MSG={message.id} "
                f"TIME={to_beijing(message.date)}"
            )


        # =================================================
        # 第三步：
        # 严格按照排序后的顺序逐条发布
        # =================================================

        print("")
        print(
            "=" * 70
        )

        print(
            "START PUBLISHING..."
        )


        for index, item in enumerate(
            all_messages,
            start=1
        ):

            source_name = item[
                "source_name"
            ]

            message = item[
                "message"
            ]

            print("")
            print(
                f"[{index}/{len(all_messages)}]"
            )

            await process_message(
                client=client,
                bot=bot,
                target_entity=TARGET_CHANNEL,
                source_name=source_name,
                message=message,
                processed=processed
            )

            # -------------------------------------------------
            # 每处理一条就保存
            #
            # GitHub Actions 中途失败时，
            # 已成功发布的不会再次发布
            # -------------------------------------------------

            save_processed(
                processed
            )

            # -------------------------------------------------
            # 稍微等待一下
            # 避免发送过快
            # -------------------------------------------------

            await asyncio.sleep(1)


        # =================================================
        # 最终保存
        # =================================================

        save_processed(
            processed
        )

        print("")
        print(
            "=" * 70
        )

        print(
            "✅ ALL DONE"
        )

        print(
            "PROCESSED COUNT:",
            len(processed)
        )

        print(
            "=" * 70
        )


    finally:

        await client.disconnect()

        try:
            await bot.shutdown()
        except Exception:
            pass


# =========================================================
# 启动
# =========================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "PROGRAM STOPPED"
        )

    except Exception as e:

        print(
            "❌ FATAL ERROR:",
            e
        )

        raise
