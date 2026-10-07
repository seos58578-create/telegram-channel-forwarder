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
# 1. 基础配置
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

TARGET_CHANNEL = os.environ.get(
    "TARGET_CHANNEL",
    ""
).strip()

# 图片水印
WATERMARK_TEXT = ""

# 已处理记录
PROCESSED_FILE = "processed.json"


# =========================================================
# 2. SCAN_LIMIT
#
# 0 / 空 / unlimited = 扫描当天全部消息
# =========================================================

SCAN_LIMIT_RAW = os.environ.get(
    "SCAN_LIMIT",
    "0"
).strip()

if SCAN_LIMIT_RAW.lower() in (
    "",
    "0",
    "unlimited",
    "none"
):
    SCAN_LIMIT = None
else:
    SCAN_LIMIT = int(SCAN_LIMIT_RAW)


# =========================================================
# 3. 联系方式检测规则
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

    # WeChat
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


# 强联系方式
STRONG_CONTACT_TYPES = {
    "TELEGRAM_LINK",
    "TELEGRAM_USERNAME",
    "WHATSAPP_LINK",
    "WECHAT",
}


# 手机号码需要多个 OCR 模式确认
PHONE_CONTACT_TYPES = {
    "CN_PHONE",
    "VN_PHONE",
}


# =========================================================
# 4. TARGET_CHANNEL 标准化
# =========================================================

def normalize_target(target):
    """
    支持：

    @channelname
    channelname
    -1001234567890
    """

    target = target.strip()

    if not target:
        raise ValueError(
            "TARGET_CHANNEL 不能为空"
        )

    # Telegram numeric ID
    if re.fullmatch(
        r"-?\d+",
        target
    ):
        return int(target)

    # 用户名
    if not target.startswith("@"):
        target = "@" + target

    return target


TARGET_CHAT_ID = normalize_target(
    TARGET_CHANNEL
)


# =========================================================
# 5. processed.json
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

        if not isinstance(
            messages,
            list
        ):
            return set()

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


def save_processed(processed):

    # 最多保存 5000 条
    latest = list(processed)[-5000:]

    data = {
        "messages": latest
    }

    temp_file = (
        PROCESSED_FILE
        + ".tmp"
    )

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


# =========================================================
# 6. 时间处理
# =========================================================

def to_beijing(dt):

    if dt is None:
        return None

    if dt.tzinfo is None:

        dt = dt.replace(
            tzinfo=timezone.utc
        )

    return dt.astimezone(
        BEIJING_TZ
    )


def get_today():

    return datetime.now(
        BEIJING_TZ
    ).date()


def is_today(message):

    if not message.date:
        return False

    msg_time = to_beijing(
        message.date
    )

    return (
        msg_time.date()
        == get_today()
    )


# =========================================================
# 7. 文字联系方式检测
# =========================================================

def detect_text_contact(text):

    if not text:
        return []

    results = []

    for contact_type, pattern in CONTACT_PATTERNS:

        matches = pattern.findall(
            text
        )

        if matches:

            results.append({
                "type": contact_type,
                "matches": matches
            })

    return results


def text_has_contact(text):

    return bool(
        detect_text_contact(text)
    )


# =========================================================
# 8. OCR
# =========================================================

def ocr_image_variants(
    image_path
):

    results = []

    try:

        image = Image.open(
            image_path
        ).convert("RGB")


        # -------------------------------------------------
        # ORIGINAL
        # -------------------------------------------------

        try:

            text = pytesseract.image_to_string(
                image,
                lang="eng+chi_sim+vie+por"
            )

            results.append(
                (
                    "ORIGINAL",
                    text
                )
            )

        except Exception as e:

            print(
                "⚠️ OCR ORIGINAL ERROR:",
                e
            )


        # -------------------------------------------------
        # GRAYSCALE
        # -------------------------------------------------

        try:

            gray = ImageOps.grayscale(
                image
            )

            text = pytesseract.image_to_string(
                gray,
                lang="eng+chi_sim+vie+por"
            )

            results.append(
                (
                    "GRAYSCALE",
                    text
                )
            )

        except Exception as e:

            print(
                "⚠️ OCR GRAYSCALE ERROR:",
                e
            )


        # -------------------------------------------------
        # ENHANCED
        # -------------------------------------------------

        try:

            enhanced = (
                ImageEnhance.Contrast(
                    gray
                ).enhance(2.0)
            )

            text = pytesseract.image_to_string(
                enhanced,
                lang="eng+chi_sim+vie+por"
            )

            results.append(
                (
                    "ENHANCED",
                    text
                )
            )

        except Exception as e:

            print(
                "⚠️ OCR ENHANCED ERROR:",
                e
            )


        # -------------------------------------------------
        # THRESHOLD
        # -------------------------------------------------

        try:

            cv_image = cv2.imread(
                image_path
            )

            if cv_image is not None:

                gray_cv = cv2.cvtColor(
                    cv_image,
                    cv2.COLOR_BGR2GRAY
                )

                _, threshold = (
                    cv2.threshold(
                        gray_cv,
                        150,
                        255,
                        cv2.THRESH_BINARY
                    )
                )

                threshold_pil = (
                    Image.fromarray(
                        threshold
                    )
                )

                text = (
                    pytesseract.image_to_string(
                        threshold_pil,
                        lang="eng+chi_sim+vie+por"
                    )
                )

                results.append(
                    (
                        "THRESHOLD",
                        text
                    )
                )

        except Exception as e:

            print(
                "⚠️ OCR THRESHOLD ERROR:",
                e
            )


    except Exception as e:

        print(
            "❌ OCR IMAGE ERROR:",
            e
        )


    return results


# =========================================================
# 9. QR Code 检测
# =========================================================

def detect_qr_code(
    image_path
):

    try:

        image = cv2.imread(
            image_path
        )

        if image is None:

            return False


        detector = cv2.QRCodeDetector()


        # -------------------------------------------------
        # 普通检测
        # -------------------------------------------------

        try:

            data, points, _ = (
                detector.detectAndDecode(
                    image
                )
            )

            if points is not None:

                if data:

                    print(
                        "🚫 QR CODE FOUND:",
                        data
                    )

                else:

                    print(
                        "🚫 QR CODE FOUND"
                    )

                return True

        except Exception:
            pass


        # -------------------------------------------------
        # 灰度检测
        # -------------------------------------------------

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

            if points is not None:

                if data:

                    print(
                        "🚫 QR CODE FOUND:",
                        data
                    )

                else:

                    print(
                        "🚫 QR CODE FOUND"
                    )

                return True

        except Exception:
            pass


        # -------------------------------------------------
        # 多二维码检测
        # -------------------------------------------------

        try:

            result = (
                detector.detectAndDecodeMulti(
                    image
                )
            )

            if result:

                retval = result[0]

                if retval:

                    print(
                        "🚫 MULTIPLE QR CODE FOUND"
                    )

                    return True

        except Exception:
            pass


    except Exception as e:

        print(
            "⚠️ QR DETECTION ERROR:",
            e
        )


    return False


# =========================================================
# 10. 图片联系方式检测
# =========================================================

def detect_image_contact(
    image_path
):

    # -----------------------------------------------------
    # 第一层：二维码
    # -----------------------------------------------------

    if detect_qr_code(
        image_path
    ):

        return {
            "found": True,
            "reason": "QR_CODE"
        }


    # -----------------------------------------------------
    # 第二层：OCR
    # -----------------------------------------------------

    ocr_results = (
        ocr_image_variants(
            image_path
        )
    )

    detections = {}


    for pass_name, text in ocr_results:

        if not text:
            continue

        for contact_type, pattern in CONTACT_PATTERNS:

            matches = pattern.findall(
                text
            )

            if matches:

                if contact_type not in detections:

                    detections[
                        contact_type
                    ] = []

                detections[
                    contact_type
                ].append({

                    "pass": pass_name,

                    "matches": matches

                })


    # -----------------------------------------------------
    # 强联系方式
    # -----------------------------------------------------

    for contact_type in STRONG_CONTACT_TYPES:

        hits = detections.get(
            contact_type,
            []
        )

        if not hits:
            continue


        print("")
        print(
            "🚫 IMAGE CONTACT FOUND"
        )

        print(
            "PATTERN:",
            contact_type
        )


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
    # 手机号码
    # 至少两个 OCR 模式确认
    # -----------------------------------------------------

    for contact_type in PHONE_CONTACT_TYPES:

        hits = detections.get(
            contact_type,
            []
        )

        if not hits:
            continue


        pass_names = set(
            item["pass"]
            for item in hits
        )


        # -------------------------------------------------
        # 确认
        # -------------------------------------------------

        if len(pass_names) >= 2:

            print("")
            print(
                "🚫 IMAGE PHONE CONTACT CONFIRMED"
            )

            print(
                "PATTERN:",
                contact_type
            )

            print(
                "OCR PASS COUNT:",
                len(pass_names)
            )

            print(
                "OCR PASSES:",
                ", ".join(
                    sorted(pass_names)
                )
            )


            for item in hits:

                print(
                    "MATCH:",
                    item["matches"]
                )


            return {
                "found": True,
                "reason": contact_type
            }


        # -------------------------------------------------
        # 只有一次
        # 忽略 OCR 误判
        # -------------------------------------------------

        print("")
        print(
            "⚠️ PHONE OCR HIT BUT NOT CONFIRMED"
        )

        print(
            "PATTERN:",
            contact_type
        )


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
            ", ".join(
                sorted(pass_names)
            )
        )

        print(
            "ACTION: IGNORE FALSE POSITIVE"
        )


    print("")
    print(
        "✅ IMAGE OCR: NO CONTACT FOUND"
    )


    return {
        "found": False,
        "reason": None
    }


# =========================================================
# 11. 获取水印字体
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
# 12. 添加水印
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


        # 根据图片尺寸自动调整
        font_size = max(
            24,
            int(
                min(width, height)
                * 0.045
            )
        )


        font = get_font(
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
            font=font,
            stroke_width=2
        )


        text_width = (
            bbox[2]
            - bbox[0]
        )

        text_height = (
            bbox[3]
            - bbox[1]
        )


        margin = max(
            15,
            int(
                min(width, height)
                * 0.02
            )
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


        # 黑色描边
        # 白色文字
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


        result.convert(
            "RGB"
        ).save(
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
# 13. Bot 发送文字
# =========================================================

async def bot_send_text(
    bot,
    target,
    text
):

    if not text:

        return False


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

        print("")
        print(
            "❌ BOT TEXT ERROR"
        )

        print(
            "ERROR:",
            e
        )

        return False


    except Exception as e:

        print("")
        print(
            "❌ TEXT SEND ERROR"
        )

        print(
            "ERROR:",
            e
        )

        return False


# =========================================================
# 14. Bot 发送图片
# =========================================================

async def bot_send_photo(
    bot,
    target,
    image_path,
    caption=None
):

    try:

        caption = (
            caption
            or ""
        )


        # Telegram caption 限制
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

            # 图片
            with open(
                image_path,
                "rb"
            ) as photo:

                await bot.send_photo(
                    chat_id=target,
                    photo=photo
                )


            # 长文字
            await bot_send_text(
                bot,
                target,
                caption
            )


        return True


    except TelegramError as e:

        print("")
        print(
            "❌ BOT IMAGE ERROR"
        )

        print(
            "ERROR:",
            e
        )

        return False


    except Exception as e:

        print("")
        print(
            "❌ IMAGE SEND ERROR"
        )

        print(
            "ERROR:",
            e
        )

        return False


# =========================================================
# 15. 处理单条消息
# =========================================================

async def process_message(
    client,
    bot,
    target,
    source_name,
    message,
    processed
):

    # -----------------------------------------------------
    # 唯一 ID
    # -----------------------------------------------------

    message_key = (
        f"{source_name}:{message.id}"
    )


    # -----------------------------------------------------
    # 已处理
    # -----------------------------------------------------

    if message_key in processed:

        print(
            f"MESSAGE {message.id} "
            f"already processed."
        )

        return


    # -----------------------------------------------------
    # 时间
    # -----------------------------------------------------

    msg_time = to_beijing(
        message.date
    )


    print("")
    print(
        "=" * 70
    )

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


    # -----------------------------------------------------
    # 文本
    # -----------------------------------------------------

    text = (
        message.message
        or ""
    ).strip()


    # -----------------------------------------------------
    # 空消息
    # -----------------------------------------------------

    if (
        not message.media
        and not text
    ):

        print(
            "MESSAGE EMPTY, SKIPPED"
        )

        processed.add(
            message_key
        )

        return


    # =====================================================
    # 文字消息
    # =====================================================

    if not message.photo:

        if text_has_contact(
            text
        ):

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
            target,
            text
        )


        if success:

            print(
                "✅ TEXT PUBLISHED"
            )

            processed.add(
                message_key
            )

        else:

            print(
                "❌ TEXT NOT MARKED AS PROCESSED"
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


        downloaded = (
            await client.download_media(
                message,
                file=original_image
            )
        )


        if not downloaded:

            print(
                "❌ IMAGE DOWNLOAD FAILED"
            )

            return


        if not os.path.exists(
            original_image
        ):

            print(
                "❌ IMAGE FILE NOT FOUND"
            )

            return


        # -------------------------------------------------
        # Caption 联系方式
        # -------------------------------------------------

        if text_has_contact(
            text
        ):

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


        if contact_result[
            "found"
        ]:

            print(
                "🚫 IMAGE SKIPPED"
            )

            print(
                "REASON:",
                contact_result[
                    "reason"
                ]
            )

            processed.add(
                message_key
            )

            return


        # -------------------------------------------------
        # 水印
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
            target,
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

            print(
                "MESSAGE WILL BE RETRIED NEXT RUN"
            )


    except TelegramError as e:

        print("")
        print(
            "❌ TELEGRAM ERROR:"
        )

        print(
            e
        )


    except Exception as e:

        print("")
        print(
            "❌ IMAGE PROCESS ERROR:"
        )

        print(
            e
        )


    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# =========================================================
# 16. 主程序
# =========================================================

async def main():

    print("")
    print(
        "=" * 70
    )

    print(
        "TELEGRAM AUTO FORWARDER"
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # 基础检查
    # -----------------------------------------------------

    if not SOURCE_CHANNELS:

        raise ValueError(
            "SOURCE_CHANNELS 不能为空"
        )


    if not TARGET_CHANNEL:

        raise ValueError(
            "TARGET_CHANNEL 不能为空"
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
        "TARGET CHAT ID:",
        TARGET_CHAT_ID
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


    # =====================================================
    # Telegram 普通账号
    # =====================================================

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


    # =====================================================
    # Bot
    # =====================================================

    bot = Bot(
        token=BOT_TOKEN
    )


    try:

        bot_info = await bot.get_me()


        print(
            "BOT:",
            bot_info.username
        )


        # =================================================
        # 普通账号解析目标频道
        # =================================================

        print("")
        print(
            "RESOLVING TARGET CHANNEL..."
        )


        try:

            target_entity = (
                await client.get_entity(
                    TARGET_CHAT_ID
                )
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


            print(
                "TARGET USERNAME:",
                getattr(
                    target_entity,
                    "username",
                    ""
                )
            )


        except Exception as e:

            print("")
            print(
                "❌ TARGET CHANNEL RESOLVE ERROR"
            )

            print(
                "ERROR:",
                e
            )

            print("")
            print(
                "请检查 TARGET_CHANNEL："
            )

            print(
                "公开频道：填写 @频道用户名"
            )

            print(
                "私有频道：填写 -100xxxxxxxxxx"
            )

            return


        # =================================================
        # 这里不再使用 bot.get_chat()
        #
        # 直接进入发布流程
        #
        # 如果 Bot 无权限，
        # send_message/send_photo 会给出真实错误
        # =================================================

        print("")
        print(
            "ℹ️ BOT TARGET PRE-CHECK SKIPPED"
        )

        print(
            "程序将在实际发布时验证 Bot 权限"
        )


        # =================================================
        # 收集所有源频道当天消息
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


            # -------------------------------------------------
            # 获取源频道
            # -------------------------------------------------

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

                print("")
                print(
                    "❌ SOURCE CHANNEL ERROR:",
                    source_name
                )

                print(
                    "ERROR:",
                    e
                )

                continue


            count = 0


            # -------------------------------------------------
            # Telegram iter_messages：
            #
            # 默认：
            # 最新 → 最旧
            #
            # 我们先收集，
            # 再统一排序成：
            #
            # 最旧 → 最新
            # -------------------------------------------------

            async for message in client.iter_messages(
                source_entity,
                limit=SCAN_LIMIT
            ):

                if is_today(
                    message
                ):

                    all_messages.append({

                        "source_name":
                            source_name,

                        "source_index":
                            source_index,

                        "message":
                            message

                    })

                    count += 1


                else:

                    # 因为 Telegram 是
                    # 最新 → 最旧
                    #
                    # 一旦早于今天，
                    # 后面的都不可能是今天

                    if message.date:

                        message_time = (
                            to_beijing(
                                message.date
                            )
                        )

                        if (
                            message_time.date()
                            < get_today()
                        ):

                            break


            print(
                "TODAY MESSAGES:",
                count
            )


        # =================================================
        # 统计
        # =================================================

        print("")
        print(
            "=" * 70
        )

        print(
            "TOTAL TODAY MESSAGES:",
            len(all_messages)
        )


        # =================================================
        # 全局排序
        #
        # 第一：
        # 发布时间
        #
        # 第二：
        # SOURCE_CHANNELS 中的顺序
        #
        # 第三：
        # Telegram message.id
        # =================================================

        def sort_key(item):

            message = item[
                "message"
            ]

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
            "=" * 70
        )

        print(
            "FINAL PUBLISH ORDER:"
        )


        for index, item in enumerate(
            all_messages,
            start=1
        ):

            message = item[
                "message"
            ]


            print(
                f"{index}. "
                f"{item['source_name']} "
                f"MSG={message.id} "
                f"TIME={to_beijing(message.date)}"
            )


        # =================================================
        # 严格顺序发布
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
                target=TARGET_CHAT_ID,
                source_name=source_name,
                message=message,
                processed=processed
            )


            # -------------------------------------------------
            # 每条立即保存
            # -------------------------------------------------

            save_processed(
                processed
            )


            # -------------------------------------------------
            # 间隔 1 秒
            # -------------------------------------------------

            await asyncio.sleep(
                1
            )


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
            "TODAY MESSAGES:",
            len(all_messages)
        )

        print(
            "PROCESSED COUNT:",
            len(processed)
        )

        print(
            "=" * 70
        )


    finally:

        try:

            await client.disconnect()

        except Exception:
            pass


        try:

            await bot.shutdown()

        except Exception:
            pass


# =========================================================
# 17. 启动
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

        print("")
        print(
            "❌ FATAL ERROR:"
        )

        print(
            e
        )

        raise
