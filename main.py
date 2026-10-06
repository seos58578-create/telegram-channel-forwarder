import os
import re
import json
import asyncio
import tempfile
import shutil

from datetime import datetime, timezone, timedelta

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract
import cv2

from telethon import TelegramClient
from telethon.sessions import StringSession


# ============================================================
# 基础配置
# ============================================================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
SESSION = os.environ["TELEGRAM_SESSION"]

# 多个源频道：
# SOURCE_CHANNELS=channel1,channel2,channel3
SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ.get(
        "SOURCE_CHANNELS",
        ""
    ).split(",")
    if x.strip()
]

# 自己的目标频道
# TARGET_CHANNEL=mychannel
TARGET_CHANNEL = os.environ.get(
    "TARGET_CHANNEL",
    ""
).strip()

# 北京时间 UTC+8
BEIJING_TZ = timezone(
    timedelta(hours=8)
)

# 每个频道最多读取多少条
SCAN_LIMIT = int(
    os.environ.get(
        "SCAN_LIMIT",
        "100"
    )
)

# processed.json
PROCESSED_FILE = "processed.json"


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


# ============================================================
# processed.json
# ============================================================

def load_processed():

    if not os.path.exists(
        PROCESSED_FILE
    ):

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

        if not isinstance(
            data,
            dict
        ):

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

    # 去重
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


processed = load_processed()


# ============================================================
# 联系方式检测
# ============================================================

CONTACT_PATTERNS = [

    # ========================================================
    # Telegram
    # ========================================================

    # Telegram 链接
    r"(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me)/[A-Za-z0-9_+/=?-]+",

    # Telegram 用户名
    r"@[A-Za-z][A-Za-z0-9_]{4,31}",

    # ========================================================
    # WhatsApp
    # ========================================================

    r"(?:https?://)?(?:www\.)?wa\.me/\d+",
    r"\bwhatsapp\b",

    # ========================================================
    # 微信
    # ========================================================

    r"微信",
    r"\bwechat\b",

    # ========================================================
    # 联系方式关键词
    # ========================================================

    r"客服",
    r"联系我",
    r"加我",
    r"私聊",
    r"添加好友",
    r"扫码联系",
    r"二维码",
    r"联系方式",

    # ========================================================
    # 英文联系方式
    # ========================================================

    r"\bcontact\s*(?:me|us)?\b",
    r"\bcustomer\s*service\b",
    r"\btelegram\b",

    # ========================================================
    # 中国大陆手机号
    # 例如：13812345678
    # ========================================================

    r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)",

    # ========================================================
    # 越南手机号
    # 例如：0912345678
    # +84912345678
    # ========================================================

    r"(?<!\d)(?:\+?84[- ]?)?(?:0?3|0?5|0?7|0?8|0?9)\d{8}(?!\d)",

    # ========================================================
    # 香港电话号码
    # 例如：61234567
    # +852 6123 4567
    # ========================================================

    r"(?<!\d)(?:\+?852[- ]?)?[2-9]\d{3}[- ]?\d{4}(?!\d)",
]


# ============================================================
# 文字联系方式检测
# ============================================================

def contains_contact(
    text,
    print_result=True
):

    if not text:

        return False

    text = str(text)

    for pattern in CONTACT_PATTERNS:

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

    return False


# ============================================================
# 二维码检测
# ============================================================

def image_contains_qrcode(
    image_path
):

    try:

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                "QR CHECK: image load failed"
            )

            return False


        detector = cv2.QRCodeDetector()


        # 原图检测
        data, points, _ = (
            detector.detectAndDecode(
                image
            )
        )

        if data:

            print(
                "QR CODE DETECTED:",
                data[:200]
            )

            return True


        # 灰度图检测
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

            print(
                "QR CODE DETECTED:",
                data[:200]
            )

            return True


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
# OCR 单次识别
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
# OCR 图片联系方式
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


        # ====================================================
        # 第一关：二维码
        # ====================================================

        if image_contains_qrcode(
            image_path
        ):

            print(
                "❌ CONTACT CHECK: QR CODE FOUND"
            )

            return True


        # ====================================================
        # 图片放大
        # ====================================================

        if width < 1800:

            ratio = 1800 / width

            image = image.resize(
                (
                    int(width * ratio),
                    int(height * ratio)
                ),
                Image.Resampling.LANCZOS
            )


        # ====================================================
        # OCR 版本 1：原图
        # ====================================================

        ocr_text_1 = run_ocr(
            image
        )

        print(
            "OCR ORIGINAL:"
        )

        print(
            repr(
                ocr_text_1[:1000]
            )
        )


        if contains_contact(
            ocr_text_1
        ):

            print(
                "❌ IMAGE CONTACT FOUND"
            )

            return True


        # ====================================================
        # OCR 版本 2：灰度
        # ====================================================

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
                ocr_text_2[:1000]
            )
        )


        if contains_contact(
            ocr_text_2
        ):

            print(
                "❌ IMAGE CONTACT FOUND"
            )

            return True


        # ====================================================
        # OCR 版本 3：增强对比度
        # ====================================================

        enhanced = ImageEnhance.Contrast(
            gray
        ).enhance(
            2.0
        )

        enhanced = ImageEnhance.Sharpness(
            enhanced
        ).enhance(
            2.0
        )

        ocr_text_3 = run_ocr(
            enhanced
        )

        print(
            "OCR ENHANCED:"
        )

        print(
            repr(
                ocr_text_3[:1000]
            )
        )


        if contains_contact(
            ocr_text_3
        ):

            print(
                "❌ IMAGE CONTACT FOUND"
            )

            return True


        # ====================================================
        # OCR 版本 4：二值化
        # ====================================================

        threshold = enhanced.point(
            lambda p: 255
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
                ocr_text_4[:1000]
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

        # OCR 出错时，为了安全：
        # 不转发这张图片
        return True


# ============================================================
# 北京时间判断
# ============================================================

def is_today(
    message
):

    if not message.date:

        return False

    message_time = (
        message.date
        .astimezone(
            BEIJING_TZ
        )
    )

    today = datetime.now(
        BEIJING_TZ
    ).date()

    return (
        message_time.date()
        == today
    )


# ============================================================
# 记录 processed
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
# 处理单条消息
# ============================================================

async def process_message(
    client,
    message,
    source_name
):

    key = (
        f"{source_name}:{message.id}"
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
            message.date
            .astimezone(
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
        repr(text[:300])
    )


    # ========================================================
    # 重复消息
    # ========================================================

    if key in processed.get(
        "messages",
        []
    ):

        print(
            "SKIP: ALREADY PROCESSED"
        )

        return


    # ========================================================
    # 只处理今天
    # ========================================================

    if not is_today(
        message
    ):

        print(
            "SKIP: NOT TODAY"
        )

        return


    # ========================================================
    # 图片
    # ========================================================

    if message.photo:

        image_path = None

        temp_dir = None

        try:

            temp_dir = tempfile.mkdtemp()

            print(
                "DOWNLOADING IMAGE..."
            )

            image_path = (
                await client.download_media(
                    message,
                    file=temp_dir
                )
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


            # =================================================
            # OCR + QR 检测
            # =================================================

            has_contact = (
                image_contains_contact(
                    image_path
                )
            )


            if has_contact:

                print(
                    "❌ SKIP IMAGE:"
                    " CONTACT INFORMATION DETECTED"
                )

                # 记录为已处理
                mark_processed(
                    key
                )

                return


            # =================================================
            # Caption 联系方式检测
            # =================================================

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


            # =================================================
            # 转发图片
            # =================================================

            print(
                "========================================"
            )

            print(
                "SENDING IMAGE"
            )

            print(
                "TARGET:",
                TARGET_CHANNEL
            )

            print(
                "========================================"
            )


            await client.send_file(
                TARGET_CHANNEL,
                image_path,
                caption=(
                    text
                    if text.strip()
                    else None
                ),
                parse_mode=None
            )


            print(
                "✅ IMAGE FORWARDED:",
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

            # 失败不写 processed
            # 下一次继续尝试


        finally:

            if image_path:

                try:

                    os.remove(
                        image_path
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


        return


    # ========================================================
    # 纯文字消息
    # ========================================================

    if text.strip():

        # 联系方式检测
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


        try:

            print(
                "========================================"
            )

            print(
                "SENDING TEXT"
            )

            print(
                "TARGET:",
                TARGET_CHANNEL
            )

            print(
                "========================================"
            )


            await client.send_message(
                TARGET_CHANNEL,
                text
            )


            print(
                "✅ TEXT FORWARDED:",
                key
            )


            mark_processed(
                key
            )


        except Exception as e:

            print(
                "❌ TEXT SEND ERROR:",
                repr(e)
            )


        return


    # ========================================================
    # 空消息 / 其他媒体
    # ========================================================

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
        "FINAL VERSION"
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
    # Telegram
    # ========================================================

    client = TelegramClient(
        StringSession(
            SESSION
        ),
        API_ID,
        API_HASH
    )


    print(
        "CONNECTING TELEGRAM..."
    )

    await client.start()

    print(
        "✅ TELEGRAM CONNECTED"
    )


    try:

        # ====================================================
        # 检查目标频道
        # ====================================================

        try:

            target_entity = (
                await client.get_entity(
                    TARGET_CHANNEL
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
                "❌ TARGET CHANNEL ERROR:",
                repr(e)
            )

            return


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


            try:

                entity = (
                    await client.get_entity(
                        source
                    )
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


            # =================================================
            # 读取最近消息
            # =================================================

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
                        (message.text or "")[:100]
                    )
                )


                # 今天
                if is_today(
                    message
                ):

                    today_count += 1

                    before = len(
                        processed.get(
                            "messages",
                            []
                        )
                    )


                    await process_message(
                        client,
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


                else:

                    print(
                        "NOT TODAY - SKIP"
                    )


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

        await client.disconnect()

        print(
            "\n✅ TELEGRAM DISCONNECTED"
        )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
