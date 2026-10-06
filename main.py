import os
import re
import asyncio
import tempfile
import shutil
from datetime import datetime, timezone

from PIL import Image, ImageOps
import pytesseract

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import MessageMediaPhoto


# ============================================================
# 1. 读取 GitHub Secrets
# ============================================================

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
TELEGRAM_SESSION = os.environ["TELEGRAM_SESSION"]

SOURCE_CHANNELS = [
    x.strip()
    for x in os.environ.get("SOURCE_CHANNELS", "").split(",")
    if x.strip()
]

TARGET_CHANNEL = os.environ["TARGET_CHANNEL"]


# ============================================================
# 2. Telegram Client
#
# 使用 StringSession
# 不需要每次输入验证码
# ============================================================

client = TelegramClient(
    StringSession(TELEGRAM_SESSION),
    API_ID,
    API_HASH
)


# ============================================================
# 3. 联系方式检测规则
# ============================================================

# 电话号码
PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?\d[\d\s\-.()]{7,}\d)"
    r"(?!\d)",
    re.IGNORECASE
)

# Telegram / t.me
TELEGRAM_LINK_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?"
    r"(?:t\.me|telegram\.me)/"
    r"[A-Za-z0-9_+/-]+",
    re.IGNORECASE
)

# @用户名
USERNAME_PATTERN = re.compile(
    r"@[A-Za-z0-9_]{4,}",
    re.IGNORECASE
)

# WhatsApp
WHATSAPP_PATTERN = re.compile(
    r"\b(?:whatsapp|wa\.me)\b",
    re.IGNORECASE
)

# 微信
WECHAT_PATTERN = re.compile(
    r"\b(?:wechat|weixin)\b|微信",
    re.IGNORECASE
)

# LINE
LINE_PATTERN = re.compile(
    r"\b(?:line|line\.me)\b",
    re.IGNORECASE
)

# 联系方式相关词
CONTACT_KEYWORDS_PATTERN = re.compile(
    r"(telegram|whatsapp|wechat|weixin|微信|"
    r"客服|联系|加我|私聊|扫码|二维码|"
    r"contact|customer\s*service)",
    re.IGNORECASE
)


# ============================================================
# 4. 判断文字是否包含联系方式
# ============================================================

def contains_contact(text: str) -> bool:

    if not text:
        return False

    patterns = [
        PHONE_PATTERN,
        TELEGRAM_LINK_PATTERN,
        USERNAME_PATTERN,
        WHATSAPP_PATTERN,
        WECHAT_PATTERN,
        LINE_PATTERN,
    ]

    for pattern in patterns:
        if pattern.search(text):
            return True

    return False


# ============================================================
# 5. OCR 图片
# ============================================================

def ocr_image(image_path: str) -> str:

    try:

        image = Image.open(image_path)

        # 转 RGB
        image = image.convert("RGB")

        # 图片放大，提高 OCR 准确率
        width, height = image.size

        if width < 1600:

            scale = 1600 / width

            image = image.resize(
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        # 灰度
        image = ImageOps.grayscale(image)

        # OCR
        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        return text

    except Exception as e:

        print("OCR ERROR:")
        print(e)

        return ""


# ============================================================
# 6. 判断图片是否存在联系方式
# ============================================================

def image_contains_contact(image_path: str) -> bool:

    print("开始 OCR 图片...")

    ocr_text = ocr_image(image_path)

    print("--------------------------------")
    print("OCR识别结果:")
    print(ocr_text)
    print("--------------------------------")

    if not ocr_text.strip():

        print("OCR没有识别到文字")

        return False

    if contains_contact(ocr_text):

        print(
            "检测到图片中存在联系方式"
        )

        return True

    print(
        "图片没有检测到联系方式"
    )

    return False


# ============================================================
# 7. 判断消息是不是今天发布
# ============================================================

def is_today(message_date) -> bool:

    if not message_date:
        return False

    now = datetime.now(timezone.utc)

    today_start = now.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    return message_date >= today_start


# ============================================================
# 8. 处理一条消息
# ============================================================

async def process_message(message, source_name):

    if not message:
        return

    print("")
    print("=" * 60)

    print(
        f"来源频道: {source_name}"
    )

    print(
        f"消息ID: {message.id}"
    )

    print(
        f"发布时间: {message.date}"
    )

    print("=" * 60)


    # ========================================================
    # 只处理今天的消息
    # ========================================================

    if not is_today(message.date):

        print(
            "跳过：不是今天发布的消息"
        )

        return


    # ========================================================
    # 获取文字
    # ========================================================

    text = message.message or ""


    # ========================================================
    # 检查文字联系方式
    # ========================================================

    if text.strip():

        print("检查消息文字...")

        if contains_contact(text):

            print(
                "❌ 跳过：文字中检测到联系方式"
            )

            return

        print(
            "✅ 文字检查通过"
        )


    # ========================================================
    # 图片消息
    # ========================================================

    if isinstance(
        message.media,
        MessageMediaPhoto
    ):

        temp_dir = tempfile.mkdtemp()

        image_path = None

        try:

            print(
                "发现图片，开始下载..."
            )

            image_path = await client.download_media(
                message.media,
                file=temp_dir
            )

            if not image_path:

                print(
                    "❌ 图片下载失败"
                )

                return


            print(
                f"图片下载完成: {image_path}"
            )


            # =================================================
            # OCR
            # =================================================

            if image_contains_contact(
                image_path
            ):

                print(
                    "❌ 跳过整条消息："
                    "图片中发现联系方式"
                )

                return


            # =================================================
            # 图片没有联系方式
            # 发送到目标频道
            # =================================================

            print(
                "✅ 图片检查通过"
            )

            print(
                f"正在发送到: {TARGET_CHANNEL}"
            )


            await client.send_file(
                TARGET_CHANNEL,
                image_path,
                caption=text if text.strip() else None
            )


            print(
                "✅ 图片转发成功"
            )


        except Exception as e:

            print(
                "❌ 图片处理失败:"
            )

            print(e)


        finally:

            # 删除临时文件
            try:

                if temp_dir:

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

        try:

            print(
                f"正在发送文字到: {TARGET_CHANNEL}"
            )

            await client.send_message(
                TARGET_CHANNEL,
                text
            )

            print(
                "✅ 文字转发成功"
            )

        except Exception as e:

            print(
                "❌ 文字转发失败:"
            )

            print(e)

        return


    # ========================================================
    # 其他类型消息
    # ========================================================

    print(
        "跳过：当前版本只处理文字和图片"
    )


# ============================================================
# 9. 扫描来源频道
# ============================================================

async def process_channel(channel):

    print("")
    print("#" * 60)

    print(
        f"开始扫描频道: {channel}"
    )

    print("#" * 60)

    try:

        entity = await client.get_entity(
            channel
        )

    except Exception as e:

        print(
            f"❌ 无法访问频道: {channel}"
        )

        print(e)

        return


    count = 0

    try:

        async for message in client.iter_messages(
            entity,
            limit=200
        ):

            # 因为消息按最新→旧排序
            # 遇到昨天的消息以后，可以停止扫描
            if message.date and not is_today(
                message.date
            ):

                print(
                    "已经扫描到昨天的消息，"
                    "停止当前频道扫描。"
                )

                break


            await process_message(
                message,
                channel
            )

            count += 1


    except Exception as e:

        print(
            f"❌ 扫描频道失败: {channel}"
        )

        print(e)

        return


    print("")
    print(
        f"频道 {channel} 扫描完成"
    )

    print(
        f"本次检查消息数量: {count}"
    )


# ============================================================
# 10. 主程序
# ============================================================

async def main():

    print("")
    print("=" * 60)
    print("Telegram Channel Forwarder")
    print("=" * 60)

    print(
        f"来源频道数量: {len(SOURCE_CHANNELS)}"
    )

    print(
        f"目标频道: {TARGET_CHANNEL}"
    )

    print("=" * 60)


    # ========================================================
    # 连接 Telegram
    # ========================================================

    print(
        "正在连接 Telegram..."
    )

    await client.connect()


    # ========================================================
    # 检查 Session
    # ========================================================

    if not await client.is_user_authorized():

        raise RuntimeError(
            "TELEGRAM_SESSION 无效或已经失效。"
            "请重新生成 Session。"
        )


    print(
        "✅ Telegram 登录成功"
    )


    # ========================================================
    # 获取当前账号
    # ========================================================

    try:

        me = await client.get_me()

        if me:

            print(
                f"登录账号: "
                f"{me.first_name or ''} "
                f"{me.last_name or ''}"
            )

            if me.username:

                print(
                    f"用户名: @{me.username}"
                )

    except Exception as e:

        print(
            "获取账号信息失败:"
        )

        print(e)


    # ========================================================
    # 检查来源频道
    # ========================================================

    if not SOURCE_CHANNELS:

        raise RuntimeError(
            "SOURCE_CHANNELS 没有设置。"
        )


    # ========================================================
    # 开始扫描
    # ========================================================

    for channel in SOURCE_CHANNELS:

        await process_channel(
            channel
        )


    print("")
    print("=" * 60)
    print("全部频道处理完成")
    print("=" * 60)


    # ========================================================
    # 断开 Telegram
    # ========================================================

    await client.disconnect()


# ============================================================
# 11. 启动
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "程序被手动停止"
        )

    except Exception as e:

        print("")
        print("=" * 60)
        print("程序运行失败")
        print("=" * 60)

        print(e)

        raise
