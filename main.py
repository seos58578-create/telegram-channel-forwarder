if message.photo:

    image_path = None

    try:

        temp_dir = tempfile.mkdtemp()

        print(
            "Downloading image:",
            key
        )

        image_path = await client.download_media(
            message,
            file=temp_dir
        )

        if not image_path:

            print(
                "IMAGE DOWNLOAD FAILED:",
                key
            )

            return False

        # ==========================================
        # OCR 检测图片中的联系方式
        # ==========================================

        print(
            "Checking image for contact information..."
        )

        image_has_contact = image_contains_contact(
            image_path
        )

        if image_has_contact:

            print(
                "SKIP IMAGE: CONTACT DETECTED",
                key
            )

            processed.setdefault(
                "messages",
                []
            ).append(key)

            save_processed(
                processed
            )

            return False

        # ==========================================
        # 没有联系方式
        # 转发图片
        # ==========================================

        print(
            "Forwarding image:",
            key
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
            "FORWARDED IMAGE:",
            key
        )

    except Exception as e:

        print(
            "IMAGE ERROR:",
            key,
            e
        )

        return False

    finally:

        if image_path:

            try:

                os.remove(
                    image_path
                )

            except Exception:

                pass
