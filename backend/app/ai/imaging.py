"""Prepare the user's image for an external provider.

Re-encoding to a fresh JPEG strips ALL metadata (EXIF/XMP/comments), which closes the "prompt injection via
image metadata" path, and bounds the payload size regardless of what was uploaded.
"""
import io

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.errors import AppError


def prepare_for_provider(data: bytes, max_side: int, max_pixels: int) -> tuple[bytes, str]:
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise AppError(422, "invalid_image", "We couldn't read that image. It may be corrupt or incomplete.") from exc
    if im.width * im.height > max_pixels:
        raise AppError(413, "image_too_large", "That image has too many pixels. Please use a smaller photo.")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=88, optimize=True)       # no exif= argument: metadata is not carried over
    return out.getvalue(), "image/jpeg"
