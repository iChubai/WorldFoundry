"""Image/video/audio decode and SDR conditioning preprocess."""

from __future__ import annotations

import logging
from io import BytesIO

import av
import numpy as np
import torch
from PIL import ExifTags, Image, ImageCms, UnidentifiedImageError

from .color_config import HDRColorSpace
from .exr import load_exr_image_conditioning_hdr
from .range_map import normalize_images
from .resize import resize_and_center_crop

logger = logging.getLogger(__name__)

_ORIENTATION_EXIF_KEY = next(key for key, value in ExifTags.TAGS.items() if value == "Orientation")

_ORIENTATION_TO_ROTATION = {3: 180, 6: 270, 8: 90}

_SRGB_PROFILE = ImageCms.createProfile("sRGB")


def load_image_and_preprocess(
    image_path: str,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    crf: int,
    color_space: HDRColorSpace | None = None,
) -> torch.Tensor:
    """
    Loads an image from a path and preprocesses it for conditioning.
    Note: The image is resized to the nearest multiple of 2 for compatibility with video codecs.
    An ``.exr`` still requires ``color_space`` (from ``--hdr``) and is loaded via the
    HDR conditioning path instead of the SDR decode/normalize path.
    """
    if str(image_path).lower().endswith(".exr"):
        if color_space is None:
            raise ValueError(
                "EXR input requires --hdr {SRGB_LINEAR,ACESCG,ACESCCT} to declare the source colour space."
            )
        return load_exr_image_conditioning_hdr(
            exr_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            color_space=color_space,
        )
    image = decode_image(image_path=image_path)
    image = preprocess(image=image, crf=crf)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image = resize_and_center_crop(image, height, width)
    image = normalize_images(image, device, dtype)
    return image


def decode_image(image_path: str) -> np.ndarray:
    """Load an image as an oriented, sRGB, three-channel uint8 RGB array."""
    try:
        with Image.open(image_path) as source_image:
            image = source_image
            orientation = image.getexif().get(_ORIENTATION_EXIF_KEY)
            if orientation in _ORIENTATION_TO_ROTATION:
                image = image.rotate(_ORIENTATION_TO_ROTATION[orientation], expand=True)

            icc_profile = image.info.get("icc_profile")

            if image.mode == "RGBA":
                image = image.convert("RGB")
            elif image.mode == "LA":
                image = image.convert("L")

            if icc_profile:
                try:
                    source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
                    destination_profile = ImageCms.ImageCmsProfile(_SRGB_PROFILE)
                    image = ImageCms.profileToProfile(
                        image,
                        source_profile,
                        destination_profile,
                        outputMode="RGB",
                    )
                except (ImageCms.PyCMSError, OSError, ValueError) as error:
                    logger.warning("Failed to convert image to sRGB: %s", error)
                    image = image.convert("RGB")
            else:
                image = image.convert("RGB")

            return np.array(image, dtype=np.uint8)
    except UnidentifiedImageError as err:
        raise ValueError(f"Cannot decode image file '{image_path}'.") from err


def encode_single_frame(output_file: str, image_array: np.ndarray, crf: float) -> None:
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
        # Round to nearest multiple of 2 for compatibility with video codecs
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file: str) -> np.array:
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: np.array, crf: int) -> np.array:
    """Re-compress an image at ``crf`` to match the compression the model was trained on.
    ``crf`` is required: the correct value is a property of the model generation, so a
    code-level default here would silently condition on the wrong compression. Reaching this
    with ``crf=None`` means a conditioning skipped resolution -- see
    ``ImageConditioner.resolve_crf``, which every pipeline runs over its image inputs.
    """
    if crf is None:
        raise ValueError(
            "Image conditioning CRF is unresolved (crf=None). Resolve it against the checkpoint "
            "first -- ImageConditioner.resolve_crf(images), or detect_params(checkpoint_path).default_image_crf."
        )
    if crf == 0:
        return image
    if min(image.shape[0], image.shape[1]) < 2:
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array
