# Heavily inspired, and based on:
# https://github.com/mathpn/py-qoi

import argparse
import os
import sys
from PIL import Image
from dataclasses import dataclass, field

QOI_MAX_PIXELS = 400_000_000

HEADER_SIZE = 14
END_MARKER_SIZE = 8

QOI_MAGIC = ord('q') << 24 | ord('o') << 16 | ord('i') << 8 | ord('f')

QOI_OP_RUN = 0xc0     # 1100 0000
QOI_OP_INDEX = 0x00   # 0000 0000
QOI_OP_RGB = 0xfe     # 1111 1110
QOI_OP_DIFF = 0x40    # 0100 0000
QOI_OP_LUMA = 0x80    # 1000 0000


@dataclass
class Pixel:
    px_bytes: bytearray = field(init=False)

    def __post_init__(self):
        self.px_bytes = bytearray((0, 0, 0, 255))

    def update(self, px_bytes: bytes):
        n_channels = len(px_bytes)

        if n_channels not in (3, 4):
            sys.stderr.write(
                f"ERROR: Invalid number of channels {n_channels}. Value must be 3 for RGB and 4 for RGBA\n")
            exit(1)

        self.px_bytes[0:n_channels] = px_bytes

    def __str__(self) -> str:
        r, g, b, a = self.px_bytes
        return f"Pixel(r={r}, g={g}, b={b}, a={a})"

    @property
    def bytes(self) -> bytes:
        return bytes(self.px_bytes)

    @property
    def hash(self) -> int:
        r, g, b, a = self.px_bytes
        return (r * 3 + g * 5 + b * 7 + a * 11) % 64

    @property
    def red(self) -> int:
        return self.px_bytes[0]

    @property
    def green(self) -> int:
        return self.px_bytes[1]

    @property
    def blue(self) -> int:
        return self.px_bytes[2]

    @property
    def alpha(self) -> int:
        return self.px_bytes[3]


class ByteBuffer:
    """
    Initializes a byte writer with a given size
    and provides methods to write bytes to it
    """

    def __init__(self, size: int):
        self.bytes = bytearray(size)
        self.write_pos = 0

    def write(self, byte: int):
        """
        Writes a byte to the buffer and increments the write position
        """
        self.bytes[self.write_pos] = (byte % 256)
        self.write_pos += 1

    def output(self):
        """
        Returns the bytes written to the buffer up to the write position
        """
        return self.bytes[0:self.write_pos]

    def write_32_bits(self, value: int):
        """
        Writes a 32 bit integer to the buffer
        """
        self.write((0xff000000 & value) >> 0x18)  # 24 bits
        self.write((0x00ff0000 & value) >> 0x10)  # 16 bits
        self.write((0x0000ff00 & value) >> 0x08)  # 8 bits
        self.write((0x000000ff & value) >> 0x00)  # 0 bits


def fileExists(path: str) -> bool:
    return os.path.isfile(path)


def replace_ext(path: str, ext: str) -> str:
    return path[:path.rfind('.')] + ext


def encode(img_bytes: bytes, width: int, height: int, alpha: bool, srgb: bool):
    total_size = width * height

    if total_size > QOI_MAX_PIXELS:
        sys.stderr.write(
            f"ERROR: Image is too large, max pixels is {QOI_MAX_PIXELS}\n")
        exit(1)

    channel = 4 if alpha else 3
    pixel_data = (img_bytes[i:i + channel]
                  for i in range(0, len(img_bytes), channel))

    total_byte_size = HEADER_SIZE + \
        (total_size * (5 if alpha else 4)) + END_MARKER_SIZE

    buf = ByteBuffer(total_byte_size)
    hash_array = [Pixel() for _ in range(64)]

    # Write header
    buf.write_32_bits(QOI_MAGIC)
    buf.write_32_bits(width)
    buf.write_32_bits(height)
    buf.write(4 if alpha else 3)
    buf.write(0 if srgb else 1)

    # Encode pixels
    run = 0
    prev_px_value = Pixel()
    px_value = Pixel()

    for i, px in enumerate(pixel_data):
        prev_px_value.update(px_value.bytes)
        px_value.update(px)

        if px_value == prev_px_value:
            run += 1

            if run == 62 or (i + 1) >= total_size:
                buf.write(QOI_OP_RUN | (run - 1))
                run = 0

            continue

        if run:
            buf.write(QOI_OP_RUN | (run - 1))
            run = 0

        index_pos = px_value.hash
        if hash_array[index_pos] == px_value:
            buf.write(QOI_OP_INDEX | index_pos)
            continue

        hash_array[index_pos].update(px_value.bytes)

        if px_value.alpha != prev_px_value.alpha:
            buf.write(QOI_OP_RGB)
            buf.write(px_value.red)
            buf.write(px_value.green)
            buf.write(px_value.blue)
            buf.write(px_value.alpha)
            continue

        vr = px_value.red - prev_px_value.red
        vg = px_value.green - prev_px_value.green
        vb = px_value.blue - prev_px_value.blue

        vg_r = vr - vg
        vg_b = vb - vg

        if all(-3 < x < 2 for x in (vr, vg, vb)):
            buf.write(QOI_OP_DIFF | (vr + 2) << 4 | (vg + 2) << 2 | (vb + 2))
            continue

        elif all(-9 < x < 8 for x in (vg_r, vg_b)) and -33 < vg < 32:
            buf.write(QOI_OP_LUMA | (vg + 32))
            buf.write((vg_r + 8) << 4 | (vg_b + 8))
            continue

        buf.write(QOI_OP_RGB)
        buf.write(px_value.red)
        buf.write(px_value.green)
        buf.write(px_value.blue)

    # Write end marker
    # 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x01
    for _ in range(7):
        buf.write(0)
    buf.write(1)

    return buf.output()


def encode_img(img: Image.Image, srgb: bool, out_path: str) -> None:
    w, h = img.size

    if img.mode not in ["RGB", "RGBA"]:
        sys.stderr.write(f"ERROR: Unsupported image mode {img.mode}\n")
        exit(1)

    alpha = True if img.mode == "RGBA" else False
    bytes = img.tobytes()

    output = encode(bytes, w, h, alpha, srgb)

    with open(out_path, "wb") as f:
        f.write(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None,
                        help="Path to the file to encode/decode", type=str)
    parser.add_argument("-e", "--encode", help="Encode a file",
                        action="store_true", default=False)
    parser.add_argument("-d", "--decode", help="Decode a file",
                        action="store_true", default=False)

    args = parser.parse_args()

    if args.file_path is None:
        sys.stderr.write("ERROR: No file path provided\n")
        exit(1)

    if not fileExists(args.file_path):
        sys.stderr.write("ERROR: File does not exist\n")
        exit(1)

    if args.encode:
        try:
            img = Image.open(args.file_path)
        except Exception as e:
            sys.stderr.write(
                f"ERROR: Couldn't open the image {args.file_path}, {e}\n")
            exit(1)

        out_path = replace_ext(args.file_path, ".qoi")
        encode_img(img, out_path, out_path)

    if args.decode:
        pass


if __name__ == "__main__":
    main()
