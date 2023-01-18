# This file is based on @mathpn's work
# https://github.com/mathpn/py-qoi
import argparse, os, sys
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional

QOI_MAX_PIXELS   = 400_000_000
HEADER_SIZE      = 14
END_MARKER_SIZE  = 8
QOI_OP_RUN       = 0xc0    # 1100 0000
QOI_OP_INDEX     = 0x00    # 0000 0000
QOI_OP_RGB       = 0xfe    # 1111 1110
QOI_OP_DIFF      = 0x40    # 0100 0000
QOI_OP_LUMA      = 0x80    # 1000 0000
QOI_OP_RGBA      = 0xff    # 1111 1111
QOI_MASK_2       = 0xc0    # 1100 0000
QOI_MAGIC        = ord('q') << 24 | ord('o') << 16 | ord('i') << 8 | ord('f')


def panic(msg: str):
    sys.stderr.write(f"ERROR: {msg}\n")
    exit(1)


def replace_ext(path, ext): return path[:path.rfind('.')] + ext


@dataclass
class Pixel:
    px_bytes: bytearray = field(init=False)

    def __post_init__(self):
        self.px_bytes = bytearray((0, 0, 0, 255))

    def update(self, px_bytes: bytes):
        n = len(px_bytes)

        if n not in (3, 4):
            panic(f"Invalid number of channels: {n}. Value must be 3 or 4")

        self.px_bytes[0:n] = px_bytes

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
    padding_len = 8

    def __init__(self, size: int):
        self.bytes = bytearray(size)
        self.pos = 0
        self.max_pos = len(self.bytes) - self.padding_len

    def write(self, byte: int):
        self.bytes[self.pos] = (byte % 256)
        self.pos += 1

    def output(self):
        return self.bytes[0:self.pos]

    def write_32_bits(self, value: int):
        self.write((0xff000000 & value) >> 0x18)  # 24 bits
        self.write((0x00ff0000 & value) >> 0x10)  # 16 bits
        self.write((0x0000ff00 & value) >> 0x08)  # 8 bits
        self.write((0x000000ff & value) >> 0x00)  # 0 bits

    def read_32_bits(self) -> int:
        data = [self.read() for _ in range(4)]
        b1, b2, b3, b4 = data
        return b1 << 24 | b2 << 16 | b3 << 8 | b4

    def set_pos(self, pos: int):
        self.pos = pos

    def read(self) -> Optional[int]:
        if self.pos >= self.max_pos:
            return None

        byte = self.bytes[self.pos]
        self.pos += 1
        return byte


def encode(img_bytes: bytes, width: int, height: int, alpha: bool, srgb: bool):
    total_size = width * height

    if total_size > QOI_MAX_PIXELS:
        panic(f"Image is too large, max pixels is {QOI_MAX_PIXELS}")

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
    for _ in range(7):
        buf.write(0)
    buf.write(1)

    return buf.output()


def encode_img(img: Image.Image, srgb: bool, out_path: str) -> None:
    w, h = img.size

    if img.mode not in ["RGB", "RGBA"]:
        panic(f"Unsupported image mode {img.mode}")

    alpha = True if img.mode == "RGBA" else False
    bytes = img.tobytes()

    output = encode(bytes, w, h, alpha, srgb)

    with open(out_path, "wb") as f:
        f.write(output)


def decode(file_bytes: bytes):
    buf = ByteBuffer(len(file_bytes))
    for byte in file_bytes:
        buf.write(byte)
    buf.set_pos(0)

    header_magic = buf.read_32_bits()
    if header_magic != QOI_MAGIC:
        panic("Not a valid QOI file")

    w = buf.read_32_bits()
    h = buf.read_32_bits()
    channel = buf.read()
    srgb = buf.read()

    hash_array = [Pixel() for _ in range(64)]
    out_size = w * h * channel
    pixel_data = bytearray(out_size)
    px_value = Pixel()

    run = 0

    for i in range(-channel, out_size, channel):
        index_pos = px_value.hash
        hash_array[index_pos].update(px_value.bytes)

        if i >= 0:
            pixel_data[i:i + channel] = px_value.bytes

        if run:
            run -= 1
            continue

        b1 = buf.read()
        if b1 is None:
            break

        if b1 == QOI_OP_RGB:
            new_value = bytes((buf.read() for _ in range(3)))
            px_value.update(new_value)
            continue

        if b1 == QOI_OP_RGBA:
            new_value = bytes((buf.read() for _ in range(4)))
            px_value.update(new_value)
            continue

        if (b1 & QOI_MASK_2) == QOI_OP_INDEX:
            px_value.update(hash_array[b1].bytes)
            continue

        if (b1 & QOI_MASK_2) == QOI_OP_DIFF:
            red = (px_value.red + ((b1 >> 4) & 0x03) - 2) % 256
            green = (px_value.green + ((b1 >> 2) & 0x03) - 2) % 256
            blue = (px_value.blue + (b1 & 0x03) - 2) % 256
            px_value.update(bytes((red, green, blue)))
            continue

        if (b1 & QOI_MASK_2) == QOI_OP_LUMA:
            b2 = buf.read()
            vg = ((b1 & 0x3f) % 256) - 32
            red = (px_value.red + vg - 8 + ((b2 >> 4) & 0x0f)) % 256
            green = (px_value.green + vg) % 256
            blue = (px_value.blue + vg - 8 + (b2 & 0x0f)) % 256
            px_value.update(bytes((red, green, blue)))
            continue

        if (b1 & QOI_MASK_2) == QOI_OP_RUN:
            run = (b1 & 0x3f)

    out = {"w": w, "h": h, "channel": "RGBA" if channel ==
           4 else "RGB", "colorspace": srgb, "bytes": pixel_data}

    return out


def decode_to_img(img_bytes: bytes, out_path: str) -> None:
    out = decode(img_bytes)

    size = (out["w"], out["h"])
    img = Image.frombuffer(out["channel"], size, bytes(out["bytes"]), "raw")
    img.save(out_path, "png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, type=str)
    parser.add_argument("-e", "--encode", action="store_true", default=False)
    parser.add_argument("-d", "--decode", action="store_true", default=False)
    args = parser.parse_args()

    if args.file_path is None:
        panic("No file path provided")

    if not os.path.isfile(args.file_path):
        panic(f"File {args.file_path} doesn't exist")

    if args.encode:
        try:
            img = Image.open(args.file_path)
        except Exception as e:
            panic(f"Couldn't open the image {args.file_path}, {e}")

        out_path = replace_ext(args.file_path, ".qoi")
        encode_img(img, out_path, out_path)

    if args.decode:
        with open(args.file_path, "rb") as f:
            file_bytes = f.read()

        out_path = replace_ext(args.file_path, ".png")
        decode_to_img(file_bytes, out_path)


if __name__ == "__main__":
    main()
