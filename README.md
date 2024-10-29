# qoi-py

A Python image encoder/decoder for the Quite Okay Image Format (QOI), allowing you to encode images to QOI format and decode QOI files back to PNG.

## Features
- Encode images (PNG, JPEG) to QOI format.
- Decode QOI files back to PNG images.

## Requirements
To install the dependencies, use requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage
The main script (qoi.py) supports encoding and decoding. Run it from the command line with either the `--encode` or `--decode` option.

## Encoding an Image to QOI Format
To encode an image (e.g., image.png) to QOI format:

```bash
python src/qoi.py image.png --encode
```

This command outputs image.qoi in the same directory as the input file.

## Decoding a QOI Image to PNG
To decode a QOI file (e.g., image.qoi) back to PNG:

```bash
python src/qoi.py image.qoi --decode
```

This command outputs image.png in the same directory as the input file.

