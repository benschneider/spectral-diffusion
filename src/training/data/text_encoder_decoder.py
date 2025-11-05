import torch
import math

ALLOWED_ASCII_RANGE = range(32, 127)

# --- Header / framing constants ---
MAGIC_BYTES = b'DTE'  # Dense Text Encoder magic bytes (3 bytes)
VERSION_MAJOR = 1
# Header mode flags packed into the low nibble of the version byte.
# Upper 4 bits = major version, lower 4 bits = flags.
FLAG_TINY_HEADER = 0x1  # if set, tiny header layout (6 bytes) is used

# Header layouts:
# Standard (4 pixels = 12 bytes + 1 byte used_ratio = 13 bytes total): 
#   [MAGIC(3)][VER_FLAGS(1)][PROMPT_LEN_U32(4)][ANSWER_LEN_U32(4)][USED_RATIO(1)]
# Tiny (2 pixels = 6 bytes + 1 byte used_ratio = 7 bytes total): 
#   [MAGIC(3)][VER_FLAGS(1)][PROMPT_LEN_U8(1)][ANSWER_LEN_U8(1)][USED_RATIO(1)]
HEADER_SIZE_STANDARD = 12  # original header size without used_ratio
HEADER_SIZE_TINY = 6       # original header size without used_ratio
USED_RATIO_SIZE = 1        # 1 byte used ratio
HEADER_SIZE_STANDARD_TOTAL = HEADER_SIZE_STANDARD + USED_RATIO_SIZE  # 13 bytes
HEADER_SIZE_TINY_TOTAL = HEADER_SIZE_TINY + USED_RATIO_SIZE          # 7 bytes

SEPARATOR = "||SEP||"  # kept for backwards compatibility with older encodings

# Visible Sync Pattern: fixed visible RGB pixels inserted at the start of encoded tensors
# This pattern helps to quickly identify the start of encoded data visually and programmatically.
SYNC_PATTERN_PIXELS = [
    [255, 0, 255],    # Magenta
    [0, 255, 255],    # Cyan
    [255, 255, 0],    # Yellow
]
SYNC_PATTERN_BYTES = bytes([255, 0, 255, 0, 255, 255, 255, 255, 0])  # 3 pixels * 3 bytes

def _make_ver_flags(tiny: bool) -> int:
    """Compose version/flags byte: high nibble = version, low nibble = flags."""
    flags = FLAG_TINY_HEADER if tiny else 0
    return ((VERSION_MAJOR & 0x0F) << 4) | (flags & 0x0F)

def _parse_ver_flags(b: int) -> (int, bool):
    """Return (version_major, tiny_mode) from version/flags byte."""
    ver = (b >> 4) & 0x0F
    tiny = (b & FLAG_TINY_HEADER) != 0
    return ver, tiny

def _int_to_bytes(n: int, length: int) -> bytes:
    return n.to_bytes(length, byteorder='big')

def _bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, byteorder='big')

def _validate_ascii(text: str) -> str:
    """Replace characters outside ASCII 32-126 with '?'."""
    return ''.join(c if ord(c) in ALLOWED_ASCII_RANGE else '?' for c in text)

def _pack_bits_to_bytes(bits: str) -> bytes:
    """Pack a string of bits ('0'/'1') into bytes."""
    # Pad bits to multiple of 8
    pad_len = (8 - len(bits) % 8) % 8
    bits += '0' * pad_len
    b = bytearray()
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        b.append(int(byte, 2))
    return bytes(b)

def _unpack_bytes_to_bits(b: bytes) -> str:
    """Unpack bytes into a string of bits."""
    bits = ''.join(f'{byte:08b}' for byte in b)
    return bits

def _encode_text_to_bits(text: str) -> str:
    """
    Encode ASCII text (32-126) into a bitstring using 7-bit fixed encoding.
    Each character is represented by 7 bits.
    """
    bits = ''.join(f'{ord(c):07b}' for c in text)
    return bits

def _decode_bits_to_text(bits: str, length: int) -> str:
    """
    Decode a bitstring into ASCII text of given length.
    Each character is 7 bits.
    """
    chars = []
    for i in range(length):
        char_bits = bits[i*7:(i+1)*7]
        if len(char_bits) < 7:
            break
        c = chr(int(char_bits, 2))
        chars.append(c)
    return ''.join(chars)

def encode_text_to_image_dense(prompt: str, answer: str, image_size=(32, 32), *, tiny_header: bool = False) -> torch.Tensor:
    """
    Encode prompt and answer strings into a compact normalized RGB tensor.

    The encoding packs 3 ASCII characters (7 bits each) into 3 channels of each pixel (24 bits):
    - 21 bits used for characters (3*7 bits)
    - 3 bits unused per pixel (padded zero)
    A header is prepended with:
    - 3 bytes magic bytes 'DTE'
    - 1 byte version+flags
    - prompt and answer lengths (4 bytes each for standard, 1 byte each for tiny)
    - 1 byte used ratio = (needed_pixels / total_pixels) * 255
    Additionally, a fixed visible RGB sync pattern of 3 pixels is prepended before the header:
    [255, 0, 255], [0, 255, 255], [255, 255, 0].
    This pattern aids in visual identification and decoding synchronization.

    Note: CRC32 is no longer included in the header; only used_ratio remains as metadata.
    """
    # Compose full text and validate
    prompt = _validate_ascii(prompt)
    answer = _validate_ascii(answer)
    full_text = prompt + SEPARATOR + answer

    # Build header bytes
    ver_flags = _make_ver_flags(tiny=tiny_header)
    if tiny_header:
        # 2 pixels (6 bytes) original header + 1 byte used_ratio = 7 bytes total
        if len(prompt) > 255 or len(answer) > 255:
            raise ValueError("tiny_header only supports prompt/answer lengths up to 255 each.")
        header = bytearray()
        header.extend(MAGIC_BYTES)             # 3 bytes
        header.append(ver_flags)               # 1 byte (version+flags, tiny bit set)
        header.append(len(prompt) & 0xFF)      # 1 byte
        header.append(len(answer) & 0xFF)      # 1 byte
        # used_ratio will be appended later after pixel count known
        header_size_bytes = HEADER_SIZE_TINY_TOTAL   # 7
    else:
        # 4 pixels (12 bytes) original header + 1 byte used_ratio = 13 bytes total
        header = bytearray()
        header.extend(MAGIC_BYTES)                 # 3 bytes
        header.append(ver_flags)                   # 1 byte (version+flags)
        header.extend(_int_to_bytes(len(prompt), 4))  # 4 bytes
        header.extend(_int_to_bytes(len(answer), 4))  # 4 bytes
        # used_ratio will be appended later after pixel count known
        header_size_bytes = HEADER_SIZE_STANDARD_TOTAL  # 13

    # Encode header bytes to bits (byte-aligned)
    header_bits = ''.join(f'{b:08b}' for b in header)

    # Encode text payload to 7-bit stream
    text_bits = _encode_text_to_bits(full_text)

    # Combine header then payload
    combined_bits = header_bits + text_bits

    # Pad to full pixels (3 bytes = 24 bits per pixel)
    pad_len = (24 - (len(combined_bits) % 24)) % 24
    combined_bits += '0' * pad_len

    # Compute required pixels
    H, W = image_size
    num_pixels = H * W
    needed_pixels = len(combined_bits) // 24

    if needed_pixels + len(SYNC_PATTERN_PIXELS) > num_pixels:
        raise ValueError(f"Image size too small. Need at least {needed_pixels + len(SYNC_PATTERN_PIXELS)} pixels (including sync pattern) but got {num_pixels}.")

    # Compute used ratio byte: ratio of pixels used excluding sync pattern pixels
    used_ratio_float = needed_pixels / (num_pixels - len(SYNC_PATTERN_PIXELS))
    used_ratio_byte = int(min(used_ratio_float * 255, 255))

    # Append used_ratio byte to header (last byte)
    header.append(used_ratio_byte)

    # Re-encode header bits including used_ratio byte
    header_bits = ''.join(f'{b:08b}' for b in header)

    # Recombine header + text bits and pad again
    combined_bits = header_bits + text_bits
    pad_len = (24 - (len(combined_bits) % 24)) % 24
    combined_bits += '0' * pad_len
    needed_pixels = len(combined_bits) // 24

    # Pack per-pixel bytes (R,G,B) from the bitstream so that the header
    # fully occupies the *first* pixels in order.

    # Start with sync pattern pixels bytes
    pixels = []
    for pix in SYNC_PATTERN_PIXELS:
        pixels.extend(pix)

    # Now pack combined_bits into pixels
    for i in range(needed_pixels):
        start = i * 24
        chunk = combined_bits[start:start+24]
        b0 = int(chunk[0:8], 2)
        b1 = int(chunk[8:16], 2)
        b2 = int(chunk[16:24], 2)
        pixels.extend([b0, b1, b2])

    # Pad trailing pixels with zeros
    remaining_pixels = num_pixels - (needed_pixels + len(SYNC_PATTERN_PIXELS))
    pixels.extend([0] * remaining_pixels * 3)

    # To tensor (3, H, W) in [0,1]
    tensor = torch.tensor(pixels, dtype=torch.float32).reshape(H, W, 3).permute(2, 0, 1) / 255.0
    return tensor

def encode_text_to_image_dense_with_meta(prompt: str, answer: str, image_size=(32, 32), *, tiny_header: bool = False):
    """
    Variant of encode_text_to_image_dense that returns metadata dictionary
    including tiny_header flag and used_ratio.

    Note: CRC32 is not included.

    Returns:
        dict with keys:
            - 'image': encoded torch.Tensor (3, H, W)
            - 'prompt': input prompt string
            - 'answer': input answer string
            - 'tiny_header': bool flag
            - 'used_ratio': float ratio of used pixels (excluding sync pattern)
    """
    # Compose full text and validate
    prompt_valid = _validate_ascii(prompt)
    answer_valid = _validate_ascii(answer)

    # First encode image tensor
    img = encode_text_to_image_dense(prompt_valid, answer_valid, image_size=image_size, tiny_header=tiny_header)

    # Extract used_ratio from encoded image header
    # Decode header bytes from image to get used_ratio byte
    # Flatten image to bytes
    image_bytes = (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).reshape(-1, 3).flatten().tolist()

    # Sync pattern length in bytes
    sync_bytes_len = len(SYNC_PATTERN_BYTES)
    # Header start index after sync pattern
    header_start = sync_bytes_len

    # Determine header size
    # Check ver_flags byte at header_start + 3
    if len(image_bytes) < header_start + 4:
        used_ratio = 0.0
    else:
        ver_flags = image_bytes[header_start + 3]
        _, tiny_mode = _parse_ver_flags(ver_flags)
        header_size = HEADER_SIZE_TINY_TOTAL if tiny_mode else HEADER_SIZE_STANDARD_TOTAL
        if len(image_bytes) < header_start + header_size:
            used_ratio = 0.0
        else:
            used_ratio_byte = image_bytes[header_start + header_size - 1]
            used_ratio = used_ratio_byte / 255.0

    return {
        'image': img,
        'prompt': prompt_valid,
        'answer': answer_valid,
        'tiny_header': tiny_header,
        'used_ratio': used_ratio,
    }

def decode_image_to_text_dense(image: torch.Tensor) -> (str, str):
    """
    Decode a tensor produced by encode_text_to_image_dense back to (prompt, answer).
    Supports both standard and tiny headers with used_ratio.
    CRC validation is no longer performed.
    """
    # Sanitize and flatten to bytes in raster order (R,G,B per pixel)
    image = image.clamp(0, 1)
    bytes_arr = (image * 255).round().to(torch.uint8).permute(1, 2, 0).reshape(-1, 3).flatten().tolist()

    # Check for sync pattern at start
    sync_len_bytes = len(SYNC_PATTERN_BYTES)
    has_sync = False
    if len(bytes_arr) >= sync_len_bytes:
        if bytes(bytes_arr[:sync_len_bytes]) == SYNC_PATTERN_BYTES:
            has_sync = True

    # Determine header start index
    header_start = sync_len_bytes if has_sync else 0

    # We need at least 4 bytes to read magic+ver_flags
    if len(bytes_arr) < header_start + 4:
        raise ValueError("Image too small to contain header.")

    # First 4 bytes: MAGIC (3) + VER_FLAGS (1)
    magic = bytes(bytes_arr[header_start:header_start+3])
    if magic != MAGIC_BYTES:
        raise ValueError("Invalid magic bytes; not a Dense Text Encoder image.")
    ver_flags = bytes_arr[header_start+3]
    version_major, tiny_mode = _parse_ver_flags(ver_flags)
    if version_major != VERSION_MAJOR:
        raise ValueError(f"Unsupported version {version_major} (expected {VERSION_MAJOR}).")

    # Determine full header size with used_ratio if sync present
    if has_sync:
        header_size_bytes = HEADER_SIZE_TINY_TOTAL if tiny_mode else HEADER_SIZE_STANDARD_TOTAL
    else:
        # Backward compatibility: no used_ratio byte
        header_size_bytes = HEADER_SIZE_TINY if tiny_mode else HEADER_SIZE_STANDARD

    # Check length
    if len(bytes_arr) < header_start + header_size_bytes:
        raise ValueError("Image too small to contain full header.")

    # Read prompt and answer lengths
    if tiny_mode:
        prompt_len = bytes_arr[header_start+4]
        answer_len = bytes_arr[header_start+5]
    else:
        prompt_len = _bytes_to_int(bytes(bytes_arr[header_start+4:header_start+8]))
        answer_len = _bytes_to_int(bytes(bytes_arr[header_start+8:header_start+12]))

    # Extract full text bits after header
    total_chars = prompt_len + len(SEPARATOR) + answer_len
    bits = ''.join(f'{b:08b}' for b in bytes_arr[header_start:])
    header_bits_len = header_size_bytes * 8
    text_bits = bits[header_bits_len:]
    needed_bits_len = total_chars * 7
    text_bits = text_bits[:needed_bits_len]
    full_text = _decode_bits_to_text(text_bits, total_chars)

    sep_index = full_text.find(SEPARATOR)
    if sep_index == -1:
        raise ValueError("Separator not found in decoded text.")

    prompt = full_text[:sep_index]
    answer = full_text[sep_index + len(SEPARATOR):]
    return prompt, answer

def decode_image_auto(image: torch.Tensor) -> (str, str):
    """
    Auto-detect and decode Dense Text Encoder image tensor.
    Searches for MAGIC_BYTES within the image byte array (after sync pattern if present),
    and returns (prompt, answer).
    CRC validation is no longer performed.
    """
    # Sanitize and flatten to bytes in raster order (R,G,B per pixel)
    image = image.clamp(0, 1)
    bytes_arr = (image * 255).round().to(torch.uint8).permute(1, 2, 0).reshape(-1, 3).flatten().tolist()

    # Search for MAGIC_BYTES in the byte array
    magic_len = len(MAGIC_BYTES)
    found_index = -1
    for i in range(len(bytes_arr) - magic_len + 1):
        if bytes(bytes_arr[i:i+magic_len]) == MAGIC_BYTES:
            found_index = i
            break

    if found_index == -1:
        raise ValueError("MAGIC_BYTES not found in image data; not a valid Dense Text Encoder image.")

    # Check if sync pattern precedes magic bytes
    # If found_index >= sync pattern length and bytes before match sync pattern, treat as synced
    has_sync = False
    if found_index >= len(SYNC_PATTERN_BYTES):
        if bytes(bytes_arr[found_index - len(SYNC_PATTERN_BYTES):found_index]) == SYNC_PATTERN_BYTES:
            has_sync = True
            header_start = found_index
    else:
        header_start = found_index

    # If no sync pattern, header start is found_index
    if not has_sync:
        header_start = found_index

    # We need at least 4 bytes to read magic+ver_flags
    if len(bytes_arr) < header_start + 4:
        raise ValueError("Image too small to contain header.")

    ver_flags = bytes_arr[header_start + 3]
    version_major, tiny_mode = _parse_ver_flags(ver_flags)
    if version_major != VERSION_MAJOR:
        raise ValueError(f"Unsupported version {version_major} (expected {VERSION_MAJOR}).")

    # Determine header size with used_ratio if sync present
    if has_sync:
        header_size_bytes = HEADER_SIZE_TINY_TOTAL if tiny_mode else HEADER_SIZE_STANDARD_TOTAL
    else:
        header_size_bytes = HEADER_SIZE_TINY if tiny_mode else HEADER_SIZE_STANDARD

    if len(bytes_arr) < header_start + header_size_bytes:
        raise ValueError("Image too small to contain full header.")

    # Read prompt and answer lengths
    if tiny_mode:
        prompt_len = bytes_arr[header_start+4]
        answer_len = bytes_arr[header_start+5]
    else:
        prompt_len = _bytes_to_int(bytes(bytes_arr[header_start+4:header_start+8]))
        answer_len = _bytes_to_int(bytes(bytes_arr[header_start+8:header_start+12]))

    # Decode text payload bits
    total_chars = prompt_len + len(SEPARATOR) + answer_len
    bits = ''.join(f'{b:08b}' for b in bytes_arr[header_start:])
    header_bits_len = header_size_bytes * 8
    text_bits = bits[header_bits_len:]
    needed_bits_len = total_chars * 7
    text_bits = text_bits[:needed_bits_len]
    full_text = _decode_bits_to_text(text_bits, total_chars)

    sep_index = full_text.find(SEPARATOR)
    if sep_index == -1:
        raise ValueError("Separator not found in decoded text.")

    prompt = full_text[:sep_index]
    answer = full_text[sep_index + len(SEPARATOR):]
    return prompt, answer


if __name__ == "__main__":
    prompt = "Prompt: hello"
    answer = "Answer: world"
    for tiny in (False, True):
        mode = "tiny" if tiny else "standard"
        img = encode_text_to_image_dense(prompt, answer, image_size=(32, 32), tiny_header=tiny)
        decoded_prompt, decoded_answer = decode_image_to_text_dense(img)
        print(f"[{mode}] Match Prompt:", prompt == decoded_prompt, "Match Answer:", answer == decoded_answer)
    # Test auto decoder
    for tiny in (False, True):
        mode = "tiny" if tiny else "standard"
        img = encode_text_to_image_dense(prompt, answer, image_size=(32, 32), tiny_header=tiny)
        decoded_prompt, decoded_answer = decode_image_auto(img)
        print(f"[auto {mode}] Match Prompt:", prompt == decoded_prompt, "Match Answer:", answer == decoded_answer)
