"""Read selected public ZIP members using HTTP ranges, without downloading the archive.

Only metadata and published inferred ancestry results are needed for this audit.
The range response is checked before reading so an unsupported server cannot trigger
an accidental multi-gigabyte download. Each extracted member's CRC is verified.
"""
import argparse
import io
import json
import struct
from pathlib import Path
import urllib.request
import zipfile
import zlib

URL = "https://zenodo.org/api/records/15263706/files/geneticAdmixture-linguisticDiffusion.zip/content"
SIZE = 11597114787


class RangeFile(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.bytes_received = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = SIZE - self.position
        size = min(size, SIZE - self.position)
        if size == 0:
            return b""
        if size > 50_000_000:
            raise ValueError("Refusing a range larger than 50 MB")
        start, end = self.position, self.position + size - 1
        req = urllib.request.Request(
            URL + f"?audit_range={start}-{end}",
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "Study3-feasibility-audit/1.0"},
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                raise RuntimeError("Server did not honor the requested byte range")
            data = response.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Incomplete or oversized range response")
        self.position += size
        self.bytes_received += size
        return data


def inventory(remote, output):
    # This archive contains almost a million cached pairwise genetic files. Its
    # central directory alone is 157 MB. Metadata is at the start and Q matrices
    # at the end, so inspect just those boundaries and explicitly record that scope.
    remote.seek(SIZE - 65536)
    tail = remote.read(65536)
    index = tail.rfind(b"PK\x06\x06")
    end = struct.unpack("<4sQHHIIQQQQ", tail[index:index + 56])
    central_size, central_offset = end[-2:]
    entries = {}
    for start, size in [(central_offset, 500000), (central_offset + central_size - 1000000, 1000000)]:
        remote.seek(start)
        block = remote.read(size)
        offset = block.find(b"PK\x01\x02")
        while offset + 46 <= len(block) and block[offset:offset + 4] == b"PK\x01\x02":
            fields = struct.unpack("<4s6H3I5H2I", block[offset:offset + 46])
            name_length, extra_length, comment_length = fields[10:13]
            stop = offset + 46 + name_length + extra_length + comment_length
            if stop > len(block):
                break
            name = block[offset + 46:offset + 46 + name_length].decode("utf-8")
            info = zipfile.ZipInfo(name)
            info.compress_type, info.CRC = fields[4], fields[7]
            info.compress_size, info.file_size, info.header_offset = fields[8], fields[9], fields[16]
            info.extra = block[offset + 46 + name_length:offset + 46 + name_length + extra_length]
            extra_pos = 0
            while extra_pos + 4 <= len(info.extra):
                tag, length = struct.unpack("<HH", info.extra[extra_pos:extra_pos + 4])
                extra_data = info.extra[extra_pos + 4:extra_pos + 4 + length]
                if tag == 1:
                    values = iter(struct.unpack("<" + "Q" * (len(extra_data) // 8), extra_data))
                    if info.file_size == 0xFFFFFFFF:
                        info.file_size = next(values)
                    if info.compress_size == 0xFFFFFFFF:
                        info.compress_size = next(values)
                    if info.header_offset == 0xFFFFFFFF:
                        info.header_offset = next(values)
                extra_pos += 4 + length
            if not name.startswith("__MACOSX/") and "/F2results/" not in name:
                entries[name] = info
            offset = stop
    summary = {"scope": "first 500 kB and last 1 MB of central directory; excludes F2results and macOS metadata",
               "archive_entries": end[7], "central_directory_size": central_size,
               "members": [{"name": i.filename, "size": i.file_size, "compressed_size": i.compress_size,
                            "header_offset": i.header_offset} for i in entries.values()]}
    (output / "archive_inventory.json").write_text(json.dumps(summary, indent=2))
    return entries


def extract(remote, info):
    remote.seek(info.header_offset)
    header = struct.unpack("<4s5H3I2H", remote.read(30))
    if header[0] != b"PK\x03\x04":
        raise ValueError("Invalid local header")
    name_length, extra_length = header[-2:]
    remote.seek(name_length + extra_length, 1)
    compressed = remote.read(info.compress_size)
    data = zlib.decompress(compressed, -15) if info.compress_type == 8 else compressed
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError("Extracted member failed size or CRC check")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--ancestry-k-range", action="store_true", help="Extract published Q matrices for K=12..30")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    remote = RangeFile()
    entries = inventory(remote, args.output)
    if args.ancestry_k_range:
        args.member.extend(name for name in entries if "/best_runs/" in name
                           and name.endswith(tuple(f"_K{k}.Q" for k in range(12, 31))))
    for member in args.member:
        info = entries[member]
        if info.file_size > 50_000_000 or info.compress_size > 50_000_000:
            raise ValueError("Selected member exceeds the metadata extraction limit")
        target = args.output / member
        if not target.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Unsafe member path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and zlib.crc32(target.read_bytes()) == info.CRC:
            print("Verified existing", member, flush=True)
            continue
        target.write_bytes(extract(remote, info))
        print("Extracted", member, info.file_size, flush=True)
    print("Bytes transferred", remote.bytes_received, flush=True)


if __name__ == "__main__":
    main()
