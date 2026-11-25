import gzip
import io
import struct

VERSIONS = {f"o!dm{n}": n for n in range(1, 9)} | {"o!dm7min": 1007, "o!dm8min": 1008}


class Reader:
    def __init__(self, stream: io.BytesIO):
        self.stream = stream

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.stream.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        length = shift = 0
        while True:
            byte = self.stream.read(1)[0]
            length |= (byte & 0x7F) << shift
            if byte < 0x80:
                return self.stream.read(length).decode()
            shift += 7


def osdb_beatmap_ids(data: bytes) -> list[int]:
    reader = Reader(io.BytesIO(data))
    name = reader.string()
    version = VERSIONS[name]
    minimal = name.endswith("min")
    if version >= 7:
        reader = Reader(io.BytesIO(gzip.decompress(reader.stream.read())))
        reader.string()
    reader.unpack("<d")
    reader.string()
    beatmap_ids = []
    for _ in range(reader.unpack("<i")):
        reader.string()
        if version >= 7:
            reader.unpack("<i")
        for _ in range(reader.unpack("<i")):
            beatmap_ids.append(reader.unpack("<i"))
            if version >= 2:
                reader.unpack("<i")
            if not minimal:
                reader.string(), reader.string(), reader.string()
            reader.string()
            if version >= 4:
                reader.string()
            if version >= 8 or (version >= 5 and not minimal):
                reader.unpack("B")
            if version >= 8 or (version >= 6 and not minimal):
                reader.unpack("<d")
        if version >= 3:
            for _ in range(reader.unpack("<i")):
                reader.string()
    return beatmap_ids
