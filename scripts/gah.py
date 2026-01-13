import struct
import gzip
import io


def read_7bit_encoded_int(stream):
    res = 0
    shift = 0
    while True:
        byte = stream.read(1)
        if not byte:
            return None
        b = ord(byte)
        res |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return res


def read_string(stream):
    length = read_7bit_encoded_int(stream)
    if length is None:
        return ""
    return stream.read(length).decode("utf-8")


def read_int32(stream):
    return struct.unpack("<i", stream.read(4))[0]


def read_double(stream):
    return struct.unpack("<d", stream.read(8))[0]


def read_byte(stream):
    return struct.unpack("B", stream.read(1))[0]


def parse_osdb_stream(stream):
    """Parse .osdb format from a binary stream and return beatmap IDs.

    Args:
        stream: A file-like object opened in binary mode

    Returns:
        list: List of beatmap IDs from all collections in the file
    """
    versions_map = {
        "o!dm": 1,
        "o!dm2": 2,
        "o!dm3": 3,
        "o!dm4": 4,
        "o!dm5": 5,
        "o!dm6": 6,
        "o!dm7": 7,
        "o!dm8": 8,
        "o!dm7min": 1007,
        "o!dm8min": 1008,
    }

    version_string = read_string(stream)
    file_version = versions_map.get(version_string, -1)

    if file_version == -1:
        raise ValueError(f"Unrecognized osdb file version: {version_string}")

    if file_version >= 7:
        compressed_data = stream.read()
        decompressed_data = gzip.decompress(compressed_data)
        stream = io.BytesIO(decompressed_data)
        read_string(stream)

    # Metadata
    _date = read_double(stream)
    _last_editor = read_string(stream)
    num_collections = read_int32(stream)

    beatmap_ids = []
    is_minimal = version_string.endswith("min")

    for _ in range(num_collections):
        _collection_name = read_string(stream)
        if file_version >= 7:
            _online_id = read_int32(stream)

        num_beatmaps = read_int32(stream)
        for _ in range(num_beatmaps):
            map_id = read_int32(stream)
            beatmap_ids.append(map_id)

            if file_version >= 2:
                _map_set_id = read_int32(stream)

            if not is_minimal:
                _artist = read_string(stream)
                _title = read_string(stream)
                _diff_name = read_string(stream)

            _md5 = read_string(stream)

            if file_version >= 4:
                _comment = read_string(stream)

            if file_version >= 8 or (file_version >= 5 and not is_minimal):
                _play_mode = read_byte(stream)

            if file_version >= 8 or (file_version >= 6 and not is_minimal):
                _stars = read_double(stream)

        if file_version >= 3:
            num_hashes = read_int32(stream)
            for _ in range(num_hashes):
                _hash = read_string(stream)

    return beatmap_ids


def get_beatmap_ids_from_osdb(file_path):
    """Parse .osdb file from a file path and return beatmap IDs.

    Args:
        file_path: Path to the .osdb file

    Returns:
        list: List of beatmap IDs from all collections in the file
    """
    with open(file_path, "rb") as f:
        return parse_osdb_stream(f)


def get_beatmap_ids_from_osdb_bytes(data):
    """Parse .osdb format from bytes and return beatmap IDs.

    Args:
        data: Bytes containing .osdb file data

    Returns:
        list: List of beatmap IDs from all collections in the file
    """
    stream = io.BytesIO(data)
    return parse_osdb_stream(stream)