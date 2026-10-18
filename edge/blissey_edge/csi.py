"""Parse ESP32 CSI lines into (source MAC, role, 52 LLTF subcarrier amplitudes). Stdlib only.

Accepts either a raw serial line from the AP receiver
    CSI_DATA,AP,<mac>,<rssi>,...(>=25 fields)...,[i0 r0 i1 r1 ...]
or one collector JSON record (measurements.jsonl) with `firmware_fields` and `csi`.
"""
import json
import math

# Physical LLTF subcarriers kept by every model: 6..31 and 33..58 (guard/DC bins dropped).
CHANNELS = tuple(list(range(6, 32)) + list(range(33, 59)))


class PacketError(ValueError):
    pass


def amplitudes(csi):
    """First 128 int8 values = 64 (imag, real) LLTF pairs -> 52 amplitudes."""
    if len(csi) < 128:
        raise PacketError('CSI shorter than the 128-value LLTF block')
    out = []
    for k in CHANNELS:
        a, b = csi[2 * k], csi[2 * k + 1]
        if not (-128 <= a <= 127 and -128 <= b <= 127):
            raise PacketError('CSI values must be int8')
        out.append(math.hypot(a, b))
    return out


def _check(fields):
    if len(fields) < 25 or fields[0] != 'CSI_DATA':
        raise PacketError('Not a CSI_DATA record with >= 25 fields')
    try:
        channel, rx_state = int(fields[16]), int(fields[21])
    except ValueError:                       # serial noise / two lines run together shifts the fields
        raise PacketError('Corrupted CSI header')
    if channel <= 0 or rx_state != 0:
        raise PacketError('Invalid radio channel or RX state')


def parse_serial_line(text):
    """Raw serial line -> (mac, role, amplitudes)."""
    text = text.strip()
    head, sep, array = text.partition('[')
    if not sep or not array.endswith(']'):
        raise PacketError('Missing CSI array')
    fields = [f.strip() for f in head.rstrip(',').split(',')]
    _check(fields)
    try:
        csi = [int(v) for v in array[:-1].replace(',', ' ').split()]
    except ValueError:
        raise PacketError('Non-integer CSI value')
    return fields[2].upper(), fields[1], amplitudes(csi)


def parse_record(record):
    """Collector JSON record (dict or JSON text) -> (elapsed_seconds, mac, role, amplitudes)."""
    if isinstance(record, str):
        record = json.loads(record)
    fields = record['firmware_fields']
    _check(fields)
    t = float(record['elapsed_seconds'])
    if not math.isfinite(t) or t < 0:
        raise PacketError('Invalid elapsed timestamp')
    mac = str(record.get('source_mac', fields[2])).upper()
    return t, mac, str(record.get('role', fields[1])), amplitudes(record['csi'])
