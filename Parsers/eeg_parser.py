from Models.eeg_state import EEGState
from typing import Optional
import serial
import time
from config import *

# =========================================================
# EEG PARSER
# =========================================================
def read_exact(ser: serial.Serial, n: int) -> bytes:
    data = ser.read(n)
    if len(data) != n:
        raise RuntimeError(f"Expected {n} bytes, got {len(data)}")
    return data


def parse_raw_value(high: int, low: int) -> int:
    value = (high << 8) | low
    if value >= 32768:
        value -= 65536
    return value


def read_packet(ser: serial.Serial) -> Optional[list[int]]:
    while True:
        b = ser.read(1)
        if not b:
            return None

        if b[0] == 0xAA:
            b2 = ser.read(1)
            if not b2:
                return None
            if b2[0] == 0xAA:
                break

    payload_len_b = ser.read(1)
    if not payload_len_b:
        return None
    payload_len = payload_len_b[0]

    payload = read_exact(ser, payload_len)
    checksum = read_exact(ser, 1)[0]

    calc = (~(sum(payload) & 0xFF)) & 0xFF
    if calc != checksum:
        return None

    return list(payload)


def parse_payload(payload: list[int]) -> dict:
    i = 0
    result: dict = {}

    while i < len(payload):
        code = payload[i]
        i += 1

        if code == 0x02:
            if i < len(payload):
                result["poor_signal"] = payload[i]
                i += 1

        elif code == 0x04:
            if i < len(payload):
                result["attention"] = payload[i]
                i += 1

        elif code == 0x05:
            if i < len(payload):
                result["meditation"] = payload[i]
                i += 1

        elif code == 0x16:
            if i < len(payload):
                result["blink"] = payload[i]
                i += 1

        elif code == 0x80:
            if i >= len(payload):
                break
            vlen = payload[i]
            i += 1

            if vlen == 2 and i + 1 < len(payload):
                high = payload[i]
                low = payload[i + 1]
                result["raw"] = parse_raw_value(high, low)

            i += vlen

        elif code == 0x83:
            if i >= len(payload):
                break
            vlen = payload[i]
            i += 1

            if vlen == 24 and i + 23 < len(payload):
                bands = []
                for _ in range(8):
                    val = (payload[i] << 16) | (payload[i + 1] << 8) | payload[i + 2]
                    bands.append(val)
                    i += 3

                result["delta"] = bands[0]
                result["theta"] = bands[1]
                result["low_alpha"] = bands[2]
                result["high_alpha"] = bands[3]
                result["low_beta"] = bands[4]
                result["high_beta"] = bands[5]
                result["low_gamma"] = bands[6]
                result["mid_gamma"] = bands[7]
            else:
                i += vlen

        else:
            if code >= 0x80:
                if i >= len(payload):
                    break
                vlen = payload[i]
                i += 1 + vlen
            else:
                break

    return result


def serial_worker(state: EEGState) -> None:
    while True:
        try:
            with serial.Serial(PORT, BAUD, timeout=2) as ser:
                with state.lock:
                    state.connected = True
                    state.last_error = ""

                while True:
                    payload = read_packet(ser)
                    if payload is None:
                        continue

                    parsed = parse_payload(payload)
                    if not parsed:
                        continue

                    with state.lock:
                        if "raw" in parsed:
                            state.raw_buffer.append(parsed["raw"])

                        if "poor_signal" in parsed:
                            state.poor_signal = parsed["poor_signal"]
                        if "attention" in parsed:
                            state.attention = parsed["attention"]
                        if "meditation" in parsed:
                            state.meditation = parsed["meditation"]
                        if "blink" in parsed:
                            state.blink = parsed["blink"]

                        if "delta" in parsed:
                            state.delta = parsed["delta"]
                        if "theta" in parsed:
                            state.theta = parsed["theta"]
                        if "low_alpha" in parsed:
                            state.low_alpha = parsed["low_alpha"]
                        if "high_alpha" in parsed:
                            state.high_alpha = parsed["high_alpha"]
                        if "low_beta" in parsed:
                            state.low_beta = parsed["low_beta"]
                        if "high_beta" in parsed:
                            state.high_beta = parsed["high_beta"]
                        if "low_gamma" in parsed:
                            state.low_gamma = parsed["low_gamma"]
                        if "mid_gamma" in parsed:
                            state.mid_gamma = parsed["mid_gamma"]

        except Exception as e:
            with state.lock:
                state.connected = False
                state.last_error = f"{type(e).__name__}: {e}"

            time.sleep(2)

