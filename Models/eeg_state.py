import threading
from collections import deque
from config import RAW_BUFFER_SIZE

# =========================================================
# EEG STATE
# =========================================================
class EEGState:
    def __init__(self) -> None:
        self.lock = threading.Lock()

        self.raw_buffer = deque([0] * RAW_BUFFER_SIZE, maxlen=RAW_BUFFER_SIZE)

        self.connected: bool = False
        self.last_error: str = ""

        self.poor_signal: Optional[int] = None
        self.attention: Optional[int] = None
        self.meditation: Optional[int] = None
        self.blink: Optional[int] = None

        self.delta: Optional[int] = None
        self.theta: Optional[int] = None
        self.low_alpha: Optional[int] = None
        self.high_alpha: Optional[int] = None
        self.low_beta: Optional[int] = None
        self.high_beta: Optional[int] = None
        self.low_gamma: Optional[int] = None
        self.mid_gamma: Optional[int] = None
