"""Read text lines from the ESP32 receiver's USB serial device (POSIX incl. QNX). Stdlib only.

Baud is set with termios when available: Linux needs the B921600 constant; QNX and macOS accept the
numeric rate. If termios cannot set it, the port is still read as-is and `baud_note` says so; then
set it once with stty (QNX: `stty baud=921600 < /dev/serusb1`).
"""
import os
import select
import time


class SerialSource:
    def __init__(self, device, baud=921600):
        self.device, self.baud = device, baud
        self.fd = None
        self.buffer = b''
        self.baud_note = 'not configured'
        self.bytes_read = 0                          # total bytes received (garbled-stream detection)

    def open(self):
        self.fd = os.open(self.device, os.O_RDONLY | getattr(os, 'O_NOCTTY', 0) | os.O_NONBLOCK)
        try:
            import termios
            attrs = termios.tcgetattr(self.fd)
            iflag, oflag, cflag, lflag, _, _, cc = attrs
            # raw 8N1, receiver enabled, no modem control, no echo/canonical processing
            iflag = 0
            oflag = 0
            cflag = (cflag & ~getattr(termios, 'CSIZE', 0) & ~getattr(termios, 'PARENB', 0)
                     & ~getattr(termios, 'CSTOPB', 0) & ~getattr(termios, 'HUPCL', 0)   # no DTR drop -> no ESP32 reset on close
                     ) | termios.CS8 | termios.CREAD | getattr(termios, 'CLOCAL', 0)
            lflag = 0
            speed = getattr(termios, 'B%d' % self.baud, self.baud)
            termios.tcsetattr(self.fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, speed, speed, cc])
            self.baud_note = 'termios raw 8N1 @ %d' % self.baud
        except Exception as exc:                     # not a tty, or baud unsupported by this API
            self.baud_note = 'termios not applied (%s); set baud with stty if lines look garbled' % exc
        return self

    def readline(self, timeout=1.0):
        """Next complete line as str (without newline), or None if no complete line arrived within timeout.
        The deadline also covers garbled input with no newlines (e.g. wrong baud), so callers keep running."""
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            left = deadline - time.monotonic()
            ready, _, _ = select.select([self.fd], [], [], max(left, 0)) if left > 0 else ([], [], [])
            if not ready:
                return None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                raise OSError('Serial device closed')
            self.buffer += chunk; self.bytes_read += len(chunk)
            if len(self.buffer) > 1 << 20:           # garbage without newlines: drop it
                self.buffer = self.buffer[-4096:]
        line, _, self.buffer = self.buffer.partition(b'\n')
        return line.decode('utf-8', 'replace').strip()

    def close(self):
        fd, self.fd = self.fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:                          # already gone (e.g. USB device vanished)
                pass
