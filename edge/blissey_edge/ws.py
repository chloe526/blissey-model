"""Minimal WebSocket client (RFC 6455) on socket + ssl. Standard library only.

Text messages only, client frames masked, ping -> pong handled internally, certificate verification on.
Plain ws:// is allowed only for loopback test servers.
"""
import base64
import hashlib
import os
import socket
import struct
from urllib.parse import urlsplit

GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
MAX_MESSAGE = 16 * 1024


class WSError(Exception):
    pass


class AuthError(WSError):
    """Server rejected the credential (HTTP 401/403 on upgrade, or close 1008)."""


class Closed(WSError):
    pass


class WebSocket:
    def __init__(self, url, headers=None, timeout=10.0, ca_file=None):
        u = urlsplit(url)
        if u.scheme not in ('ws', 'wss') or u.query or u.fragment or u.username or u.password:
            raise WSError('Use a ws(s):// URL without query, fragment or credentials')
        if u.scheme == 'ws' and u.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise WSError('Plain ws:// is allowed only for loopback test servers; use wss://')
        self.url, self.u, self.headers, self.timeout, self.ca_file = url, u, dict(headers or {}), timeout, ca_file
        self.sock = None
        self.server_date = None

    def connect(self):
        port = self.u.port or (443 if self.u.scheme == 'wss' else 80)
        raw = socket.create_connection((self.u.hostname, port), timeout=self.timeout)
        if self.u.scheme == 'wss':
            import ssl
            ctx = ssl.create_default_context(cafile=self.ca_file)     # verifies certificate + hostname
            raw = ctx.wrap_socket(raw, server_hostname=self.u.hostname)
        key = base64.b64encode(os.urandom(16)).decode()
        host = self.u.hostname + ('' if self.u.port is None else ':%d' % self.u.port)
        lines = ['GET %s HTTP/1.1' % (self.u.path or '/'), 'Host: %s' % host, 'Upgrade: websocket',
                 'Connection: Upgrade', 'Sec-WebSocket-Key: %s' % key, 'Sec-WebSocket-Version: 13']
        lines += ['%s: %s' % kv for kv in self.headers.items()]
        raw.sendall(('\r\n'.join(lines) + '\r\n\r\n').encode())
        head = b''
        while b'\r\n\r\n' not in head:
            chunk = raw.recv(4096)
            if not chunk:
                raw.close(); raise WSError('Connection closed during handshake')
            head += chunk
            if len(head) > 65536:
                raw.close(); raise WSError('Handshake response too large')
        head, _, rest = head.partition(b'\r\n\r\n')
        status_line, *header_lines = head.decode('latin-1').split('\r\n')
        status = int(status_line.split()[1])
        hdr = {k.strip().lower(): v.strip() for k, v in (h.split(':', 1) for h in header_lines if ':' in h)}
        self.server_date = hdr.get('date')
        if status in (401, 403):
            raw.close(); raise AuthError('Upgrade rejected: HTTP %d' % status)
        if status != 101:
            raw.close(); raise WSError('Upgrade failed: HTTP %d' % status)
        expect = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        if hdr.get('sec-websocket-accept') != expect:
            raw.close(); raise WSError('Bad Sec-WebSocket-Accept')
        self.sock, self._buf = raw, rest
        return self

    # ---- frames
    def _send_frame(self, opcode, payload):
        n = len(payload)
        header = bytes([0x80 | opcode])
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 1 << 16:
            header += bytes([0x80 | 126]) + struct.pack('!H', n)
        else:
            header += bytes([0x80 | 127]) + struct.pack('!Q', n)
        mask = os.urandom(4)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + mask + body)

    def send(self, text):
        data = text.encode('utf-8')
        if len(data) > MAX_MESSAGE:
            raise WSError('Message exceeds 16 KiB')
        self._send_frame(0x1, data)

    def _read(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise Closed('Connection closed')
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def recv(self, timeout):
        """Next text message, or None on timeout. Pings are answered; close raises Closed/AuthError."""
        self.sock.settimeout(timeout)
        started = False
        try:
            message, opcode_first = b'', None
            while True:
                b0, b1 = self._read(2)
                started = True                            # from here a timeout would desync the stream
                self.sock.settimeout(self.timeout)       # rest of a started frame must arrive promptly
                fin, opcode, n = b0 & 0x80, b0 & 0x0F, b1 & 0x7F
                if b1 & 0x80:
                    raise WSError('Server frames must not be masked')
                if n == 126:
                    n = struct.unpack('!H', self._read(2))[0]
                elif n == 127:
                    n = struct.unpack('!Q', self._read(8))[0]
                if n > MAX_MESSAGE:
                    raise WSError('Server frame too large')
                payload = self._read(n)
                if opcode in (0x9, 0xA):                 # control frame between messages
                    if opcode == 0x9:
                        self._send_frame(0xA, payload)
                    if not message:
                        started = False
                        self.sock.settimeout(timeout)
                    continue
                if opcode == 0x8:
                    code = struct.unpack('!H', payload[:2])[0] if len(payload) >= 2 else 1005
                    try:
                        self._send_frame(0x8, payload[:2])
                    except OSError:
                        pass
                    self.close()
                    if code == 1008:
                        raise AuthError('Closed by server (1008): %s' % payload[2:].decode('utf-8', 'replace'))
                    raise Closed('Closed by server (%d): %s' % (code, payload[2:].decode('utf-8', 'replace')))
                if opcode in (0x1, 0x2):
                    opcode_first = opcode
                message += payload
                if len(message) > MAX_MESSAGE:
                    raise WSError('Server message too large')
                if fin:
                    if opcode_first != 0x1:
                        raise WSError('Unexpected binary message')
                    return message.decode('utf-8')
        except socket.timeout:
            if started:
                self.close()
                raise WSError('Timed out in the middle of a frame')
            return None

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
