"""Wire protocol between the SO-101 policy server and the robot client.

Both sides import this module, so it only depends on the standard library. A message is

    >I header length | JSON header | binary blobs

The header lists the byte length of each blob in ``"blobs"``; the blobs follow in that order. The client sends
camera frames as JPEG blobs, the server answers with JSON only. No pickle: the server listens on a shared compute
node, and unpickling data from a local port would let any user on that node run code as the server's user.

Messages (client -> server, server reply):

    {"cmd": "hello", "token": str}   -> {"ok": true, "joint_names", "views", "n_obs", "hz", "chunk_len", "prompt"}
    {"cmd": "reset"}                 -> {"ok": true}
    {"cmd": "predict", "joints": [D], "views": {name: n_obs}, "stop_step": int | null} + n_obs JPEGs per view,
        in the order of "views", oldest frame first
                                     -> {"ok": true, "chunk": [[D] x chunk_len], "latency": s}

Any failure is answered with ``{"ok": false, "error": str}``.
"""

import json
import socket
import struct

DEFAULT_PORT = 8766
TOKEN_ENV = "MIMIC_POLICY_TOKEN"

_LEN = struct.Struct(">I")
# Upper bound for one message, well above 5 JPEG frames of 3 views at 640x480.
MAX_MESSAGE_BYTES = 64 * 2**20


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def send(sock: socket.socket, header: dict, blobs: list[bytes] | tuple[bytes, ...] = ()) -> None:
    """Send ``header`` (JSON-serialisable) followed by ``blobs``."""
    raw = json.dumps({**header, "blobs": [len(b) for b in blobs]}).encode()
    sock.sendall(b"".join([_LEN.pack(len(raw)), raw, *blobs]))


def recv(sock: socket.socket) -> tuple[dict, list[bytes]]:
    """Receive one message and return its header and blobs."""
    (n,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if n > MAX_MESSAGE_BYTES:
        raise ValueError(f"header of {n} bytes exceeds the limit")
    header = json.loads(_recv_exact(sock, n))
    sizes = header.pop("blobs", [])
    if not isinstance(sizes, list) or sum(sizes) > MAX_MESSAGE_BYTES:
        raise ValueError("bad blob sizes")
    return header, [_recv_exact(sock, size) for size in sizes]


def call(sock: socket.socket, header: dict, blobs: list[bytes] | tuple[bytes, ...] = ()) -> dict:
    """Send a request and return the reply header; raise ``RuntimeError`` on an error reply."""
    send(sock, header, blobs)
    reply, _ = recv(sock)
    if not reply.get("ok"):
        raise RuntimeError(f"policy server: {reply.get('error')}")
    return reply
