from __future__ import annotations

import os


class AESGCMCipher:
    """Caller owns key generation, secure storage and recovery; key is never saved here."""
    def __init__(self, key: bytes):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if len(key) != 32:
            raise ValueError("Supply a 32-byte AES-256 key")
        self._cipher = AESGCM(key)

    def protect(self, data: bytes) -> bytes:
        nonce = os.urandom(12)
        return b"FKAE1" + nonce + self._cipher.encrypt(nonce, data, b"face-id-kit/embedding/v1")

    def unprotect(self, data: bytes) -> bytes:
        if not data.startswith(b"FKAE1") or len(data) < 33:
            raise ValueError("Unknown encrypted embedding format")
        return self._cipher.decrypt(data[5:17], data[17:], b"face-id-kit/embedding/v1")


class DPAPICipher:
    """Windows current-user encryption. Compatible with existing DPAPI1 blobs."""
    def _apply(self, data: bytes, decrypt: bool) -> bytes:
        if os.name != "nt":
            raise RuntimeError("DPAPI requires Windows; supply another cipher on this platform")
        import ctypes
        from ctypes import wintypes
        class Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]
        buffer = ctypes.create_string_buffer(data)
        source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
        output = Blob()
        crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        if decrypt:
            succeeded = crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output))
        else:
            succeeded = crypt32.CryptProtectData(ctypes.byref(source), "face-id-kit", None, None, None, 1, ctypes.byref(output))
        if not succeeded:
            raise ctypes.WinError()
        try:
            return ctypes.string_at(output.pbData, output.cbData)
        finally:
            kernel32.LocalFree(output.pbData)

    def protect(self, data: bytes) -> bytes:
        return b"DPAPI1\x00" + self._apply(data, False)

    def unprotect(self, data: bytes) -> bytes:
        if not data.startswith(b"DPAPI1\x00"):
            raise ValueError("Unknown encrypted embedding format (plaintext is not accepted)")
        return self._apply(data[7:], True)
