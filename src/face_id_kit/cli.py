from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from .models import download_models, verify_models


def main():
    parser = argparse.ArgumentParser(prog="face-id-kit")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download-models", "verify-models"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--models", type=Path, required=True)
    demo.add_argument("--database", type=Path, help="Omit for ephemeral in-memory enrollment")
    demo.add_argument("--cipher", choices=["dpapi", "aesgcm"])
    demo.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.command == "download-models":
        download_models(args.directory)
        print("Models verified.")
    elif args.command == "verify-models":
        verify_models(args.directory)
        print("Models verified.")
    else:
        from .matching import FaceIdentifier
        from .opencv import OpenCVBackend
        from .stores import InMemoryStore, SQLiteStore
        from .crypto import AESGCMCipher, DPAPICipher
        from .demo.app import create_app
        import uvicorn
        if args.database:
            if not args.cipher:
                parser.error("--database requires --cipher dpapi or aesgcm")
            if args.cipher == "dpapi":
                if os.name != "nt":
                    parser.error("DPAPI requires Windows")
                cipher = DPAPICipher()
            else:
                try:
                    key = base64.b64decode(os.environ["FACE_ID_KIT_KEY"], validate=True)
                    cipher = AESGCMCipher(key)
                except (KeyError, ValueError):
                    parser.error("Set FACE_ID_KIT_KEY to a base64-encoded 32-byte key")
            store = SQLiteStore(args.database, cipher=cipher)
        else:
            store = InMemoryStore()
        backend = OpenCVBackend(args.models)
        backend.ensure_loaded()
        print(f"Face ID Kit demo: http://127.0.0.1:{args.port} (enrollments {'persisted' if args.database else 'in memory'})")
        uvicorn.run(create_app(FaceIdentifier(backend, store)), host="127.0.0.1",
                    port=args.port, ws_max_size=8_000_000)
