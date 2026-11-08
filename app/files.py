"""Browser ownership checks for Gradio uploads, generated files, and video streams."""
from __future__ import annotations

import json
import secrets
import threading
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import unquote

from starlette.requests import Request
from starlette.responses import Response


COOKIE_NAME = "argavatar_browser"
_active_owner: ContextVar[tuple | None] = ContextVar("argavatar_file_owner", default=None)


def _paths(value):
    """Walk Gradio's nested FileData, Gallery, and component update structures."""
    if isinstance(value, dict):
        for child in value.values():
            yield from _paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _paths(child)
    elif isinstance(value, (str, Path)):
        name = str(value)
        if len(name) > 4096 or "\n" in name or not name:
            return
        if "/file=" in name:
            name = unquote(name.split("/file=", 1)[1].split("?", 1)[0])
        try:
            path = Path(name).resolve()
            if path.is_file():
                yield path
        except (OSError, ValueError):
            return


class SessionFiles:
    def __init__(self, public_root: Path):
        self.public_root = public_root.resolve()
        self.public_files: set[Path] = set()
        self.file_owners: dict[Path, set[str]] = {}
        self.session_owners: dict[str, str] = {}
        self.lock = threading.RLock()

    def register(self, value, owner: str | None = None):
        with self.lock:
            for path in _paths(value):
                if owner is None:
                    self.public_files.add(path)
                else:
                    self.file_owners.setdefault(path, set()).add(owner)

    def allowed(self, path: Path, owner: str) -> bool:
        path = path.resolve()
        with self.lock:
            return self.is_public(path) or owner in self.file_owners.get(path, set())

    def is_public(self, path: Path) -> bool:
        with self.lock:
            return path.is_relative_to(self.public_root) or path in self.public_files

    def bind_session(self, session: str, owner: str) -> bool:
        with self.lock:
            return self.session_owners.setdefault(session, owner) == owner

    def validate_inputs(self, value, owner: str):
        if any(not self.allowed(path, owner) for path in _paths(value)):
            raise PermissionError("This file belongs to another browser session.")


def register_output(path: str | Path) -> None:
    """Register files embedded in custom HTML, outside Gradio's FileData outputs."""
    active = _active_owner.get()
    if active is not None:
        access, owner = active
        access.register(path, owner)


class SessionFileMiddleware:
    """Keep native file and HLS URLs private even when their full path is known.

    The HttpOnly cookie is issued automatically; visitors need no account. Original
    uploads and Gradio's content-addressed cached copies both require ownership.
    """
    def __init__(self, app, access: SessionFiles, public_config=None):
        self.app, self.access = app, access
        # Gradio may copy bundled previews into its own cache before serving them.
        access.register(public_config)
        from gradio.processing_utils import hash_file
        from gradio.utils import get_upload_folder
        from gradio_client.utils import strip_invalid_filename_characters

        cache_root = Path(get_upload_folder()).resolve()
        for path in access.public_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4"}:
                # These cache files are created lazily after the initial config.
                # Allow only the exact hashes of the bundled public media.
                cached = cache_root / hash_file(path) / strip_invalid_filename_characters(path.name)
                access.public_files.add(cached.resolve())

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        owner = request.cookies.get(COOKIE_NAME)
        fresh = not owner or len(owner) != 43
        if fresh:
            owner = secrets.token_urlsafe(32)
            # The same request may create a queued Gradio Request object.
            headers = [(k, v) for k, v in scope["headers"] if k.lower() != b"cookie"]
            cookie = request.headers.get("cookie", "")
            headers.append((b"cookie", f"{cookie}; {COOKIE_NAME}={owner}".lstrip("; ").encode()))
            scope = {**scope, "headers": headers}

        path = scope["path"]
        body = None
        messages = []
        if request.headers.get("content-type", "").startswith("application/json"):
            while True:
                message = await receive()
                messages.append(message)
                if message["type"] != "http.request" or not message.get("more_body"):
                    break
            try:
                body = json.loads(b"".join(m.get("body", b"") for m in messages))
            except (ValueError, UnicodeDecodeError):
                pass  # Gradio reports malformed requests itself.

        async def replay():
            if messages:
                return messages.pop(0)
            return await receive()

        session = request.query_params.get("session_hash")
        if isinstance(body, dict):
            session = body.get("session_hash") or session
        for prefix in ("/gradio_api/stream/", "/gradio_api/heartbeat/"):
            if prefix in path and scope["method"] in ("GET", "HEAD"):
                session = path.split(prefix, 1)[1].split("/", 1)[0]
        forbidden = session and not self.access.bind_session(str(session), owner)
        private_media = "/gradio_api/stream/" in path
        if isinstance(body, dict):
            try:
                self.access.validate_inputs(body.get("data"), owner)
            except PermissionError:
                forbidden = True
        if "/file=" in path:
            try:
                target = Path(path.split("/file=", 1)[1]).resolve()
                forbidden = forbidden or not self.access.allowed(target, owner)
                private_media = not self.access.is_public(target)
            except (OSError, ValueError):
                forbidden = True
        if forbidden:
            return await Response(status_code=404, headers={"Cache-Control": "no-store"})(scope, replay, send)

        uploaded_response = bytearray()
        async def protected_send(message):
            if message["type"] == "http.response.start" and private_media:
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() not in (b"cache-control", b"vary")]
                headers.extend([(b"cache-control", b"private, no-store"), (b"vary", b"Cookie")])
                message = {**message, "headers": headers}
            if message["type"] == "http.response.start" and fresh:
                cookie_response = Response()
                cookie_response.set_cookie(
                    COOKIE_NAME, owner, httponly=True, samesite="lax",
                    secure=scope.get("scheme") == "https", max_age=86400,
                )
                message = {**message, "headers": [*message.get("headers", []),
                    *[(k, v) for k, v in cookie_response.raw_headers if k == b"set-cookie"]]}
            if path.endswith("/upload") and message["type"] == "http.response.body":
                uploaded_response.extend(message.get("body", b""))
                if not message.get("more_body"):
                    try:
                        self.access.register(json.loads(uploaded_response), owner)
                    except (ValueError, UnicodeDecodeError):
                        pass
            await send(message)

        await self.app(scope, replay, protected_send)
