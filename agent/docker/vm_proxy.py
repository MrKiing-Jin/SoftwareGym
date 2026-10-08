"""Ensure the host proxy relay is up and configure the OSWorld guest VM to use it.

Background
----------
The host proxy (clash) binds to ``127.0.0.1:7890``, which is unreachable from the
QEMU guest. The guest reaches the host through the docker bridge gateway
``172.17.0.1``. ``proxy_relay.py`` bridges ``172.17.0.1:7890 -> 127.0.0.1:7890``
so the guest can use ``http://172.17.0.1:7890`` as its proxy.

This module provides:

* :func:`ensure_relay` -- idempotently make sure the relay is listening on the
  docker-bridge address (started on demand, tracked via a pidfile).
* :func:`configure_vm_proxy` -- inject the proxy into a *booted, healthy* guest
  through the OSWorld ``/execute`` API: system-wide CLI env (``/etc/environment``)
  and the GNOME desktop proxy (``gsettings``) so GUI apps such as Chrome use it.

It can also be run as a CLI against an already-running VM::

    python -m agent.docker.vm_proxy --api-port 5050 --proxy http://172.17.0.1:7890
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib import request

REPO_ROOT = Path(__file__).resolve().parents[2]
RELAY_SCRIPT = REPO_ROOT / "agent" / "tools" / "proxy_relay.py"
RELAY_PID_FILE = REPO_ROOT / "docker" / "proxy_relay.pid"
RELAY_LOG_FILE = REPO_ROOT / "docker" / "proxy_relay.log"

# Docker bridge gateway == the host as seen from inside the guest.
DEFAULT_PROXY = "http://172.17.0.1:7890"
DEFAULT_RELAY_LISTEN = "172.17.0.1:7890"
DEFAULT_RELAY_TARGET = "127.0.0.1:7890"

# Marker so the /etc/environment block is idempotent (rewritten, never duplicated).
_ENV_BEGIN = "# >>> APPEVAL_PROXY >>>"
_ENV_END = "# <<< APPEVAL_PROXY <<<"

ExecuteFn = Callable[[str, Optional[float]], Dict[str, Any]]


# --------------------------------------------------------------------------- #
# Host-side relay management
# --------------------------------------------------------------------------- #
def _port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def ensure_relay(
    listen: str = DEFAULT_RELAY_LISTEN,
    target: str = DEFAULT_RELAY_TARGET,
    wait_seconds: float = 8.0,
) -> Dict[str, Any]:
    """Make sure the TCP relay is listening on ``listen``; start it if not.

    Idempotent: if the port is already accepting connections nothing happens, so
    repeatedly starting VMs will not spawn duplicate relays.
    """
    host, port_s = listen.rsplit(":", 1)
    port = int(port_s)

    if _port_listening(host, port):
        return {"relay": "already_running", "listen": listen, "target": target}

    RELAY_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(RELAY_LOG_FILE, "ab")
    proc = subprocess.Popen(
        [sys.executable, str(RELAY_SCRIPT), "--listen", listen, "--target", target],
        stdout=log_handle,
        stderr=log_handle,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(REPO_ROOT),
    )
    RELAY_PID_FILE.write_text(str(proc.pid), encoding="utf-8")

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _port_listening(host, port):
            return {"relay": "started", "pid": proc.pid, "listen": listen, "target": target}
        if proc.poll() is not None:
            raise RuntimeError(
                f"proxy relay exited early (code {proc.returncode}); see {RELAY_LOG_FILE}"
            )
        time.sleep(0.3)
    raise RuntimeError(f"proxy relay did not start listening on {listen} within {wait_seconds}s")


# --------------------------------------------------------------------------- #
# Guest-side proxy configuration
# --------------------------------------------------------------------------- #
def _build_guest_script(
    proxy: str,
    scope: str,
    sudo_password: str,
    no_proxy: str,
) -> str:
    """Return a bash script (run inside the guest) that applies the proxy."""
    host_port = proxy.split("://", 1)[-1]
    host = host_port.rsplit(":", 1)[0]
    port = host_port.rsplit(":", 1)[1]
    socks = f"socks5://{host}:{port}"

    env_block = "\n".join(
        [
            _ENV_BEGIN,
            f"http_proxy={proxy}",
            f"https_proxy={proxy}",
            f"ftp_proxy={proxy}",
            f"all_proxy={socks}",
            f"HTTP_PROXY={proxy}",
            f"HTTPS_PROXY={proxy}",
            f"FTP_PROXY={proxy}",
            f"ALL_PROXY={socks}",
            f"no_proxy={no_proxy}",
            f"NO_PROXY={no_proxy}",
            _ENV_END,
        ]
    )

    cli_part = f"""
# --- CLI / system-wide environment ---
SUDO() {{ echo {sudo_password!r} | sudo -S -p '' "$@"; }}
tmp_env=$(mktemp)
# strip any previous APPEVAL_PROXY block, then append a fresh one
if [ -f /etc/environment ]; then
  sed '/{_ENV_BEGIN}/,/{_ENV_END}/d' /etc/environment > "$tmp_env"
else
  : > "$tmp_env"
fi
cat >> "$tmp_env" <<'APPEVAL_ENV_EOF'
{env_block}
APPEVAL_ENV_EOF
SUDO cp "$tmp_env" /etc/environment
rm -f "$tmp_env"
echo "[vm_proxy] /etc/environment updated"
"""

    gui_part = f"""
# --- GNOME desktop proxy (Chrome / GUI apps) ---
export DISPLAY="${{DISPLAY:-:0}}"
export DBUS_SESSION_BUS_ADDRESS="${{DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}}"
gsettings set org.gnome.system.proxy mode 'manual'
gsettings set org.gnome.system.proxy.http  host '{host}'
gsettings set org.gnome.system.proxy.http  port {port}
gsettings set org.gnome.system.proxy.https host '{host}'
gsettings set org.gnome.system.proxy.https port {port}
gsettings set org.gnome.system.proxy.ftp   host '{host}'
gsettings set org.gnome.system.proxy.ftp   port {port}
gsettings set org.gnome.system.proxy.socks host '{host}'
gsettings set org.gnome.system.proxy.socks port {port}
gsettings set org.gnome.system.proxy ignore-hosts "['localhost', '127.0.0.1', '20.20.20.0/24']"
echo "[vm_proxy] gsettings proxy set -> {host}:{port}"
"""

    body = "set -e\n"
    if scope in ("both", "cli"):
        body += cli_part
    if scope in ("both", "gui"):
        body += gui_part
    body += '\necho "__APPEVAL_VM_PROXY_DONE__"\n'
    return body


def configure_vm_proxy(
    execute: ExecuteFn,
    proxy: str = DEFAULT_PROXY,
    scope: str = "both",
    sudo_password: str = "password",
    no_proxy: str = "localhost,127.0.0.1,20.20.20.0/24",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Configure the proxy inside a booted guest via an ``/execute`` callable.

    ``execute(command, timeout)`` must run a shell command in the guest and return
    the OSWorld API response dict. ``scope`` is one of ``both`` / ``cli`` / ``gui``.
    """
    script = _build_guest_script(proxy, scope, sudo_password, no_proxy)
    result = execute(script, timeout)
    text = json.dumps(result, ensure_ascii=False)
    ok = "__APPEVAL_VM_PROXY_DONE__" in text
    return {"configured": ok, "proxy": proxy, "scope": scope, "result": result}


# --------------------------------------------------------------------------- #
# Standalone executor (talk to a running VM's OSWorld API directly)
# --------------------------------------------------------------------------- #
def make_http_executor(api_base: str) -> ExecuteFn:
    def _execute(command: str, timeout: Optional[float]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"command": ["/bin/bash", "-c", command], "shell": False}
        if timeout:
            payload["timeout"] = timeout
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            api_base.rstrip("/") + "/execute",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=(timeout or 30.0) + 5.0) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    return _execute


def apply_to_running_vm(
    api_base: str,
    proxy: str = DEFAULT_PROXY,
    scope: str = "both",
    ensure_relay_first: bool = True,
    sudo_password: str = "password",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if ensure_relay_first:
        target = "127.0.0.1:" + proxy.rsplit(":", 1)[1]
        listen = proxy.split("://", 1)[-1]
        out["relay"] = ensure_relay(listen=listen, target=target)
    out["vm"] = configure_vm_proxy(
        make_http_executor(api_base), proxy=proxy, scope=scope, sudo_password=sudo_password
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure OSWorld guest VM to use the host proxy relay.")
    parser.add_argument("--api-port", type=int, help="OSWorld control API port on 127.0.0.1 (e.g. 5050)")
    parser.add_argument("--api-base", help="Full OSWorld API base URL (overrides --api-port)")
    parser.add_argument("--proxy", default=os.environ.get("OSWORLD_VM_PROXY", DEFAULT_PROXY))
    parser.add_argument("--scope", choices=["both", "cli", "gui"], default="both")
    parser.add_argument("--sudo-password", default=os.environ.get("OSWORLD_CLIENT_PASSWORD", "password"))
    parser.add_argument("--no-relay", action="store_true", help="Do not ensure the host relay is running")
    args = parser.parse_args()

    if args.api_base:
        api_base = args.api_base
    elif args.api_port:
        api_base = f"http://127.0.0.1:{args.api_port}"
    else:
        parser.error("one of --api-port or --api-base is required")

    result = apply_to_running_vm(
        api_base,
        proxy=args.proxy,
        scope=args.scope,
        ensure_relay_first=not args.no_relay,
        sudo_password=args.sudo_password,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
