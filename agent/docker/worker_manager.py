from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import io
import fcntl
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_IMAGE = REPO_ROOT / "docker" / "base" / "Ubuntu_osworld_minimal.qcow2"
DEFAULT_DOCKER_IMAGE = "happysixd/osworld-docker:latest"
DEFAULT_OVERLAY_DIR = REPO_ROOT / "docker" / "overlays"
PORT_ALLOCATION_LOCK = REPO_ROOT / "docker" / "port-allocation.lock"
# Four independent host-port bases. Allocation advances all four by the same
# offset and keeps scanning until TCP's upper bound instead of relying on a
# fixed-size list. Existing OSWORLD_PORT_GROUPS remains a finite explicit
# override for operators that need a custom allowlist.
DEFAULT_PORT_BASES = (5031, 8031, 8201, 9301)
MAX_TCP_PORT = 65535
DEFAULT_PORT_BIND_RETRIES = 5
_PORT_ALLOCATION_THREAD_LOCK = threading.Lock()


class InfrastructureError(RuntimeError):
    """Raised when OSWorld/Docker cannot provide a trustworthy GUI environment."""


class AppLaunchError(RuntimeError):
    """Raised when the app itself fails to launch inside a PROVEN-healthy container/display.

    This is deliberately NOT a subclass of ``InfrastructureError``: an app that
    cannot start (broken ``release/run.sh`` entry, missing runtime dependency,
    startup crash/traceback, window that never maps) is a REAL software bug, not
    environment noise. Keeping it a sibling of ``InfrastructureError`` forces
    callers to handle it explicitly instead of silently folding it back into the
    infra bucket (where repair rounds are told to ignore it and never fix it).
    """


@dataclass
class WorkerConfig:
    software_type: str
    style_id: str
    worker_name: str
    container_name: str
    api_port: int
    vnc_port: int
    http_port: int
    cdp_port: int
    overlay_path: Path
    base_image: Path = DEFAULT_BASE_IMAGE
    docker_image: str = DEFAULT_DOCKER_IMAGE
    cpu_cores: int = 8
    ram_size: str = "8G"
    health_passes: int = 3
    health_interval_seconds: float = 2.0
    settle_seconds: float = 5.0
    request_timeout_seconds: float = 10.0
    max_restarts: int = 0  # <= 0 means unlimited (no total restart budget)
    unhealthy_threshold: int = 2
    case_batch_size: int = 1
    restart_every_case_batch: bool = True
    reset_overlay_on_recovery: bool = True
    auto_fallback_ports: bool = True

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def novnc_url(self) -> str:
        return f"http://127.0.0.1:{self.vnc_port}/vnc.html"


@dataclass
class RecoveryResult:
    action: str
    healthy: bool
    restarts: int
    message: str = ""


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def config_int(values: Dict[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None or raw == "":
        return env_int(name, default)
    return int(raw)


def config_bool(values: Dict[str, str], name: str, default: bool = False) -> bool:
    raw = values.get(name)
    if raw is None or raw == "":
        return env_bool(name, default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configured_port_bases() -> tuple[int, int, int, int]:
    return (
        env_int("OSWORLD_API_PORT_BASE", DEFAULT_PORT_BASES[0]),
        env_int("OSWORLD_VNC_PORT_BASE", DEFAULT_PORT_BASES[1]),
        env_int("OSWORLD_HTTP_PORT_BASE", DEFAULT_PORT_BASES[2]),
        env_int("OSWORLD_CDP_PORT_BASE", DEFAULT_PORT_BASES[3]),
    )


def port_group_for_offset(
    offset: int,
    bases: Optional[tuple[int, int, int, int]] = None,
) -> tuple[int, int, int, int]:
    if offset < 0:
        raise ValueError("port offset must be non-negative")
    selected_bases = bases or configured_port_bases()
    group = tuple(base + offset for base in selected_bases)
    if any(port < 1 or port > MAX_TCP_PORT for port in group):
        raise ValueError(f"port offset {offset} exceeds TCP port range for bases {selected_bases}")
    return group  # type: ignore[return-value]


def iter_port_groups(
    start_offset: int = 0,
    bases: Optional[tuple[int, int, int, int]] = None,
) -> Iterable[tuple[int, int, int, int]]:
    selected_bases = bases or configured_port_bases()
    max_offset = MAX_TCP_PORT - max(selected_bases)
    for offset in range(max(0, start_offset), max_offset + 1):
        yield port_group_for_offset(offset, selected_bases)


def is_port_free(port: int) -> bool:
    """Check whether Docker can bind a host port on IPv4 and IPv6.

    A connect probe can report a bound-but-not-accepting port as free. Binding a
    temporary socket is a stronger preflight check; Docker remains the final
    authority and start_container retries if an external process wins the race.
    """
    if port < 1 or port > MAX_TCP_PORT:
        return False
    probes = [
        (socket.AF_INET, ("0.0.0.0", port)),
        (socket.AF_INET6, ("::", port)),
    ]
    for family, address in probes:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock.bind(address)
        except OSError:
            return False
    return True


def port_group_free(group: Iterable[int]) -> bool:
    return all(is_port_free(port) for port in group)


@contextmanager
def port_allocation_lock():
    """Serialize allocators in this repository only.

    Other checkouts or users may ignore this lock, so docker-run bind failures
    are still retried with a newly scanned port group.
    """
    with _PORT_ALLOCATION_THREAD_LOCK:
        PORT_ALLOCATION_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with PORT_ALLOCATION_LOCK.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_port_groups(raw: str) -> list[tuple[int, int, int, int]]:
    groups: list[tuple[int, int, int, int]] = []
    for item in raw.split(";"):
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) != 4:
            continue
        try:
            groups.append(tuple(int(part) for part in parts))  # type: ignore[arg-type]
        except ValueError:
            continue
    return groups


def load_worker_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_worker_env(path: Path, config: WorkerConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_worker_env(path)
    path.write_text(
        "\n".join(
            [
                f"SOFTWARE_TYPE={config.software_type}",
                f"STYLE_ID={config.style_id}",
                f"WORKER_NAME={config.worker_name}",
                f"OSWORLD_CONTAINER={config.container_name}",
                f"OSWORLD_API_PORT={config.api_port}",
                f"OSWORLD_VNC_PORT={config.vnc_port}",
                f"OSWORLD_HTTP_PORT={config.http_port}",
                f"OSWORLD_CDP_PORT={config.cdp_port}",
                f"OSWORLD_OVERLAY={config.overlay_path}",
                f"OSWORLD_BASE_IMAGE={config.base_image}",
                f"OSWORLD_DOCKER_IMAGE={config.docker_image}",
                f"OSWORLD_CASE_BATCH_SIZE={config.case_batch_size}",
                f"OSWORLD_RECOVERY_HEALTH_PASSES={config.health_passes}",
                f"OSWORLD_RESET_OVERLAY_ON_RECOVERY={int(config.reset_overlay_on_recovery)}",
                f"OSWORLD_RESTART_EVERY_CASE_BATCH={int(config.restart_every_case_batch)}",
                f"OSWORLD_AUTO_FALLBACK_PORTS={int(config.auto_fallback_ports)}",
                f"MAX_RESTARTS={config.max_restarts}",
                f"UNHEALTHY_THRESHOLD={config.unhealthy_threshold}",
                f"VARIANT_DIR={existing.get('VARIANT_DIR', '')}",
                f"PID_FILE={existing.get('PID_FILE', '')}",
                f"LOG_FILE={existing.get('LOG_FILE', '')}",
                f"CURRENT_LOG_FILE={existing.get('CURRENT_LOG_FILE', '')}",
                f"RESTART_ENV={existing.get('RESTART_ENV', '')}",
                f"RESTART_COMMAND={existing.get('RESTART_COMMAND', '')}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def derive_worker_config(
    variant_dir: Path,
    software_type: str,
    style_id: str,
    port_base: int = 5030,
    base_image: Path = DEFAULT_BASE_IMAGE,
) -> WorkerConfig:
    worker_env = load_worker_env(variant_dir / "docker" / "worker.env")
    suffix = style_id.replace("style_", "")
    worker_name = worker_env.get("WORKER_NAME") or f"{software_type.lower()}_{style_id}"
    container_name = worker_env.get("OSWORLD_CONTAINER") or f"kangj_{software_type.lower()}_{style_id}"
    index = int(suffix) if suffix.isdigit() else 1
    bases = configured_port_bases()
    if port_base != 5030:
        # Preserve the legacy optional API base argument while deriving the other
        # three bases with their historical offsets.
        bases = (port_base + 1, port_base + 3001, port_base + 3171, port_base + 4271)
    default_group = port_group_for_offset(max(0, index - 1), bases)
    api_port = int(worker_env.get("OSWORLD_API_PORT") or os.environ.get("OSWORLD_API_PORT") or default_group[0])
    vnc_port = int(worker_env.get("OSWORLD_VNC_PORT") or os.environ.get("OSWORLD_VNC_PORT") or default_group[1])
    http_port = int(worker_env.get("OSWORLD_HTTP_PORT") or os.environ.get("OSWORLD_HTTP_PORT") or default_group[2])
    cdp_port = int(worker_env.get("OSWORLD_CDP_PORT") or os.environ.get("OSWORLD_CDP_PORT") or default_group[3])
    overlay = Path(
        worker_env.get("OSWORLD_OVERLAY")
        or os.environ.get("OSWORLD_OVERLAY", "")
        or DEFAULT_OVERLAY_DIR / f"{software_type}_{style_id}.qcow2"
    )
    # OSWORLD_BASE_IMAGE may be specified as a path relative to the repo root
    # (e.g. docker/base/Ubuntu_osworld_minimal.qcow2).
    base_image_value = os.environ.get("OSWORLD_BASE_IMAGE")
    resolved_base_image = Path(base_image_value) if base_image_value else Path(base_image)
    if not resolved_base_image.is_absolute():
        resolved_base_image = REPO_ROOT / resolved_base_image
    return WorkerConfig(
        software_type=software_type,
        style_id=style_id,
        worker_name=worker_name,
        container_name=container_name,
        api_port=api_port,
        vnc_port=vnc_port,
        http_port=http_port,
        cdp_port=cdp_port,
        overlay_path=overlay,
        base_image=resolved_base_image,
        docker_image=os.environ.get("OSWORLD_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE),
        # QEMU resource limits (-smp / -m) for each OSWorld VM. Defaults to 8C/8G;
        # override via worker.env or process env vars OSWORLD_CPU_CORES / OSWORLD_RAM_SIZE.
        cpu_cores=config_int(worker_env, "OSWORLD_CPU_CORES", WorkerConfig.cpu_cores),
        ram_size=(worker_env.get("OSWORLD_RAM_SIZE")
                  or os.environ.get("OSWORLD_RAM_SIZE")
                  or WorkerConfig.ram_size),
        # Default timeout for HTTP requests such as /execute and /platform. On a
        # freshly booted or heavily loaded VM, deploy commands (pkill, mkdir,
        # uploads, etc.) can exceed 10s; increase via OSWORLD_REQUEST_TIMEOUT.
        request_timeout_seconds=float(config_int(worker_env, "OSWORLD_REQUEST_TIMEOUT", int(WorkerConfig.request_timeout_seconds))),
        health_passes=config_int(worker_env, "OSWORLD_RECOVERY_HEALTH_PASSES", 3),
        max_restarts=config_int(worker_env, "MAX_RESTARTS", 0),
        unhealthy_threshold=config_int(worker_env, "UNHEALTHY_THRESHOLD", 2),
        case_batch_size=config_int(worker_env, "OSWORLD_CASE_BATCH_SIZE", WorkerConfig.case_batch_size),
        restart_every_case_batch=config_bool(worker_env, "OSWORLD_RESTART_EVERY_CASE_BATCH", WorkerConfig.restart_every_case_batch),
        reset_overlay_on_recovery=config_bool(
            worker_env,
            "OSWORLD_RESET_OVERLAY_ON_RECOVERY",
            WorkerConfig.reset_overlay_on_recovery,
        ),
        auto_fallback_ports=config_bool(worker_env, "OSWORLD_AUTO_FALLBACK_PORTS", True),
    )


class DockerWorkerManager:
    def __init__(self, config: WorkerConfig, state_dir: Optional[Path] = None, dry_run: bool = False) -> None:
        self.config = config
        self.state_dir = state_dir or Path.cwd()
        self.dry_run = dry_run
        self.restart_count = 0
        self.consecutive_unhealthy = 0
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.state_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def run_command(self, command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

    def ensure_overlay(self, reset: bool = False) -> None:
        if self.dry_run:
            self.config.overlay_path.parent.mkdir(parents=True, exist_ok=True)
            return
        if not self.config.base_image.exists():
            raise InfrastructureError(f"OSWorld base image does not exist: {self.config.base_image}")
        self.config.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.config.overlay_path.exists():
            self.config.overlay_path.unlink()
        if self.config.overlay_path.exists():
            return
        command = [
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            "/base/Ubuntu.qcow2",
            "-u",
            str(self.config.overlay_path),
            "80G",
        ]
        self.run_command(command)

    def remove_container(self) -> None:
        # `-v` is required: the OSWorld image declares VOLUME /storage, so every
        # `docker run` creates a fresh ~17GB anonymous volume (the QEMU disk).
        # Removing the container without -v orphans that volume; over a long run
        # (10 styles x 3 rounds x workers x health-check restarts) this leaks
        # hundreds of GB of dangling anonymous volumes onto the space-constrained `/`.
        self.run_command(["docker", "rm", "-f", "-v", self.config.container_name], check=False)

    def start_container(self, reset_container: bool = True, reset_overlay: bool = False) -> None:
        if reset_container:
            self.remove_container()
        self.ensure_overlay(reset=reset_overlay)

        # Optimistic pre-scan avoids holding the repository lock while looking at
        # an already occupied preferred group. The same group is checked again
        # under the lock before docker run.
        self.ensure_ports_available(allow_existing_container=not reset_container)
        rejected_groups: set[tuple[int, int, int, int]] = set()
        max_bind_attempts = max(1, env_int("OSWORLD_PORT_BIND_RETRIES", DEFAULT_PORT_BIND_RETRIES))
        result: Optional[subprocess.CompletedProcess[str]] = None
        command: list[str] = []
        with port_allocation_lock():
            for bind_attempt in range(1, max_bind_attempts + 1):
                self.ensure_ports_available(
                    allow_existing_container=not reset_container,
                    excluded_groups=rejected_groups,
                )
                command = self._docker_run_command()
                result = self.run_command(command, check=False)
                if result.returncode == 0:
                    break
                if not (self.config.auto_fallback_ports and self._is_docker_port_conflict(result)):
                    raise InfrastructureError(
                        f"docker run failed with exit code {result.returncode}: "
                        f"{(result.stderr or result.stdout).strip()[-1000:]}"
                    )
                rejected_groups.add(self.current_port_group())
                self.run_command(
                    ["docker", "rm", "-f", "-v", self.config.container_name],
                    check=False,
                )
                if bind_attempt >= max_bind_attempts or not self.select_fallback_ports(rejected_groups):
                    raise InfrastructureError(
                        f"docker port binding failed after {bind_attempt} attempts: "
                        f"{(result.stderr or result.stdout).strip()[-1000:]}"
                    )
            else:
                raise InfrastructureError("docker port binding attempts exhausted")
        if result is None or result.returncode != 0:
            raise InfrastructureError("docker container did not start")
        self.write_state(
            "docker-start.json",
            {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "container": self.config.container_name,
                "base_url": self.config.base_url,
                "novnc_url": self.config.novnc_url,
            },
        )
        self.wait_until_healthy()
        self.configure_vm_proxy()

    def current_port_group(self) -> tuple[int, int, int, int]:
        return (
            self.config.api_port,
            self.config.vnc_port,
            self.config.http_port,
            self.config.cdp_port,
        )

    def _docker_run_command(self) -> list[str]:
        return [
            "docker",
            "run",
            "-d",
            "--name",
            self.config.container_name,
            "--privileged",
            "-e",
            "BOOT=/boot.qcow2",
            "-e",
            f"CPU_CORES={self.config.cpu_cores}",
            "-e",
            f"RAM_SIZE={self.config.ram_size}",
            "-p",
            f"{self.config.api_port}:5000",
            "-p",
            f"{self.config.vnc_port}:8006",
            "-p",
            f"{self.config.http_port}:8080",
            "-p",
            f"{self.config.cdp_port}:9222",
            "-v",
            f"{self.config.base_image}:/base/Ubuntu.qcow2:ro",
            "-v",
            f"{self.config.overlay_path}:/boot.qcow2",
            self.config.docker_image,
        ]

    @staticmethod
    def _is_docker_port_conflict(result: subprocess.CompletedProcess[str]) -> bool:
        if result.returncode != 125:
            return False
        text = f"{result.stdout}\n{result.stderr}".lower()
        return any(
            marker in text
            for marker in (
                "port is already allocated",
                "address already in use",
                "failed programming external connectivity",
                "bind for 0.0.0.0",
            )
        )

    def configure_vm_proxy(self) -> None:
        # After the guest is healthy, point its network proxy at the host relay
        # (docker bridge gateway 172.17.0.1), started on demand. Enabled by
        # default; set OSWORLD_VM_PROXY="" or OSWORLD_VM_PROXY_DISABLE=1 to skip.
        if self.dry_run or env_bool("OSWORLD_VM_PROXY_DISABLE", False):
            return
        proxy = os.environ.get("OSWORLD_VM_PROXY", "http://172.17.0.1:7890")
        if not proxy.strip():
            return
        try:
            from agent.docker.vm_proxy import (
                DEFAULT_RELAY_TARGET,
                configure_vm_proxy,
                ensure_relay,
            )

            listen = proxy.split("://", 1)[-1]
            target = "127.0.0.1:" + proxy.rsplit(":", 1)[1] if ":" in proxy else DEFAULT_RELAY_TARGET
            relay = ensure_relay(listen=listen, target=target)
            scope = os.environ.get("OSWORLD_VM_PROXY_SCOPE", "both")
            sudo_password = os.environ.get("OSWORLD_CLIENT_PASSWORD", "password")
            outcome = configure_vm_proxy(
                self.execute, proxy=proxy, scope=scope, sudo_password=sudo_password
            )
            self.write_state("vm-proxy.json", {"relay": relay, **outcome})
        except Exception as exc:  # proxy setup must never break VM startup
            self.write_state("vm-proxy.json", {"configured": False, "error": str(exc), "proxy": proxy})

    def ensure_ports_available(
        self,
        allow_existing_container: bool = False,
        excluded_groups: Optional[set[tuple[int, int, int, int]]] = None,
    ) -> None:
        if self.dry_run:
            return
        excluded_groups = excluded_groups or set()
        current = self.current_port_group()
        if current not in excluded_groups and port_group_free(current):
            return
        if allow_existing_container:
            return
        if self.config.auto_fallback_ports and self.select_fallback_ports(excluded_groups):
            return
        occupied = [port for port in current if not is_port_free(port)]
        raise InfrastructureError(f"Port is already in use and no fallback group is available: {occupied}")

    def select_fallback_ports(
        self,
        excluded_groups: Optional[set[tuple[int, int, int, int]]] = None,
    ) -> bool:
        excluded_groups = excluded_groups or set()
        raw_groups = os.environ.get("OSWORLD_PORT_GROUPS", "")
        groups: Iterable[tuple[int, int, int, int]]
        groups = parse_port_groups(raw_groups) if raw_groups else iter_port_groups()
        current = self.current_port_group()
        for api_port, vnc_port, http_port, cdp_port in groups:
            group = (api_port, vnc_port, http_port, cdp_port)
            if group == current or group in excluded_groups:
                continue
            if port_group_free(group):
                old = {
                    "api_port": self.config.api_port,
                    "vnc_port": self.config.vnc_port,
                    "http_port": self.config.http_port,
                    "cdp_port": self.config.cdp_port,
                }
                self.config.api_port = api_port
                self.config.vnc_port = vnc_port
                self.config.http_port = http_port
                self.config.cdp_port = cdp_port
                self.write_state("port-fallback.json", {"old": old, "new": self.snapshot_state()})
                return True
        return False

    def http_get(self, path: str) -> str:
        url = self.config.base_url + path
        with request.urlopen(url, timeout=self.config.request_timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")

    def http_get_bytes(self, path: str) -> bytes:
        url = self.config.base_url + path
        with request.urlopen(url, timeout=self.config.request_timeout_seconds) as response:
            return response.read()

    def http_post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        request_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = self.config.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=request_timeout or self.config.request_timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    def try_execute_payloads(self, command: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        marker = f"__APPEVAL_EXEC_SCHEMA_{int(time.time() * 1000)}__"
        wrapped_command = f"printf '{marker}\\n'\n{command}"
        payloads: list[Dict[str, Any]] = [
            {"command": wrapped_command, "shell": True},
            {"cmd": wrapped_command, "shell": True},
            {"script": wrapped_command},
            {"command": wrapped_command},
        ]
        if timeout:
            for payload in payloads:
                payload["timeout"] = timeout
        request_timeout = max(self.config.request_timeout_seconds, float(timeout) + 5.0) if timeout else None
        errors: list[str] = []
        for payload in payloads:
            try:
                result = self.http_post_json("/execute", payload, request_timeout=request_timeout)
                text = json.dumps(result, ensure_ascii=False)
                if marker in text:
                    self.write_state("execute-schema.json", {"selected_payload": payload, "result_sample": result})
                    return result
                errors.append(text[:500])
            except Exception as exc:
                errors.append(f"{payload}: {exc}")
        raise InfrastructureError(f"Could not execute command through OSWorld API. Tried payloads: {errors[-4:]}")

    @staticmethod
    def _looks_inactive_png(data: bytes) -> bool:
        try:
            from PIL import Image
        except Exception:
            return False
        try:
            with Image.open(io.BytesIO(data)) as image:
                sample = image.convert("L").resize((160, 90))
                pixels = list(sample.getdata())
        except Exception:
            return False
        if not pixels:
            return False
        dark_ratio = sum(1 for value in pixels if value < 25) / len(pixels)
        bright_ratio = sum(1 for value in pixels if value > 200) / len(pixels)
        average = sum(pixels) / len(pixels)
        return dark_ratio > 0.88 and bright_ratio < 0.08 and average < 35

    def check_platform(self) -> bool:
        if self.dry_run:
            return True
        try:
            text = self.http_get("/platform")
            return "linux" in text.lower() or "ubuntu" in text.lower()
        except Exception:
            return False

    def check_execute(self) -> bool:
        if self.dry_run:
            return True
        try:
            result = self.try_execute_payloads("printf __APPEVAL_HEALTH_OK__")
            text = json.dumps(result, ensure_ascii=False)
            return "__APPEVAL_HEALTH_OK__" in text
        except Exception:
            return False

    def check_screenshot(self) -> bool:
        if self.dry_run:
            return True
        try:
            data = self.http_get_bytes("/screenshot")
            return data.startswith(b"\x89PNG") and len(data) > 1024 and not self._looks_inactive_png(data)
        except Exception:
            return False

    def is_healthy_once(self) -> bool:
        return self.check_platform() and self.check_execute() and self.check_screenshot()

    def wait_until_healthy(self) -> None:
        passes = 0
        # attempts bounds how long we wait for the VM to become healthy (each
        # attempt waits health_interval_seconds). Under high load or I/O
        # contention the first boot can be very slow; use OSWORLD_HEALTH_ATTEMPTS
        # to extend the startup window instead of raising health_passes, which is
        # the consecutive-pass threshold and would make stability harder to reach.
        attempts = env_int("OSWORLD_HEALTH_ATTEMPTS", max(self.config.health_passes * 10, 30))
        failures: list[str] = []
        for _ in range(attempts):
            if self.is_healthy_once():
                passes += 1
                if passes >= self.config.health_passes:
                    time.sleep(self.config.settle_seconds)
                    self.consecutive_unhealthy = 0
                    self.write_state("health.json", {"healthy": True, "passes": passes})
                    return
            else:
                passes = 0
                failures.append("platform_execute_or_screenshot_failed")
            time.sleep(self.config.health_interval_seconds)
        self.write_state("health.json", {"healthy": False, "failures": failures[-10:]})
        raise InfrastructureError("OSWorld stable health check did not pass")

    def health_check_for_running_worker(self) -> bool:
        healthy = self.is_healthy_once()
        if healthy:
            self.consecutive_unhealthy = 0
            return True
        self.consecutive_unhealthy += 1
        return False

    def recover(self, reason: str, force_overlay_reset: bool = False) -> RecoveryResult:
        # max_restarts <= 0 means unlimited (no total restart budget); this is
        # required for case_batch_size=1, where every case forces one restart.
        if self.config.max_restarts > 0 and self.restart_count >= self.config.max_restarts:
            return RecoveryResult("give_up", False, self.restart_count, f"max restarts reached: {reason}")
        self.restart_count += 1
        reset_overlay = force_overlay_reset or self.config.reset_overlay_on_recovery
        try:
            self.start_container(reset_container=True, reset_overlay=reset_overlay)
            return RecoveryResult("restart_overlay" if reset_overlay else "restart_container", True, self.restart_count, reason)
        except InfrastructureError as exc:
            if not reset_overlay and (self.config.max_restarts <= 0 or self.restart_count < self.config.max_restarts):
                try:
                    self.start_container(reset_container=True, reset_overlay=True)
                    return RecoveryResult("restart_container_then_overlay", True, self.restart_count, str(exc))
                except InfrastructureError as second_exc:
                    return RecoveryResult("failed", False, self.restart_count, str(second_exc))
            return RecoveryResult("failed", False, self.restart_count, str(exc))

    def after_case_batch(self, cases_completed_since_recovery: int) -> RecoveryResult:
        if self.config.case_batch_size <= 0:
            return RecoveryResult("skip", True, self.restart_count)
        if cases_completed_since_recovery <= 0 or cases_completed_since_recovery % self.config.case_batch_size != 0:
            return RecoveryResult("skip", True, self.restart_count)
        if self.config.restart_every_case_batch:
            return self.recover(f"forced restart after {cases_completed_since_recovery} cases since last recovery")
        if self.health_check_for_running_worker():
            return RecoveryResult("health_check_passed", True, self.restart_count)
        if self.consecutive_unhealthy >= self.config.unhealthy_threshold:
            return self.recover(f"unhealthy after case batch {cases_completed_since_recovery} since last recovery")
        return RecoveryResult("unhealthy_below_threshold", True, self.restart_count)

    def execute(self, command: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self.dry_run:
            return {"dry_run": True, "command": command}
        try:
            payload: Dict[str, Any] = {"command": command, "shell": True}
            if timeout:
                payload["timeout"] = timeout
            request_timeout = max(self.config.request_timeout_seconds, float(timeout) + 5.0) if timeout else None
            return self.http_post_json("/execute", payload, request_timeout=request_timeout)
        except Exception as exc:
            self.write_state("execute-primary-error.json", {"error": str(exc), "command_sample": command[:500]})
            return self.try_execute_payloads(command, timeout=timeout)

    def snapshot_state(self) -> Dict[str, Any]:
        payload = asdict(self.config)
        payload["overlay_path"] = str(self.config.overlay_path)
        payload["base_image"] = str(self.config.base_image)
        payload["base_url"] = self.config.base_url
        payload["novnc_url"] = self.config.novnc_url
        payload["restart_count"] = self.restart_count
        payload["consecutive_unhealthy"] = self.consecutive_unhealthy
        return payload


__all__ = [
    "DockerWorkerManager",
    "InfrastructureError",
    "RecoveryResult",
    "WorkerConfig",
    "derive_worker_config",
    "write_worker_env",
]
