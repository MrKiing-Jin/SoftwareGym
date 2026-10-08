#!/usr/bin/env python3
"""Lightweight TCP relay so OSWorld VMs can reach a host-loopback proxy.

The host proxy (e.g. clash) usually binds to 127.0.0.1:7890, which is not
reachable from the QEMU guest (it arrives via the docker bridge gateway
172.17.0.1). This relay listens on the docker-bridge-facing address and
forwards every connection to the loopback proxy.

Usage:
    python proxy_relay.py --listen 172.17.0.1:7890 --target 127.0.0.1:7890
"""
import argparse
import asyncio


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(local_reader, local_writer, target_host, target_port):
    try:
        remote_reader, remote_writer = await asyncio.open_connection(target_host, target_port)
    except OSError:
        local_writer.close()
        return
    await asyncio.gather(
        pipe(local_reader, remote_writer),
        pipe(remote_reader, local_writer),
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="172.17.0.1:7890")
    parser.add_argument("--target", default="127.0.0.1:7890")
    args = parser.parse_args()

    lhost, lport = args.listen.rsplit(":", 1)
    thost, tport = args.target.rsplit(":", 1)
    tport = int(tport)

    server = await asyncio.start_server(
        lambda r, w: handle(r, w, thost, tport), lhost, int(lport)
    )
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"relay listening on {addrs} -> {thost}:{tport}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
