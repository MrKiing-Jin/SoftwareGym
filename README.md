# SoftwareGym

SoftwareGym is a **software generation pipeline** that synthesizes a complete desktop application from a software-type description, then evaluates it in a real GUI environment.

Given a software type (e.g. `ChatAPP`, `PicEditor`), the pipeline:

1. **Design** — FunctionDesigner writes the requirement spec; StyleDesigner / SubStyleDesigner produce N distinct UI style blueprints (Avalonia `.axaml`); IconDesigner / SubIconDesigner produce per-style SVG icon sets (with hard validation gates).
2. **Build** — Coder implements the app (.NET 8 + Avalonia 11); DevCoder reviews and fixes it in place. All design/coding stages run through the Codex CLI.
3. **Evaluate** — EvalTaskGenerateAgent creates GUI test cases; GUIEval boots OSWorld Docker containers (real Ubuntu VMs) and drives the app with a GUI agent (GPT-5.4-style computer-use or Seed), producing a per-round `summary.json`.
4. **Iterate** — rounds v1 → v3: each repair round feeds the previous round's eval summary and review back to Coder/DevCoder for incremental fixes.

**OSWorld environment alignment.** To align software synthesis and validation with downstream OSWorld evaluation, SoftwareGym uses the same Docker-based virtual-machine infrastructure and Ubuntu desktop environment as OSWorld. Generated applications are deployed into this shared runtime and tested through screenshots and PyAutoGUI actions. This keeps GUI interaction and execution conditions consistent with OSWorld evaluation, while the resulting test feedback drives SoftwareGym's iterative repair loop.

The final deliverable of every run is `output/<run>/software_variants/<style>/loop/vN/coder/app` — the generated software package that gets evaluated.

## Package layout

```
SoftwareGym/
├── agent/                  # all pipeline code
│   ├── loop_controller/    # cli.py (scaffold), runner.py (main loop), launcher.py,
│   │                       # resume.py, manifest.py, validators.py,
│   │                       # avalonia_blueprint.py, usage.py (cost stats)
│   ├── eval/               # run_eval.py (GUIEval), gui_agent_adapter.py, state_probe.py,
│   │                       # validate_cases.py, cost.py
│   ├── docker/             # worker_manager.py — OSWorld container/overlay lifecycle
│   │                       # (+ cloudflared.py, vm_proxy.py, watchdog.py, smoke_test.py)
│   ├── icons/              # icon projection/SVG gates (project_style_manifest.py,
│   │                       # svg_contract.py, validate_style_icons.py, validate_icon_usage.py)
│   ├── metrics/            # style-diversity & fidelity metrics (style_diversity.py,
│   │                       # built_diversity.py, visual_fidelity.py, blueprint_tree.py, vocab.py)
│   ├── mm_agents/          # gpt54_agent.py, seed_agent.py (GUI agent implementations,
│   │                       # dynamically loaded by gui_agent_adapter.py)
│   ├── prompt/             # 10 stage prompt templates (JSON)
│   ├── config/             # agent-runtime.json (per-agent Codex policy)
│   ├── skill/              # 8 per-stage SKILL.md (embedded into rendered prompts)
│   ├── tools/              # env_check, status, cleanup, proxy_relay, volume_cleanup_loop.sh
│   └── runtime_env.py      # shared runtime/environment helpers
├── Resource/SampleSoftware_resources/pipeline_parameters.json
│                           # template; copy it to add a new software type
├── docker/base/            # put Ubuntu_osworld_minimal.qcow2 here
│                           # (OSWorld VM base image, ~35 GB, NOT included in this package)
├── docker/overlays/        # runtime VM overlays land here (empty)
├── .env.example            # annotated template (placeholders, no secrets); copy to .env
└── requirements.txt        # pinned pip requirements for the osworld conda env
```

`output/` (run outputs and `_pipeline_logs/`) and `.env` (the single source for model URL / model name / API keys) are created at setup/runtime and are not part of the checked-in tree.

## Prerequisites (NOT included in this package)

1. **System tools**: `docker`, `qemu-img`, `codex` CLI. The OSWorld Docker container environment is **Ubuntu 22.04.3 LTS (x86_64)**. Optional: `cloudflared` — only needed to expose a worker's noVNC desktop through a temporary `*.trycloudflare.com` URL (`agent/docker/cloudflared.py`); everything works without it.
   - Install the Codex CLI (requires **Node.js 24.18.0**, pinned to **@openai/codex 0.156.1**):
     ```bash
     npm install -g @openai/codex@0.156.1
     ```
2. **The osworld conda environment** (`/opt/miniconda3/envs/osworld/bin/python`): both the orchestration side (scaffold/runner) and GUIEval use this interpreter; the pipeline enforces it at startup. Create it from the pinned `requirements.txt` in this package:
   ```bash
   conda create -n osworld python=3.10.0
   conda activate osworld
   pip install -r requirements.txt
   ```
3. **.NET 8 SDK** (pinned to **8.0.422**): the Coder/DevCoder stages build the generated app (.NET 8 + Avalonia 11, `net8.0` target). The pipeline locates `dotnet` in this order (`avalonia_blueprint.py`): PATH → `SOFTWAREGYM_DOTNET` env var → `~/.local/dotnet8/dotnet` (portable) → `/tmp/dotnet8/dotnet`. Either install it system-wide:
   ```bash
   sudo apt-get update && sudo apt-get install -y dotnet-sdk-8.0
   ```
   or install the portable SDK into the preferred fallback location:
   ```bash
   curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
   bash dotnet-install.sh --channel 8.0 --install-dir ~/.local/dotnet8
   ```
4. **A model endpoint**: any OpenAI-compatible endpoint works — set `BASE_URL`/`OPENAI_BASE_URL`/keys in `.env` accordingly.
5. **Disk**: peak concurrent qcow2 overlays ≈ `parallel-workers × eval-window` (1–4 GB each); check free space on the package's disk before large runs.
6. **Host sysctl**: `fs.inotify.max_user_instances=1024` or higher (the default 128 can silently break VM networking under parallel GUIEval).

## Configuration

Copy `.env.example` to `.env` and fill in real values. All model routing — Codex stages (`MODEL`, `BASE_URL`, `OPENAI_BASE_URL`, `API_KEY`, `OPENAI_API_KEY`, `MODEL_PROVIDER`), GUI agent (`GUIAGENT_*` or the `APPEVAL_LLM_MODEL`/`OPENAI_*` fallback chain), Seed executor/judge (`SEED_EXEC_*`/`SEED_JUDGE_*`) — is read from environment variables only; the code contains no hardcoded endpoints or model names.

Note: `codex-command.sh` scripts bake in the model/base_url/key **at scaffold time** — changing Codex routing in `.env` requires re-scaffolding; GUIEval re-reads `.env` on every launch (`SOFTWAREGYM_ENV_OVERRIDE=1`), so its side changes take effect immediately.

## Proxy / networking

**Host side (pipeline → model endpoint).** With `SOFTWAREGYM_NO_PROXY=1` in `.env` (the default), every child process spawned by the runner/GUIEval gets `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` stripped and `NO_PROXY=*` set (`agent/runtime_env.py`), which is what you want for localhost/intranet model endpoints. The Quick start launch command additionally clears proxy vars for the runner process itself. If your endpoint can only be reached through an HTTP proxy, set `SOFTWAREGYM_NO_PROXY=0` and export `HTTPS_PROXY` normally.

**VM side (OSWorld guest → internet).** Enabled by default: once a container is healthy, `worker_manager.configure_vm_proxy()` points the guest's proxy at `http://172.17.0.1:7890` (the docker bridge gateway, i.e. the host as seen from the guest) and starts `agent/tools/proxy_relay.py` on demand, bridging `172.17.0.1:7890 → 127.0.0.1:7890` on the host (where e.g. clash listens). The proxy is injected both system-wide (`/etc/environment`) and into the GNOME desktop (`gsettings`, so Chrome uses it). Relevant env vars:

- `OSWORLD_VM_PROXY` — proxy URL the guest should use (default `http://172.17.0.1:7890`; the relay target port is derived from it)
- `OSWORLD_VM_PROXY=""` or `OSWORLD_VM_PROXY_DISABLE=1` — skip guest proxy setup entirely (do this if the host has no proxy on `127.0.0.1:7890`)
- `OSWORLD_VM_PROXY_SCOPE` — `both` (default) | `cli` | `gui`
- `OSWORLD_CLIENT_PASSWORD` — guest sudo password (default `password`)

To (re)configure an already-running VM by hand: `python -m agent.docker.vm_proxy --api-port <api-port> --proxy http://172.17.0.1:7890`.

## Quick start

Add your software type first: copy `Resource/SampleSoftware_resources/` to `Resource/<YourType>_resources/`, edit `pipeline_parameters.json` (`software_type`, `product_positioning`, optional `eval_task_seedfile`), and keep `default_style_count` aligned with `--style-count`.

All commands run from the package root with:

```bash
PY=/opt/miniconda3/envs/osworld/bin/python
```

**Step 1 — scaffold** (creates directories and renders prompts only; no API calls):

```bash
SOFTWARE_TYPE=ChatAPP
STYLE_COUNT=5
RUN_ID=$(printf '%s_%s_full%s' "$(echo "$SOFTWARE_TYPE" | tr '[:upper:]' '[:lower:]')" "$(date +%Y%m%d_%H%M%S)" "$STYLE_COUNT")
RUN_DIR=$("$PY" -m agent.loop_controller.cli scaffold \
  --software-type "$SOFTWARE_TYPE" \
  --style-count "$STYLE_COUNT" \
  --max-rounds 3 \
  --run-id "$RUN_ID" | tail -n 1)
```

**Step 2 — run** (long-running; always launch detached with setsid + nohup):

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p output/_pipeline_logs
LOG=output/_pipeline_logs/${SOFTWARE_TYPE}_${STAMP}.log
PIDFILE=output/_pipeline_logs/${SOFTWARE_TYPE}_${STAMP}.pid

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY='*' no_proxy='*' \
    SOFTWAREGYM_NO_PROXY=1 SOFTWAREGYM_ENV_OVERRIDE=1 \
    nohup setsid "$PY" -m agent.loop_controller.runner \
      --run-dir "$RUN_DIR" \
      --max-rounds 3 \
      --parallel-styles \
      --parallel-workers 5 \
      --eval-window 3 \
      --coder-workers 5 \
      --case-attempts 1 \
      --case-trajectory-reruns 1 \
      --infra-case-retries 1 \
      --max-steps 50 \
      --codex-stage-retries 2 \
      --codex-stage-timeout-seconds 7200 \
      --coder-stage-timeout-seconds 10800 \
      --stop-on-pass \
      > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
```

**Step 3 — verify the launch**: `ps -p $(cat $PIDFILE) -o pid,ppid,sid,stat,etime,args` (PPID=1, SID=PID), no proxy vars in `/proc/<pid>/environ`, and a fresh `runner_started` line in `$RUN_DIR/runner-events.jsonl`.

## Monitoring and results

- `$RUN_DIR/runner-events.jsonl` — source of truth for pipeline progress (`stage_started` / `stage_completed` / `runner_completed`).
- `$RUN_DIR/software_variants/<style>/loop/vN/eval/summary.json` — per-round GUIEval scores.
- `$RUN_DIR/cost_summary.json` — run-level token-usage table (`python -m agent.loop_controller.usage --run-dir <RUN> --software all` to rebuild).
- `python -m agent.tools.status <RUN_DIR>` — per-style status board; `python -m agent.tools.env_check --json` — pre-flight environment check.

## Notes and limitations

- **Seed-material evaluation**: if `pipeline_parameters.json` sets `eval_task_seedfile` (media/file-type software), put the real seed files into `Resource/<YourType>_resources/<seed-dir>/` yourself — this package ships no sample assets.
- **Container/overlay cleanup**: the runner does not reap finished OSWorld containers by itself. For large runs, run `agent/tools/volume_cleanup_loop.sh` (or an equivalent reaper) alongside, and `docker volume prune -f` afterwards.
- This package is self-contained for its own code, but relies on the host's conda environment, model endpoint, Docker daemon, the OSWorld VM base image (`docker/base/Ubuntu_osworld_minimal.qcow2`, not included), and .NET 8 SDK (`dotnet` on PATH or the portable fallback locations checked by `avalonia_blueprint.py`).
