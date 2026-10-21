"""Command Line Interface for Verigate (PRD §7.6).

Subcommands:
- gateway: run the Go gateway process
- verifier: run the verifier worker
- calibrate: run empirical null calibration
- dashboard: run the FastAPI dashboard and JSON API
- lab up: start the Substitution Lab scenarios server
- lab eval: execute detection experiments across scenarios
- audit: run active audit queries against a provider
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import uvicorn

from .calibrate import calibrate_model
from .config import load as load_config
from .store import Store
from .worker import VerifierWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("verigate.cli")


def cmd_gateway(args):
    """Launches the compiled Go gateway executable."""
    gw_dir = Path("gateway")
    exe = gw_dir / "verigate-gateway.exe" if sys.platform == "win32" else gw_dir / "verigate-gateway"
    if not exe.exists():
        log.info("Building Go gateway binary...")
        subprocess.run(["go", "build", "-o", str(exe), "./cmd/verigate-gateway"], cwd=str(gw_dir), check=True)
    cmd = [str(exe.resolve()), "--config", args.config]
    log.info("Running gateway: %s", " ".join(cmd))
    subprocess.run(cmd)


def cmd_verifier(args):
    """Runs the verifier worker."""
    cfg = load_config(args.config)
    worker = VerifierWorker(cfg)
    try:
        worker.run()
    except KeyboardInterrupt:
        log.info("Stopping verifier worker...")
        worker.stop()


def cmd_calibrate(args):
    """Builds null distribution for a model."""
    cfg = load_config(args.config)
    n = calibrate_model(cfg, args.model)
    log.info("Calibrated %s with %d null samples.", args.model, n)


def cmd_dashboard(args):
    """Runs the FastAPI dashboard and JSON API."""
    os.environ["VERIGATE_CONFIG"] = args.config
    cfg = load_config(args.config)
    host, port = cfg.dashboard_listen.split(":")
    log.info("Starting dashboard at http://%s:%s", host, port)
    uvicorn.run("verigate_verifier.api:app", host=host, port=int(port), log_level="info")


def cmd_lab_up(args):
    """Starts the Substitution Lab scenario providers."""
    log.info("Starting Substitution Lab server on port %d...", args.port)
    uvicorn.run("lab.server:app", host="127.0.0.1", port=args.port, log_level="info")


def cmd_audit(args):
    """Sends active audit traffic to the gateway."""
    cfg = load_config(args.config)
    gw_url = f"http://{cfg.raw.get('gateway', {}).get('listen', '127.0.0.1:8080')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.provider:
        headers["X-Verigate-Provider"] = args.provider

    prompts = [
        "Explain quantum computing in one short paragraph.",
        "Write a Python function to compute the Fibonacci sequence.",
        "Summarize the benefits of verifiable inference.",
        "What is the capital of France and what is its population?",
    ]
    log.info("Sending %d requests to %s for provider=%s model=%s...", args.n, gw_url, args.provider, args.model)
    with httpx.Client(timeout=30.0) as client:
        for i in range(args.n):
            p = prompts[i % len(prompts)]
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": f"{p} [req {i}]"}],
                "temperature": 0.7,
            }
            try:
                resp = client.post(gw_url, json=payload, headers=headers)
                req_id = resp.headers.get("X-Verigate-Request-Id", "none")
                log.info("[%d/%d] HTTP %d (req_id=%s)", i + 1, args.n, resp.status_code, req_id)
            except Exception as e:
                log.error("[%d/%d] Request failed: %v", i + 1, args.n, e)
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(prog="verigate", description="Verigate verification gateway & tools")
    parser.add_argument("--config", default="verigate.yaml", help="Path to verigate.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # gateway
    p_gw = subparsers.add_parser("gateway", help="Run the gateway")
    p_gw.set_defaults(func=cmd_gateway)

    # verifier
    p_vf = subparsers.add_parser("verifier", help="Run the verifier worker")
    p_vf.set_defaults(func=cmd_verifier)

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Build null distributions")
    p_cal.add_argument("--model", required=True, help="Model alias to calibrate")
    p_cal.set_defaults(func=cmd_calibrate)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Run dashboard web UI")
    p_dash.set_defaults(func=cmd_dashboard)

    # lab
    p_lab = subparsers.add_parser("lab", help="Substitution lab commands")
    lab_subs = p_lab.add_subparsers(dest="lab_command", required=True)
    p_lab_up = lab_subs.add_parser("up", help="Start lab scenario endpoints")
    p_lab_up.add_argument("--port", type=int, default=9000, help="Port to listen on")
    p_lab_up.set_defaults(func=cmd_lab_up)

    # audit
    p_aud = subparsers.add_parser("audit", help="Send test audit requests")
    p_aud.add_argument("--provider", required=True, help="Provider name")
    p_aud.add_argument("--model", required=True, help="Model alias")
    p_aud.add_argument("--n", type=int, default=50, help="Number of requests")
    p_aud.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
