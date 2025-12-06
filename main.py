# main.py
from __future__ import annotations
import argparse
import re
import random
import time
import json
import csv
from itertools import chain
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda import build_seed_prompt, default_system_prompt
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code, extract_json, extract_cuda_kernel_names
from scripts.individual import KernelIndividual  # adjust path if needed
from prompts.error import build_error_prompt
from prompts.optimization import build_optimization_prompt
from prompts.judger_repair import build_correctness_prompts
_INVOCATION_SPLITTER = "Invoked with:"


def _sanitize_error_message(exc: Exception) -> str:
    """Strip pybind's large‑tensor printouts and keep only the key error text."""
    msg = str(exc)
    if _INVOCATION_SPLITTER in msg:
        msg = msg.split(_INVOCATION_SPLITTER, 1)[0].rstrip()
    return msg

# ------------------------- CLI -------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Single-LLM self-iterative kernel generation/optimization")
    p.add_argument(
        "kernel_src",
        type=Path,
        help="Path to a single CUDA kernel file (.cu/.cuh) OR a directory containing kernel files",
    )
    p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    p.add_argument("--server_type", default="local", help="LLM provider (local, openai, deepseek, vllm, etc.)")
    p.add_argument("--server_address", default="localhost", help="LLM server address (for vllm/sglang)")
    p.add_argument("--server_port", type=int, default=8000, help="LLM server port (for vllm/sglang)")
    p.add_argument("--model_name", default="deepseek-ai/deepseek-coder-6.7b-instruct", help="LLM model")
    p.add_argument("--round", "-G", type=int, default=10, help="Number of generations per task")
    p.add_argument("--work_dir", type=Path, default=Path("run"), help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-3, help="Max |err| tolerated")
    p.add_argument("--max_tokens", type=int, default=16384, help="LLM max new tokens")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")
    # multi-task controls
    p.add_argument("--first_n", type=int, default=0,
                   help="When arch_py is a directory, take the first N tasks (sorted)")
    p.add_argument("--num_tasks", type=int, default=1,
                   help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0 = time)")

    p.add_argument("--subproc_id", type=int, default=0,
                   help="Identifier for sub-process (e.g., when running multiple in parallel)")

    return p


# ---------------------- naming helpers -----------------
def _slugify_tag(text: str, max_len: int = 80) -> str:
    """Collapse a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if max_len > 0:
        slug = slug[:max_len]
    return slug or "unknown"


def _build_run_tag(server_type: str, model_name: str) -> str:
    server_tag = _slugify_tag(server_type)
    model_tag = _slugify_tag(model_name)
    return f"{server_tag}_{model_tag}"


# ---------------------- small utils --------------------
def _last_n_lines(text: str, n: int = 150) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_full_cuda_source(text: str) -> str:
    """Extract CUDA source from a Python or markdown-like file.

    Order:
      1) ```cuda ... ``` fenced code
      2) source = \"\"\" ... \"\"\"
      3) fallback: raw text
    """
    m = re.search(r"```cuda\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"source\s*=\s*([\"']{3})(.*?)(?:\1)", text, flags=re.DOTALL)
    if m:
        return m.group(2).strip()
    return text.strip()


def _build_history_block(code_dir: Path, keep_last: int = 10) -> str:
    """Collect the CUDA `source` of the most recent *keep_last* kernel files from code_dir."""
    if not code_dir.exists():
        return "## Existing kernels\n(None yet)\n"

    files: List[Path] = sorted(
        list(code_dir.glob("*.py")) + list(code_dir.glob("*.cu")),
        key=lambda p: p.stat().st_mtime,
    )[-keep_last:]

    if not files:
        return "## Existing kernels\n(None yet)\n"

    snippets: List[str] = []
    for idx, p in enumerate(files, 1):
        try:
            cuda_src = _extract_full_cuda_source(_read_text(p))
        except Exception:
            cuda_src = "(failed to read/extract)"
        snippets.append(f"### Kernel {idx} · {p.name}\n```cuda\n{cuda_src}\n```")

    return "## Existing kernels\n" + "\n\n".join(snippets) + "\n"


# ------------------- LLM & eval steps ------------------
def _make_llm_caller(args):

    def call_llm(
        prompt: str,
        sys_prompt: Optional[str] = None,
        log_path: Optional[Path] = None,
        call_type: str = "unknown",
        round_idx: int = -1,
    ) -> str:
        sp = default_system_prompt if sys_prompt is None else sys_prompt
        res = query_server(
            prompt=prompt,
            system_prompt=sp,
            server_type=args.server_type,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            server_address=args.server_address,
            server_port=args.server_port,
            log_path=str(log_path) if log_path else None,
            call_type=call_type,
            round_idx=round_idx,
        )
        if isinstance(res, list):
            return res[0] if res else ""
        return str(res)
    return call_llm


def _llm_to_kernel(
    prompt: str,
    code_dir: Path,
    call_llm,
    io_dir: Path,
    round_idx,
    sys_prompt: Optional[str] = None,   # New: optional system prompt
    log_path: Optional[Path] = None,
    call_type: str = "unknown",
) -> KernelIndividual:
    """LLM → code → save → KernelIndividual (no evaluation)."""
    raw = call_llm(
        prompt,
        sys_prompt=sys_prompt,
        log_path=log_path,
        call_type=call_type,
        round_idx=round_idx,
    )
    reply_file = io_dir / f"{round_idx}_raw_reply.txt"
    reply_file.write_text(raw, encoding="utf-8")
    code = extract_code_block(raw) or raw  # fallback
    path = save_kernel_code(code, code_dir)
    ind = KernelIndividual(code)
    ind.code_path = path  # type: ignore[attr-defined]
    return ind

# ================== Top-level worker: MUST live at module top level, not inside another function ==================


def _bench_worker_entry(test_cu: str,
                        ref_cu: str,
                        device_idx: int,
                        warmup: int,
                        repeat: int,
                        tol: float,
                        conn) -> None:
    """
    Subprocess entry: set GPU, call compare_and_bench, and send result or error
    back to the parent via a Pipe. Note: we pass string paths here to avoid
    non-picklable objects.
    """
    import torch
    from pathlib import Path
    from utils.compile_and_run import CompilationError, AccuracyError

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_idx)

        res = compare_and_bench(
            ref_cu=Path(ref_cu),
            test_cu=Path(test_cu),
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
        )
        conn.send(("ok", res))
    except Exception as e:
        # Clean the error message if helper is available; otherwise fall back to str(e)
        try:
            cleaned = _sanitize_error_message(e)
            msg = _last_n_lines(cleaned)
        except Exception:
            msg = str(e)

        if isinstance(e, CompilationError):
            err_type = "CompilationError"
        elif isinstance(e, AccuracyError):
            err_type = "AccuracyError"
        else:
            err_type = e.__class__.__name__

        conn.send(("err", {"type": err_type, "message": msg}))
    finally:
        # Try to sync at the end so errors surface within this round
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_idx)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


# ================== Keep original behavior: _bench_and_score (uses spawn + top-level worker) ==================
def _bench_and_score(
    ind: KernelIndividual,
    *,
    ref_cu: Path,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
    phase: str = "seed",
    metrics_dir: Path | None = None,
) -> None:
    """
    Benchmark and update the individual's metrics/score; on exception, fill in
    failure info and save metrics (if a directory is provided).
    Same functionality as the original version, but runs compare_and_bench in a
    **spawned subprocess** to isolate the CUDA context.
    """
    import torch
    from multiprocessing import get_context

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Only pass picklable arguments (e.g., string paths)
    p = ctx.Process(
        target=_bench_worker_entry,
        args=(
            str(ind.code_path),  # type: ignore[attr-defined]
            str(ref_cu),
            device_idx,
            warmup,
            repeat,
            tol,
            child_conn,
        ),
    )
    p.start()
    # Parent does not use the child end
    try:
        child_conn.close()
    except Exception:
        pass

    # Wait for child and receive the payload
    p.join()
    payload = parent_conn.recv() if parent_conn.poll() else None
    try:
        parent_conn.close()
    except Exception:
        pass

    # —— Update metrics/score based on child payload (same logic as before) ——
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
            metrics["phase"] = phase
            speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup
            print(f"[{phase}] score={speedup:.4f}")

            # # === Optional: on successful compile+run, copy code to root/test_kernel.cu ===
            # try:
            #     from pathlib import Path as _Path
            #     import shutil as _shutil
            #     root_dir = _Path(__file__).resolve().parent
            #     dst = root_dir / "test_kernel.cu"
            #     src = _Path(ind.code_path)  # type: ignore[arg-type]
            #     if src.exists():
            #         _shutil.copy2(src, dst)
            #         print(f"[{phase}] saved successful kernel to: {dst}")
            #     else:
            #         print(f"[{phase}] WARNING: source code file not found: {src}")
            # except Exception as _copy_exc:
            #     print(f"[{phase}] WARNING: failed to save test_kernel.py: {_copy_exc}")

        else:
            err_type = "RuntimeError"
            message = data
            if isinstance(data, dict):
                err_type = data.get("type", err_type) or err_type
                message = data.get("message", message)

            if not isinstance(message, str):
                message = str(message)

            print(f"\033[91mTest Error ({err_type}):\033[0m {message}")
            ind.metrics = {
                "runnable": False,
                "phase": phase,
                "error_type": err_type,
                "message": message,
            }
            ind.score = float("-inf")
            print(f"[{phase}] failed. See metrics.message for details.")
    else:
        # Subprocess exited unexpectedly with no payload
        ind.metrics = {
            "runnable": False,
            "phase": phase,
            "error_type": "SubprocessCrashed",
            "message": "subprocess exited unexpectedly (no payload received)",
        }
        ind.score = float("-inf")
        print(f"[{phase}] failed. Subprocess crashed.")

    # —— As before: try to save metrics regardless of success/failure ——
    if metrics_dir is not None:
        try:
            saved = ind.save_metrics(metrics_dir)
            print(f"[{phase}] metrics saved to: {saved}")
        except Exception as save_exc:
            print(f"[{phase}] WARNING: failed to save metrics: {save_exc}")

    # Light cleanup in parent (not required, but safer)
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device_idx)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ---------------------- task helpers -------------------
def _collect_tasks(maybe_dir: Path) -> List[Path]:
    """If a directory, return all .py files (sorted); if a file, return [file]."""
    if maybe_dir.is_file():
        return [maybe_dir]
    if maybe_dir.is_dir():
        return sorted([p for p in maybe_dir.rglob("*.cuh") if p.is_file()])
    raise FileNotFoundError(f"{maybe_dir} not found, expected file with '.cuh' suffix or directory which contains it ")


def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]


def _sample_tasks(all_tasks: List[Path], k: int, seed: int | None) -> List[Path]:
    if not all_tasks:
        raise RuntimeError("No .cu/.cuh tasks found.")
    k = max(1, min(k, len(all_tasks)))
    if seed is None or seed == 0:
        seed = int(time.time())
    rng = random.Random(seed)
    return rng.sample(all_tasks, k)


def _plot_scores(save_path: Path, scores: List[float], err_flags: List[bool], title: str):
    """Plot per-round score curve; mark error rounds with an 'x'."""
    xs = list(range(len(scores)))
    plt.figure()
    plt.plot(xs, scores, marker="o")
    for x, y, bad in zip(xs, scores, err_flags):
        if bad:
            plt.scatter([x], [y], marker="x")
    plt.xlabel("Round")
    plt.ylabel("Speedup (ref/test)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _append_usage_totals(log_path: Path) -> Dict[str, int]:
    """Append a totals row to usage.csv and return the summed token counts."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not log_path.exists():
        return totals

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames or not rows:
        return totals

    for row in rows:
        if row.get("call_type") == "sum" or row.get("timestamp") == "Total":
            continue
        for key in totals:
            try:
                totals[key] += int(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue

    total_row = {fn: "" for fn in fieldnames}
    for key, value in totals.items():
        if key in total_row:
            total_row[key] = str(value)

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(total_row)

    return totals


# --------------------- single-task run -----------------
def _run_single_task(task_path: Path, args, batch_dir: Path) -> Dict[str, Any] | None:
    # --- per-task directories under the SAME batch_dir
    task_root = (batch_dir / task_path.stem).resolve()
    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir = eval_dir / "llm_io"

    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "usage.csv"

    # naive CUDA baseline（.cu 必须存在；prompt 优先使用对应的 .cuh 源）
    root_dir = Path(__file__).resolve().parent
    test_kernel = root_dir / f"test_kernel_{args.subproc_id}.cu"
    if task_path.suffix not in (".cu", ".cuh"):
        raise ValueError(f"Expected a CUDA baseline file (.cu/.cuh), got {task_path}")

    # Determine reference CUDA file
    if task_path.suffix == ".cu":
        ref_cu = task_path
    else:
        # If task_path is .cuh, try for a .cu wrapper (legacy), otherwise use .cuh directly
        # (KernelRunner will handle codegen if wrapper is missing)
        wrapper_path = task_path.with_suffix(".cu")
        ref_cu = wrapper_path if wrapper_path.exists() else task_path

    if not ref_cu.exists():
        raise FileNotFoundError(f"Baseline CUDA file not found: {ref_cu}")
    prompt_src = task_path if task_path.suffix == ".cuh" else (
        task_path.with_suffix(".cuh") if task_path.with_suffix(".cuh").exists() else task_path
    )

    call_llm = _make_llm_caller(args)

    current_kernel: Optional[KernelIndividual] = None
    best_kernel: Optional[KernelIndividual] = None
    best_score: float = float("-inf")

    scores: List[float] = []
    err_flags: List[bool] = []
    last_score_for_curve = 0.0  # 早期失败时绘图的默认基线

    for round_idx in range(args.round):
        print(f"[{task_path.name}] Round {round_idx}")

        if round_idx == 0:
            print("[Seed] Generating the initial kernel ...")
            seed_prompt = build_seed_prompt(arch_path=prompt_src, gpu_name=args.gpu)
            prompt_file = io_dir / f"round{round_idx:03d}_seed_prompt.txt"
            prompt_file.write_text(seed_prompt, encoding="utf-8")
            return None
            ind = _llm_to_kernel(seed_prompt, code_dir, call_llm, io_dir,
                                 round_idx, log_path=log_path, call_type="seed")
            _bench_and_score(
                ind,
                ref_cu=ref_cu,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                phase="seed",
                metrics_dir=eval_dir,
            )

        else:
            is_runnable = bool(getattr(current_kernel, "metrics", {}).get(
                "runnable", False)) if current_kernel else False

            if not is_runnable:
                print("[Repair] start repairing")
                error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get(
                    "message", "")) if current_kernel else ""

                problem_system_prompt, problem_prompt = build_correctness_prompts(error_log=error_log,
                                                                                  arch_path=prompt_src,
                                                                                  cuda_code=current_kernel.code)
                prompt_file = io_dir / f"round{round_idx:03d}_problem_identify_prompt.txt"
                prompt_file.write_text(problem_prompt, encoding="utf-8")
                raw = call_llm(problem_prompt, problem_system_prompt, log_path=log_path,
                               call_type="problem_identify", round_idx=round_idx)
                reply_file = io_dir / f"{round_idx}_raw_problem_identify_reply.txt"
                reply_file.write_text(raw, encoding="utf-8")
                problem_json = extract_json(raw)

                repair_prompt = build_error_prompt(
                    old_code=current_kernel.code,
                    error_log=error_log,
                    problem=problem_json,
                    gpu_name=args.gpu,
                )
                prompt_file = io_dir / f"round{round_idx:03d}_repair_prompt.txt"
                prompt_file.write_text(repair_prompt, encoding="utf-8")
                ind = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir,
                                     round_idx, log_path=log_path, call_type="repair")
                _bench_and_score(
                    ind,
                    ref_cu=ref_cu,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="repair",
                    metrics_dir=eval_dir,
                )
            else:
                print("[Refine] runnable kernel, generating optimization prompt without NCU")
                history_block = _build_history_block(code_dir, keep_last=10)
                # Build a simple optimization hint based on current score
                opt_hint = None
                try:
                    cur_score = float(current_kernel.score) if current_kernel.score is not None else None
                    if cur_score is not None and cur_score > 0:
                        opt_hint = {
                            "bottleneck": "Improve latency/throughput on current runnable kernel",
                            "optimisation method": "Apply kernel-level tiling/unrolling/memory optimization",
                            "modification plan": f"Target speedup > {cur_score * 1.05:.2f}x baseline",
                        }
                except Exception:
                    opt_hint = None

                opt_prompt = build_optimization_prompt(
                    arch_path=current_kernel.code_path,  # type: ignore[union-attr]
                    gpu_name=args.gpu,
                    history_block=history_block,
                    optimization_suggestion=opt_hint,
                )
                prompt_file = io_dir / f"round{round_idx:03d}_opt_prompt.txt"
                prompt_file.write_text(opt_prompt, encoding="utf-8")
                ind = _llm_to_kernel(opt_prompt, code_dir, call_llm, io_dir, round_idx,
                                     log_path=log_path, call_type="optimization")
                _bench_and_score(
                    ind,
                    ref_cu=ref_cu,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="opt",
                    metrics_dir=eval_dir,
                )

        # -------- update state + record curve --------
        current_kernel = ind
        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
        this_score = ind.score if (ind.score is not None and runnable) else None

        if this_score is not None:
            last_score_for_curve = this_score
            scores.append(this_score)
            err_flags.append(False)
            if this_score > best_score:
                best_score = this_score
                best_kernel = ind

                with open(test_kernel, "w") as f:
                    f.write(best_kernel.code)

        else:
            # On failure: reuse last score and mark error
            scores.append(last_score_for_curve)
            err_flags.append(True)

    # Plot per-task curve
    fig_path = fig_dir / f"{task_path.stem}_score.png"
    _plot_scores(fig_path, scores, err_flags, title=f"{task_path.stem} (best={best_score:.4f})")
    print(f"[{task_path.name}] Figure saved to: {fig_path}")

    usage_totals = _append_usage_totals(log_path)

    return {
        "task": str(task_path),
        "best_score": float(best_score) if best_score != float("-inf") else 0.0,
        "best_runnable": bool(getattr(best_kernel, "metrics", {}).get("runnable", False)) if best_kernel else False,
        "task_dir": str(task_root),
        "figure": str(fig_path),
        "input_tokens_sum": usage_totals["input_tokens"],
        "output_tokens_sum": usage_totals["output_tokens"],
        "total_tokens_sum": usage_totals["total_tokens"],
    }


# --------------------- 汇总保存 ------------------
def _save_global_summary(batch_dir: Path, summary: List[Dict[str, Any]], avg_speedup: float, accuracy: float, total_tokens_sum: float) -> None:
    """在 batch_dir 下保存 summary.json / summary.csv。"""
    batch_dir.mkdir(parents=True, exist_ok=True)

    # JSON 输出
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "total_tokens_sum": total_tokens_sum,
        "num_tasks": len(summary),
        "tasks": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # CSV 输出
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "best_score", "best_runnable", "task_dir", "figure"])
        for s in summary:
            writer.writerow([s["task"], f'{s["best_score"]:.6f}', int(
                bool(s["best_runnable"])), s["task_dir"], s["figure"]])
        writer.writerow([])
        writer.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        writer.writerow(["accuracy", f"{accuracy:.6f}"])
        writer.writerow(["total_tokens_sum", f"{int(total_tokens_sum)}"])

    print(f"[GLOBAL] Saved: {batch_dir/'summary.json'}")
    print(f"[GLOBAL] Saved: {csv_path}")


# --------------------------- 入口 ----------------------
def main():
    args = _build_arg_parser().parse_args()

    all_tasks = _collect_tasks(args.kernel_src)

    # ---- 本次运行只创建一个 batch 目录 ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _build_run_tag(args.server_type, args.model_name)
    # 命名提示：单文件用文件名，目录用 batch
    if args.kernel_src.is_file():
        batch_name = f"{stamp}_{args.kernel_src.stem}_{run_tag}"
    else:
        # 加入抽样信息便于追溯
        pick_note = f"first{args.first_n}" if (args.first_n and args.first_n >
                                               0) else f"num{args.num_tasks}_seed{args.shuffle_seed}"
        batch_name = f"{stamp}_batch_{pick_note}_{run_tag}"
    batch_dir = (args.work_dir / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] Output folder: {batch_dir}")

    # 单文件：只跑一次（仍在同一 batch 文件夹）
    if args.kernel_src.is_file():
        res = _run_single_task(all_tasks[0], args, batch_dir=batch_dir)
        if res == None:  # NOTE: DEBUG USE
            return

        summary = [res]
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        total_tokens_sum = res.get("total_tokens_sum", 0)
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
        return

    # 目录模式：优先 first_n，否则抽样
    if args.first_n and args.first_n > 0:
        picked = _pick_first_n(all_tasks, args.first_n)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, taking first {len(picked)} (sorted).")
    else:
        picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

    summary: List[Dict[str, Any]] = []
    for i, task in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task} =====")
        res = _run_single_task(task, args, batch_dir=batch_dir)
        if res == None:
            continue
        summary.append(res)

    # 汇总各任务的最优 kernel
    if summary:
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        total_tokens_sum = sum(int(s.get("total_tokens_sum", 0) or 0) for s in summary)
        print("\n===== SUMMARY =====")
        for s in summary:
            print(f"{s['task']}: best_score={s['best_score']:.4f}  runnable={s['best_runnable']}  fig={s['figure']}")
        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        # ---- 统一保存在同一个 batch 目录 ----
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
    else:
        print("No tasks were run.")


if __name__ == "__main__":
    main()
