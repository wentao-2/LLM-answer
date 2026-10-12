"""Frozen extension inference, with immutable terminal records and native templates.

CLI: python source/run_extension.py --config config.json --model MODEL_NAME
The parent scheduler owns model downloads, manifests, freezing, and GPU assignment.
This worker uses one local checkpoint and one SQLite database. Importing it requires
only the standard library, so protocol/readout tests do not load a GPU runtime.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time


READOUT_VERSIONS = {
    "probability": "unique-final-json-v1",
    "answer_probability": "unique-final-answer-probability-v1",
}
RESULT_COLUMNS = (
    "id", "model", "item_id", "status", "probability", "raw_output",
    "input_tokens", "output_tokens", "duration_sec", "created_at", "metadata_json",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def relative_artifact(run_dir, name):
    """Frozen artifacts must remain inside their portable run directory."""
    if not isinstance(name, str) or not name:
        raise ValueError("Artifact path must be a nonempty string")
    root = Path(run_dir).resolve()
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Artifact escapes run directory: {name}")
    return candidate


def validate_frozen(config_path, running_source=None):
    config_path = Path(config_path).resolve()
    run_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = json.loads((run_dir / "frozen.json").read_text(encoding="utf-8"))
    source_dir = run_dir / "source"
    sources = {p.relative_to(source_dir).as_posix(): digest(p)
               for p in sorted(source_dir.rglob("*.py"))}
    actual = {
        "config_sha256": digest(config_path),
        "manifest_sha256": digest(relative_artifact(run_dir, config["manifest"])),
        "source_hashes": sources,
    }
    for key, value in actual.items():
        if value != frozen.get(key):
            raise RuntimeError(f"Frozen artifact mismatch: {key}. Refusing inference.")
    protocols = frozen.get("protocol_hashes")
    if not isinstance(protocols, dict) or not protocols:
        raise RuntimeError("Frozen protocol_hashes must be a nonempty mapping")
    if config.get("experiment_protocol") not in protocols:
        raise RuntimeError("experiment_protocol is absent from frozen protocol_hashes")
    for name, expected in protocols.items():
        if digest(relative_artifact(run_dir, name)) != expected:
            raise RuntimeError(f"Frozen protocol mismatch: {name}. Refusing inference.")
    executing = Path(running_source or __file__).resolve()
    if executing != (source_dir / "run_extension.py").resolve():
        raise RuntimeError("Execute the frozen run_dir/source/run_extension.py copy")
    if "run_extension.py" not in sources:
        raise RuntimeError("Frozen source set omits run_extension.py")
    return config, frozen


@contextmanager
def worker_lock(path):
    """An OS lock survives stale pid files and is released when a process dies."""
    with Path(path).open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another worker holds this run directory") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another worker holds this run directory") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def connect_db(path):
    db = sqlite3.connect(str(path), timeout=60)
    # Shared network storage requires rollback journaling rather than WAL.
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("""CREATE TABLE IF NOT EXISTS results (
        id TEXT PRIMARY KEY, model TEXT NOT NULL, item_id TEXT NOT NULL,
        status TEXT NOT NULL, probability REAL, raw_output TEXT NOT NULL,
        input_tokens INTEGER, output_tokens INTEGER, duration_sec REAL,
        created_at TEXT NOT NULL, metadata_json TEXT NOT NULL)""")
    columns = tuple(row[1] for row in db.execute("PRAGMA table_info(results)"))
    if columns != RESULT_COLUMNS:
        db.close()
        raise RuntimeError(f"Unexpected results schema: {columns}")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS model_item ON results(model,item_id)")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.commit()
    return db


def require_metadata(db, key, value):
    encoded = canonical_json(value)
    old = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    if old and old[0] != encoded:
        raise RuntimeError(f"Immutable run metadata changed: {key}. Create a new run.")
    if not old:
        with db:
            db.execute("INSERT INTO metadata VALUES (?,?)", (key, encoded))


def insert_result(db, values):
    """Never overwrite or retry any saved terminal status."""
    with db:
        db.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)", values)


def select_rows(rows, model_name):
    selected = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Every manifest line must be an object")
        scope = item.get("model_scope")
        if scope is not None and (not isinstance(scope, list)
                                  or any(not isinstance(m, str) for m in scope)):
            raise ValueError("model_scope must be a list of model names")
        if scope is None or model_name in scope:
            selected.append(item)
    ids = [item.get("id") for item in selected]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Selected manifest items require nonempty string ids")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate item ids after model_scope selection")
    return selected


def remaining_rows(db, model_name, rows):
    done = {row[0] for row in db.execute("SELECT item_id FROM results WHERE model=?", (model_name,))}
    if not done.issubset({item["id"] for item in rows}):
        raise RuntimeError("Saved results contain ids outside the selected frozen manifest")
    return [item for item in rows if item["id"] not in done], done


def positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an explicit positive integer")
    return value


def close_requirement(value):
    keys = [key for key in ("require_think_close", "requires_think_close") if key in value]
    if not keys:
        return None
    if any(type(value[key]) is not bool for key in keys):
        raise ValueError("require_think_close must be Boolean")
    if len({value[key] for key in keys}) != 1:
        raise ValueError("Conflicting reasoning-close aliases")
    return value[keys[0]]


@dataclass(frozen=True)
class GenerationProfile:
    max_new_tokens: int
    max_input_tokens: int
    chat_template_kwargs_json: str
    require_think_close: bool
    readout: str

    @property
    def chat_template_kwargs(self):
        return json.loads(self.chat_template_kwargs_json)

    def metadata(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "max_input_tokens": self.max_input_tokens,
            "chat_template_kwargs": self.chat_template_kwargs,
            "require_think_close": self.require_think_close,
            "readout": self.readout,
            "do_sample": False,
            "num_beams": 1,
        }


def effective_profile(config, model_spec, item):
    kwargs = {}
    for source in (config, model_spec, item):
        if "do_sample" in source and source["do_sample"] is not False:
            raise ValueError("Only greedy decoding (do_sample=false) is supported")
        if "num_beams" in source and (type(source["num_beams"]) is not int
                                       or source["num_beams"] != 1):
            raise ValueError("Only num_beams=1 is supported")
        extra = source.get("chat_template_kwargs", {})
        if not isinstance(extra, dict):
            raise ValueError("chat_template_kwargs must be an object")
        if set(extra) & {"tokenize", "add_generation_prompt", "chat_template",
                          "return_tensors", "return_dict", "continue_final_message"}:
            raise ValueError("Template overrides cannot replace the native template or rendering mode")
        kwargs.update(extra)
    output_limit = item.get("max_new_tokens", model_spec.get("max_new_tokens", config.get("max_new_tokens")))
    input_limit = item.get("max_input_tokens", model_spec.get("max_input_tokens", config.get("max_input_tokens")))
    readout = item.get("readout", model_spec.get("readout", config.get("readout", "probability")))
    if readout not in READOUT_VERSIONS:
        raise ValueError(f"Unsupported readout: {readout}")
    requirements = [close_requirement(source) for source in (config, model_spec, item)]
    if "enable_thinking" in kwargs:
        if type(kwargs["enable_thinking"]) is not bool:
            raise ValueError("enable_thinking must be Boolean")
        # The Qwen off template closes thinking in the prompt, not generation.
        required = kwargs["enable_thinking"]
        if requirements[-1] is not None and requirements[-1] != required:
            raise ValueError("Item reasoning-close requirement conflicts with enable_thinking")
    else:
        required = next((value for value in reversed(requirements) if value is not None), False)
    return GenerationProfile(positive_int("max_new_tokens", output_limit),
                             positive_int("max_input_tokens", input_limit),
                             canonical_json(kwargs), required, readout)


def group_profiles(rows, config, model_spec):
    groups = OrderedDict()
    for item in rows:
        groups.setdefault(effective_profile(config, model_spec, item), []).append(item)
    return list(groups.items())


def item_messages(item):
    if "messages" in item:
        messages = item["messages"]
    else:
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Item {item.get('id')} requires messages or a prompt")
        messages = [{"role": "user", "content": prompt}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if (not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)):
            raise ValueError("Text messages require a system, user, or assistant role and string content")
    # Forward exactly the supplied history. A native HF template may supply its
    # own default system text, but this runner never inserts one.
    return messages


def validate_context_budget(input_tokens, output_limit, context_limit):
    for name, value in (("input_tokens", input_tokens), ("max_new_tokens", output_limit),
                        ("max_position_embeddings", context_limit)):
        positive_int(name, value)
    if input_tokens + output_limit > context_limit:
        raise RuntimeError(f"Input plus output exceeds model context: {input_tokens}+{output_limit}>{context_limit}")


def parse_final(text, readout="probability"):
    """One terminal line and exact keys. No recovery from prose or reasoning."""
    if readout not in READOUT_VERSIONS:
        raise ValueError(f"Unsupported readout: {readout}")
    if not isinstance(text, str) or text.count("FINAL_JSON:") != 1:
        return None
    match = re.search(r"(?:^|\n)[ \t]*FINAL_JSON:[ \t]*(\{[^\r\n]*\})[ \t]*$", text.strip())
    if not match:
        return None

    def unique_keys(pairs):
        value = {}
        for key, entry in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = entry
        return value

    try:
        value = json.loads(match.group(1), object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        return None
    expected = {"p_h1"} if readout == "probability" else {"answer", "p_h1"}
    if not isinstance(value, dict) or set(value) != expected:
        return None
    probability = value["p_h1"]
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1 or not math.isfinite(probability)):
        return None
    answer = value.get("answer")
    if readout == "answer_probability" and (type(answer) is not int or answer not in (0, 1)):
        return None
    return {"probability": float(probability), "parsed_answer": answer}


def after_reasoning(token_ids, close_ids):
    if not close_ids:
        raise ValueError("Empty reasoning-close marker")
    ends = [i + len(close_ids) for i in range(len(token_ids) - len(close_ids) + 1)
            if token_ids[i:i + len(close_ids)] == close_ids]
    return token_ids[ends[-1]:] if ends else None


def classify_output(token_ids, tokenizer, eos_ids, profile):
    stop_at = next((i for i, value in enumerate(token_ids) if value in eos_ids), None)
    terminated = stop_at is not None
    if terminated:
        token_ids = token_ids[:stop_at + 1]
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False) if profile.require_think_close else []
    final_tokens = after_reasoning(token_ids, close_ids) if profile.require_think_close else token_ids
    final = tokenizer.decode(final_tokens, skip_special_tokens=True).strip() if final_tokens is not None else None
    # A valid-looking final line before EOS or before </think> is never parsed.
    parsed = parse_final(final, profile.readout) if terminated and final is not None else None
    status = "truncated" if not terminated else ("ok" if parsed is not None else "invalid")
    metadata = {
        "final_output": final,
        "parsed_answer": parsed["parsed_answer"] if parsed else None,
        "finish_reason": "eos" if terminated else "length",
        "termination_reason": "eos" if terminated else "length",
        "thinking_segment_closed": (final_tokens is not None) if profile.require_think_close else None,
        "reasoning_tokens": (len(token_ids) - len(final_tokens) - len(close_ids))
                            if profile.require_think_close and final_tokens is not None else None,
        "reasoning_nonempty": bool(raw.split("</think>", 1)[0].replace("<think>", "").strip())
                              if profile.require_think_close and final_tokens is not None else None,
        "visible_rationale_before_final": bool(final.split("FINAL_JSON:", 1)[0].strip())
                                          if final and "FINAL_JSON:" in final else None,
    }
    return status, (parsed["probability"] if parsed else None), raw, token_ids, metadata


def model_provenance(model_path, model_spec):
    files = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
                 "special_tokens_map.json", "chat_template.jinja", "model.safetensors.index.json",
                 "MODEL_PROVENANCE.json", "download_manifest.json"):
        path = model_path / name
        if path.is_file():
            files[name] = {"sha256": digest(path)}
    weights = [{"name": p.name, "size": p.stat().st_size, "resolved_path": str(p.resolve())}
               for pattern in ("*.safetensors", "pytorch_model*.bin") for p in sorted(model_path.glob(pattern))]
    return {"model_spec": model_spec, "resolved_path": str(model_path.resolve()),
            "files": files, "weight_files": weights}


def backup_db(db, run_dir):
    temporary = run_dir / "results.backup.pending.sqlite"
    backup = sqlite3.connect(str(temporary))
    try:
        db.backup(backup)
    finally:
        backup.close()
    os.replace(temporary, run_dir / "results.backup.sqlite")


def run(config_path, model_name):
    config_path = Path(config_path).resolve()
    run_dir = config_path.parent
    with worker_lock(run_dir / "worker.lock"):
        config, frozen = validate_frozen(config_path)
        matches = [spec for spec in config["models"] if spec["name"] == model_name]
        if len(matches) != 1:
            raise ValueError(f"Expected one model spec for {model_name}")
        model_spec = matches[0]
        manifest_path = relative_artifact(run_dir, config["manifest"])
        rows = select_rows([json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
                            if line.strip()], model_name)
        # Validate every selected item even on resume, before loading a model.
        for item in rows:
            effective_profile(config, model_spec, item)
            item_messages(item)
        batch_size = positive_int("batch_size", model_spec.get("batch_size", config.get("batch_size")))
        seed = config.get("seed")
        if type(seed) is not int:
            raise ValueError("seed must be an explicit integer")
        store_tokens = config.get("store_token_ids", True)
        if type(store_tokens) is not bool:
            raise ValueError("store_token_ids must be Boolean")
        db = connect_db(run_dir / "results.sqlite")
        try:
            for key in ("config_sha256", "manifest_sha256", "source_hashes", "protocol_hashes"):
                require_metadata(db, key, frozen[key])
            require_metadata(db, "runner_schema", "extension-runner-v1")
            remaining, done = remaining_rows(db, model_name, rows)
            if not remaining:
                print(f"{model_name}: all {len(rows)} selected records already saved.", flush=True)
                return
            random.Random(seed).shuffle(remaining)
            groups = group_profiles(remaining, config, model_spec)
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

            torch.manual_seed(seed)
            torch.set_num_threads(config.get("cpu_threads", 8))
            torch.backends.cuda.matmul.allow_tf32 = False
            if not torch.cuda.is_available():
                raise RuntimeError("This worker requires a CUDA device assigned by the scheduler")
            environment = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                           "accelerate": importlib.metadata.version("accelerate")}
            require_metadata(db, "environment", environment)
            progress = {"model": model_name, "saved": len(done), "total": len(rows), "pid": os.getpid()}
            atomic_json(run_dir / "progress.json", dict(progress, state="loading", updated_at=now()))
            model_path = Path(model_spec["path"]).expanduser().resolve()
            if not model_path.is_dir():
                raise ValueError(f"Local model path does not exist: {model_path}")
            provenance = model_provenance(model_path, model_spec)
            require_metadata(db, f"model_provenance:{model_name}", provenance)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            if not tokenizer.chat_template:
                raise RuntimeError("Checkpoint tokenizer has no native chat template")
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise RuntimeError("Tokenizer requires an explicit padding or EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            loader = model_spec.get("loader", "causal-lm")
            if loader == "image-text-to-text":
                from transformers import AutoModelForImageTextToText
                model_class = AutoModelForImageTextToText
            elif loader == "causal-lm":
                model_class = AutoModelForCausalLM
            else:
                raise ValueError(f"Unsupported loader: {loader}")
            if config.get("dtype", "bfloat16") != "bfloat16":
                raise ValueError("Only the frozen BF16 configuration is supported")
            attention = model_spec.get("attention", config.get("attention", "sdpa"))
            model = model_class.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.bfloat16, device_map={"": 0},
                                                attn_implementation=attention)
            model.eval()
            context_limit = getattr(getattr(model.config, "text_config", model.config), "max_position_embeddings", None)
            eos = model.generation_config.eos_token_id
            if eos is None:
                eos = tokenizer.eos_token_id
            eos_set = set(eos if isinstance(eos, list) else [eos])
            if not eos_set or any(type(value) is not int for value in eos_set):
                raise RuntimeError("An explicit model EOS token is required")
            require_metadata(db, "effective_model_readout", {"context_limit": context_limit, "eos_token_ids": sorted(eos_set),
                                                            "pad_token_id": tokenizer.pad_token_id,
                                                            "chat_template_sha256": text_digest(canonical_json(tokenizer.chat_template))})
            # All remaining prompts pass native rendering and context checks before
            # the first generation. Same-profile padding preserves these budgets.
            rendered = {}
            for profile, items in groups:
                for item in items:
                    prompt = tokenizer.apply_chat_template(item_messages(item), tokenize=False,
                                                           add_generation_prompt=True, **profile.chat_template_kwargs)
                    length = len(tokenizer.encode(prompt, add_special_tokens=False))
                    if length > profile.max_input_tokens:
                        raise RuntimeError(f"Item {item['id']} exceeds frozen input budget: {length}>{profile.max_input_tokens}")
                    validate_context_budget(length, profile.max_new_tokens, context_limit)
                    rendered[item["id"]] = prompt
            atomic_json(run_dir / "progress.json", dict(progress, state="generating", updated_at=now()))
            started = time.monotonic()
            initial_saved = len(done)
            saved_since_backup = 0
            for profile, items in groups:
                generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=profile.max_new_tokens,
                                              eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
                offset = 0
                while offset < len(items):
                    batch = items[offset:offset + batch_size]
                    prompts = [rendered[item["id"]] for item in batch]
                    inputs = generated = None
                    began = time.monotonic()
                    try:
                        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False,
                                           add_special_tokens=False).to("cuda")
                        validate_context_budget(inputs.input_ids.shape[1], profile.max_new_tokens, context_limit)
                        with torch.inference_mode():
                            generated = model.generate(**inputs, generation_config=generation)
                        torch.cuda.synchronize()
                    except torch.OutOfMemoryError:
                        del inputs, generated
                        torch.cuda.empty_cache()
                        if len(batch) == 1:
                            raise
                        batch_size = max(1, len(batch) // 2)
                        print(json.dumps({"event": "oom_reduce_batch", "batch_size": batch_size,
                                          "profile": profile.metadata(), "time": now()}), flush=True)
                        continue
                    duration = time.monotonic() - began
                    for item, input_ids, mask, output_ids, prompt in zip(batch, inputs.input_ids,
                                                                        inputs.attention_mask, generated, prompts):
                        token_ids = output_ids[input_ids.shape[0]:].tolist()
                        status, probability, raw, retained_ids, output_metadata = classify_output(token_ids, tokenizer, eos_set, profile)
                        metadata = dict(item)
                        metadata.update(output_metadata)
                        metadata.update({"rendered_prompt": prompt, "rendered_prompt_sha256": text_digest(prompt),
                                         "actual_batch_size": len(batch), "effective_max_new_tokens": profile.max_new_tokens,
                                         "effective_chat_template_kwargs": profile.chat_template_kwargs,
                                         "require_think_close": profile.require_think_close,
                                         "thinking_enabled": profile.chat_template_kwargs.get("enable_thinking"),
                                         "model_mode": model_spec.get("mode"), "readout": profile.readout,
                                         "readout_version": READOUT_VERSIONS[profile.readout],
                                         "generation_profile": profile.metadata(), "model_provenance": provenance})
                        if store_tokens:
                            metadata["generated_token_ids"] = retained_ids
                        insert_result(db, (model_name + ":" + item["id"], model_name, item["id"], status,
                                           probability, raw, int(mask.sum().item()), len(retained_ids),
                                           duration / len(batch), now(), canonical_json(metadata)))
                        done.add(item["id"])
                    offset += len(batch)
                    saved_since_backup += len(batch)
                    elapsed = time.monotonic() - started
                    completed = len(done) - initial_saved
                    progress.update(saved=len(done), actual_batch_size=batch_size, elapsed_sec=elapsed,
                                    samples_per_sec=completed / elapsed,
                                    eta_model_sec=(len(rows) - len(done)) * elapsed / completed,
                                    peak_gpu_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
                    atomic_json(run_dir / "progress.json", dict(progress, state="running", updated_at=now()))
                    print(json.dumps({"model": model_name, "saved": len(done), "total": len(rows),
                                      "elapsed_sec": round(elapsed, 2), "time": now()}), flush=True)
                    if saved_since_backup >= 128 or len(done) == len(rows):
                        backup_db(db, run_dir)
                        saved_since_backup = 0
                    del generated, inputs
            atomic_json(run_dir / "progress.json", dict(progress, state="model_complete", updated_at=now()))
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    run(args.config, args.model)


if __name__ == "__main__":
    main()
