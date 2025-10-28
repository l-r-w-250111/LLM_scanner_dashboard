"""
All-in-one LLM/Embedding environment scanner (Windows/Linux).
Features:
 - CPU / Memory (psutil)
 - GPU (nvidia-smi via subprocess)
 - Local process scan (psutil)
 - Docker / Podman container listing (CLI)
 - Async localhost port scan + API probe to extract model names
 - Streamlit UI
"""

import streamlit as st
import psutil
import subprocess
import json
import asyncio
import aiohttp
import socket
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# -----------------------
# Configuration
# -----------------------
# Ports to try (common LLM server ports + wide range)
PORTS_TO_SCAN = list(range(5000, 5011)) + [7860, 8000, 8080, 11434, 11435, 8085, 5005, 9000]  # add specifics
# Endpoints to probe (common LLM endpoints)
ENDPOINT_PATHS = [
    "/v1/models", "/models", "/api/ps", "/api/models", "/health", "/v1/engines", "/v1/status"
]
# Known signatures for runtime detection (key fragments mapped to runtime)
RUNTIME_SIGNATURES = {
    "ollama": ["ollama", "Ollama"],
    "vllm": ["vllm"],
    "text-generation-inference": ["text-generation-inference", "text_generation_inference"],
    "transformers": ["transformers", "huggingface"],
    "fastchat": ["fastchat"],
    "lmstudio": ["lm-studio", "lmstudio"],
    "text-generation-webui": ["text-generation-webui", "server.py"],
    "embedding": ["sentence_transformers", "instructor", "embedding"]
}

# -----------------------
# Utilities
# -----------------------
def safe_join_cmdline(cmdline):
    if not cmdline:
        return ""
    try:
        return " ".join(map(str, cmdline))
    except Exception:
        return str(cmdline)

def get_all_pids_for_container(container_id: str, tool: str = "docker") -> List[int]:
    """
    Get a list of all host PIDs running inside a container, including all descendants.
    Uses `docker inspect` to get the main PID and `psutil` to traverse the process tree.
    """
    pids = []
    try:
        # 1. Get the main process PID of the container from the host's perspective
        inspect_cmd = [tool, "inspect", "-f", "{{.State.Pid}}", container_id]
        res = subprocess.run(inspect_cmd, capture_output=True, text=True, check=True)
        main_pid = int(res.stdout.strip())

        # 2. Use psutil to find all children of the main PID
        if main_pid > 0:
            parent = psutil.Process(main_pid)
            pids = [parent.pid] + [p.pid for p in parent.children(recursive=True)]
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, psutil.NoSuchProcess):
        # Container might have stopped, tool not found, or process ended.
        pass
    return pids

# -----------------------
# System usage
# -----------------------
def get_system_usage() -> Dict[str, Any]:
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": cpu,
        "mem_percent": mem.percent,
        "total_mem_MB": round(mem.total / 1024 ** 2, 1),
        "available_mem_MB": round(mem.available / 1024 ** 2, 1)
    }

# -----------------------
# GPU usage scan (nvidia-smi)
# -----------------------
def scan_gpu_usage() -> Dict[str, Any]:
    """
    Scans GPU summary and per-process usage using nvidia-smi.
    Returns a dictionary with 'summary' and 'processes' keys.
    """
    gpu_data = {"summary": [], "processes": {}}

    # 1. Get overall GPU summary
    try:
        summary_cmd = [
            "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        summary_res = subprocess.run(summary_cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
        for r in summary_res.stdout.splitlines():
            if not r.strip(): continue
            parts = [p.strip() for p in r.split(",")]
            if len(parts) >= 5:
                gpu_data["summary"].append({
                    "index": int(parts[0]), "name": parts[1], "used_MB": float(parts[2]),
                    "total_MB": float(parts[3]), "util_percent": float(parts[4])
                })
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass # nvidia-smi might not be installed or fails

    # 2. Get per-process GPU memory and utilization usage
    try:
        # Query pid, used_memory, and gpu_utilization for all running processes on GPUs
        pmon_cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory,utilization.gpu", "--format=csv,noheader,nounits"]
        pmon_res = subprocess.run(pmon_cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')

        for line in pmon_res.stdout.splitlines():
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    pid = int(parts[0])
                    gpu_mem_mb = float(parts[1])
                    gpu_util = float(parts[2])
                    # Store it in a PID-keyed dictionary for easy lookup
                    gpu_data["processes"][pid] = {"gpu_mem_mb": gpu_mem_mb, "gpu_util_percent": gpu_util}
                except ValueError:
                    continue
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return gpu_data

# -----------------------
# Local process scan
# -----------------------
def scan_local_processes(rules: Dict[str, List[str]]=None) -> List[Dict[str, Any]]:
    results = []
    # Add 'cpu_percent' and 'memory_info' to the iterator
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'exe', 'cpu_percent', 'memory_info']):
        try:
            # Get basic info
            proc_info = proc.info
            pid = proc_info.get('pid')
            cmd = safe_join_cmdline(proc_info.get('cmdline'))
            name = proc_info.get('name') or ""
            exe = proc_info.get('exe') or ""

            # Get resource usage
            # The first call to cpu_percent may be 0.0, but it's okay for a snapshot.
            cpu_percent = proc_info.get('cpu_percent', 0.0)
            memory_info = proc_info.get('memory_info')
            mem_mb = round(memory_info.rss / 1024**2, 2) if memory_info else 0.0

            started = ""
            try:
                started = datetime.fromtimestamp(proc_info.get('create_time')).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                started = ""

            # Match against runtime signatures
            hit_runtime = None
            confidence = 0.0
            search_str = f"{cmd} {name} {exe}".lower()
            for runtime, sigs in RUNTIME_SIGNATURES.items():
                if any(s.lower() in search_str for s in sigs):
                    hit_runtime = runtime
                    confidence = 0.7
                    break

            results.append({
                "pid": pid,
                "name": name,
                "cpu_percent": cpu_percent,
                "mem_mb": mem_mb,
                "exe": exe,
                "cmd": cmd,
                "started": started,
                "runtime_guess": hit_runtime,
                "confidence": confidence
            })
        except (psutil.AccessDenied, psutil.ZombieProcess, psutil.NoSuchProcess):
            continue
        except Exception:
            continue
    return results

# -----------------------
# Docker / Podman listing (CLI)
# -----------------------
def scan_containers() -> List[Dict[str, Any]]:
    all_containers = {}
    # Get CPU core count for normalization
    cpu_cores = psutil.cpu_count() or 1 # Avoid division by zero

    for tool in ("docker", "podman"):
        try:
            # 1. Get basic container info with 'ps'
            ps_res = subprocess.run([tool, "ps", "--format", "{{json .}}"], capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            for line in ps_res.stdout.splitlines():
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    cid = data.get("ID", "")[:12]
                    if cid:
                        # Parse host ports from port mapping string
                        port_info = data.get("Ports", "")
                        host_ports = []
                        if isinstance(port_info, str) and port_info:
                            mappings = port_info.split(',')
                            for mapping in mappings:
                                try:
                                    host_part = mapping.split('->')[0]
                                    if ':' in host_part:
                                        port_str = host_part.split(':')[-1]
                                        if port_str:
                                            host_ports.append(int(port_str))
                                except (ValueError, IndexError):
                                    continue

                        all_containers[cid] = {
                            "engine": tool, "id": cid, "image": data.get("Image", ""),
                            "command": data.get("Command", ""), "status": data.get("Status", ""),
                            "names": data.get("Names", ""), "ports": list(set(host_ports))
                        }
                except json.JSONDecodeError:
                    continue

            # 2. Get resource stats with 'stats'
            stats_res = subprocess.run([tool, "stats", "--no-stream", "--format", "{{json .}}"], capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            for line in stats_res.stdout.splitlines():
                if not line.strip(): continue
                try:
                    stats = json.loads(line)
                    cid = stats.get("ID", "")[:12]
                    if cid in all_containers:
                        # Normalize CPU percentage
                        raw_cpu_perc = float(stats.get("CPUPerc", "0.0").replace('%', '').strip())
                        normalized_cpu = round(raw_cpu_perc / cpu_cores, 2)

                        # Parse Memory
                        mem_mb = 0.0
                        mem_str = stats.get("MemUsage", "").split('/')[0].strip()
                        if "GiB" in mem_str:
                            mem_mb = float(mem_str.replace("GiB", "").strip()) * 1024
                        elif "MiB" in mem_str:
                            mem_mb = float(mem_str.replace("MiB", "").strip())

                        all_containers[cid]["cpu_percent_normalized"] = normalized_cpu
                        all_containers[cid]["mem_mb"] = round(mem_mb, 2)
                except (json.JSONDecodeError, ValueError):
                    continue

        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    return list(all_containers.values())

# -----------------------
# Ollama model scan via docker exec
# -----------------------
def scan_ollama_models_in_container(container_id: str) -> List[Dict[str, Any]]:
    """
    Executes 'ollama ps' inside a container to get the list of running models.
    Tries JSON format first, then falls back to parsing plain text for older versions.
    """
    # 1. Try with --format json (for newer ollama versions)
    try:
        cmd_json = ["docker", "exec", container_id, "ollama", "ps", "--format", "json"]
        res = subprocess.run(cmd_json, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')

        models = []
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        for line in lines:
            try:
                data = json.loads(line)
                if "models" in data and isinstance(data["models"], list):
                    models.extend(data["models"])
                else:
                    models.append(data)
            except json.JSONDecodeError:
                continue
        return models
    except subprocess.CalledProcessError as e:
        # If the error is "unknown flag", we fall back. Otherwise, we fail.
        if "unknown flag" not in e.stderr.lower():
            return [] # It's a different error, so we can't proceed.
    except (FileNotFoundError, json.JSONDecodeError):
        # Docker not found or JSON was invalid for some reason
        return []

    # 2. Fallback to plain text parsing (for older ollama versions)
    try:
        cmd_text = ["docker", "exec", container_id, "ollama", "ps"]
        res = subprocess.run(cmd_text, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')

        models = []
        lines = res.stdout.strip().splitlines()

        # Ensure there's a header and at least one data line
        if len(lines) < 2:
            return []

        header_line = lines[0].upper()
        headers = re.split(r'\s{2,}', header_line)

        try:
            # Find the index of NAME and PROCESSOR columns
            name_idx = headers.index("NAME")
            proc_idx = headers.index("PROCESSOR")
        except ValueError:
            return [] # Essential columns are missing

        for line in lines[1:]:
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) > max(name_idx, proc_idx):
                model_name = parts[name_idx]
                processor_info = parts[proc_idx]

                model_data = {"name": model_name}
                if "GPU" in processor_info.upper():
                    model_data["processor"] = "GPU"
                else:
                    model_data["processor"] = "CPU"
                models.append(model_data)
        return models
    except (subprocess.CalledProcessError, FileNotFoundError):
        # This fallback command also failed.
        return []

# -----------------------
# Async local port + endpoint probe
# -----------------------
async def probe_endpoint(session: aiohttp.ClientSession, base_url: str, path: str, timeout_s: float=3.0) -> Optional[Dict[str, Any]]:
    url = base_url + path
    try:
        async with session.get(url, timeout=timeout_s) as resp:
            ct = resp.headers.get("Content-Type", "")
            text = await resp.text(encoding='utf-8', errors='replace')
            # Try JSON
            try:
                j = await resp.json()
            except Exception:
                j = None
            return {"url": url, "status": resp.status, "text": text, "json": j, "content_type": ct}
    except Exception as e:
        return None

def build_local_urls(host: str="127.0.0.1", ports: List[int]=None):
    if ports is None:
        ports = PORTS_TO_SCAN
    urls = []
    for p in ports:
        base = f"http://{host}:{p}"
        for path in ENDPOINT_PATHS:
            urls.append((base, path))
    return urls

async def async_scan_endpoints(ports: List[int]=None, concurrency: int=50):
    urls = build_local_urls(ports=ports)
    connector = aiohttp.TCPConnector(limit=concurrency, force_close=True)
    timeout = aiohttp.ClientTimeout(total=5)
    results = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for base, path in urls:
            tasks.append(probe_endpoint(session, base, path))
        # run in batches to avoid overwhelming
        for fut_chunk in [tasks[i:i+concurrency] for i in range(0, len(tasks), concurrency)]:
            chunk_res = await asyncio.gather(*fut_chunk, return_exceptions=True)
            for r in chunk_res:
                if isinstance(r, dict):
                    results.append(r)
    return results

# -----------------------
# Model name extraction logic
# -----------------------
def extract_model_info_from_probe(probe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not probe:
        return None
    # prefer JSON
    j = probe.get("json")
    text = probe.get("text") or ""
    url = probe.get("url")
    status = probe.get("status")
    # Common patterns:
    # - /v1/models returns {"data":[{"id":"..."}]}  or list of models
    # - Ollama /api/ps might return model names
    # - TGI returns model list
    candidate = None
    confidence = 0.0
    if j:
        # try to find model keys
        # 1) data[*].id or name
        if isinstance(j, dict):
            # check if "data" key
            if "data" in j and isinstance(j["data"], list):
                # try extract ids
                ids = []
                for item in j["data"]:
                    if isinstance(item, dict):
                        for key in ("id","name","model"):
                            if key in item:
                                ids.append(item[key])
                if ids:
                    candidate = ids[0]
                    confidence = 0.9
            # check for keys containing "model" or "models"
            for k, v in j.items():
                if k.lower().find("model") != -1:
                    # if value is string/list/dict
                    if isinstance(v, str):
                        candidate = v
                        confidence = max(confidence, 0.8)
                    elif isinstance(v, list) and v:
                        if isinstance(v[0], str):
                            candidate = v[0]
                            confidence = max(confidence, 0.8)
        elif isinstance(j, list) and j:
            # list of model objects or strings
            if isinstance(j[0], str):
                candidate = j[0]
                confidence = 0.85
            elif isinstance(j[0], dict):
                for key in ("id", "name", "model"):
                    if key in j[0]:
                        candidate = j[0].get(key)
                        confidence = 0.85
    # fallback text heuristics
    if not candidate and text:
        # quick heuristics: look for words like "model", "loaded", "llama"
        low = text.lower()
        for kw in ["model:", "loaded model", "model_name", "llama", "mistral", "gpt"]:
            if kw in low:
                # pick a short snippet
                idx = low.find(kw)
                snippet = text[idx:idx+120]
                candidate = snippet.strip().splitlines()[0][:120]
                confidence = 0.4
                break
    if candidate:
        return {"url": url, "status": status, "model": str(candidate), "confidence": confidence}
    return None

# -----------------------
# High-level scan orchestration
# -----------------------
def run_full_scan(ports: List[int]=None) -> Dict[str, Any]:
    # 1. System and resource scans (Processes, Containers, GPU)
    usage = get_system_usage()
    procs = scan_local_processes()
    conts = scan_containers()
    gpu_data = scan_gpu_usage()

    # 2. Merge GPU process data into the main process list for direct VRAM/GPU correlation
    gpu_processes = gpu_data.get("processes", {})
    for proc in procs:
        pid = proc.get("pid")
        if pid in gpu_processes:
            proc.update(gpu_processes[pid])

    # 3. Build a map of {container_id: [pids]} using the new reliable helper
    container_to_pids = {c['id']: get_all_pids_for_container(c['id'], c['engine']) for c in conts}

    # Invert the map for faster {pid: container_id} lookups
    pid_to_container = {pid: cid for cid, pids in container_to_pids.items() for pid in pids}

    # 4. Aggregate GPU data per container based on this reliable mapping
    container_gpu_usage = {}
    gpu_pids_on_host = gpu_data.get("processes", {}).keys()

    for pid in gpu_pids_on_host:
        cid = pid_to_container.get(pid)
        if cid:
            if cid not in container_gpu_usage:
                container_gpu_usage[cid] = {"gpu_mem_mb": 0, "gpu_util_percent": 0}

            proc_gpu = gpu_data["processes"][pid]
            container_gpu_usage[cid]["gpu_mem_mb"] += proc_gpu.get("gpu_mem_mb", 0)
            container_gpu_usage[cid]["gpu_util_percent"] += proc_gpu.get("gpu_util_percent", 0)

    # 5. Merge the aggregated GPU data into the main container list
    for c in conts:
        cid = c["id"]
        gpu_usage = container_gpu_usage.get(cid, {"gpu_mem_mb": 0, "gpu_util_percent": 0})
        c["gpu_mem_mb"] = gpu_usage["gpu_mem_mb"]
        c["gpu_util_percent"] = gpu_usage["gpu_util_percent"]

    # 6. Create a mapping from listening port to PID for later association
    port_to_pid = {}
    try:
        # We only care about TCP listening ports
        connections = psutil.net_connections(kind='inet')
        for conn in connections:
            if conn.status == psutil.CONN_LISTEN and conn.laddr.port in (ports or PORTS_TO_SCAN):
                port_to_pid[conn.laddr.port] = conn.pid
    except Exception:
        pass # Best effort

    # 7. Async endpoint scanning + initial model extraction
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        probe_results = loop.run_until_complete(async_scan_endpoints(ports=ports))
    finally:
        try:
            loop.close()
        except Exception:
            pass

    # This dictionary will hold the final, merged model information
    final_models = {}

    # 8. Identify Ollama containers and their ports *before* processing HTTP probes
    ollama_containers = [c for c in conts if "ollama" in c.get("image", "").lower() or "ollama" in c.get("names", "").lower()]
    ollama_ports = set()
    for c in ollama_containers:
        for port in c.get("ports", []):
            ollama_ports.add(port)

    # 9. Process HTTP Probe results and associate with processes
    for p in probe_results:
        info = extract_model_info_from_probe(p)
        if info and info.get("model"):
            model_name = str(info["model"]).strip()
            info["source"] = "http_probe"

            # Associate with a process and its resources via port number
            try:
                port = int(info.get("url", "").split(":")[2].split("/")[0])

                # CRITICAL FIX: If this port belongs to an Ollama container,
                # ignore the HTTP probe result. We will get the definitive
                # model name from `docker exec` later.
                if port in ollama_ports:
                    continue

                pid = port_to_pid.get(port)
                if pid:
                    # Find the process in our scanned list
                    proc = next((pr for pr in procs if pr["pid"] == pid), None)
                    if proc:
                        info["pid"] = pid
                        info["CPU Util (%)"] = proc.get("cpu_percent", 0)
                        info["RAM (MB)"] = proc.get("mem_mb", 0)
                        info["VRAM (MB)"] = proc.get("gpu_mem_mb", 0)
            except (IndexError, ValueError):
                pass # Can't parse port

            final_models[model_name] = info

    # 10. Scan inside Ollama containers and associate with container resources
    ollama_containers = [c for c in conts if "ollama" in c.get("image", "").lower() or "ollama" in c.get("names", "").lower()]
    for container in ollama_containers:
        cid = container.get("id")
        if not cid: continue

        models_in_container = scan_ollama_models_in_container(cid)
        is_single_model = len(models_in_container) == 1

        for model_info in models_in_container:
            model_name = model_info.get("name")
            if model_name:
                # 1. Determine VRAM usage with enhanced logic
                vram_mb = 0
                # Priority 1: Use per-model VRAM from `ollama ps` if available (most accurate)
                if model_info.get("size_vram", 0) > 0:
                    vram_mb = round(model_info["size_vram"] / (1024**2), 2)
                # Priority 2: If only one model is running, attribute all container VRAM to it.
                elif is_single_model:
                    vram_mb = container.get("gpu_mem_mb", 0)
                # Fallback: For multiple models without per-model stats, show container total (less accurate)
                else:
                    vram_mb = container.get("gpu_mem_mb", 0)

                # 2. Determine Execution State with updated priority
                execution_state = "CPU" # Default

                # Priority 1: Check for 'processor' info from enhanced text parsing
                if model_info.get("processor") == "GPU":
                    execution_state = "GPU"
                else:
                    # Priority 2: Check for 'details' from JSON output
                    details = model_info.get("details", {})
                    if details:
                        families = details.get("families", [])
                        if families and "gpu" in str(families).lower():
                            execution_state = "GPU"
                        param_size = details.get("parameter_size", "")
                        if param_size and "B" in param_size and "/" in param_size:
                            execution_state = f"Mixed ({param_size})"

                    # Priority 3: Fallback inference based on container GPU memory
                    elif gpu_data.get("summary") and vram_mb > 0:
                        execution_state = "GPU (Inferred)"

                # 3. Get Model Size
                model_size_gb = round(model_info.get("size", 0) / (1024**3), 2) if model_info.get("size") else 0.0

                # 4. Overwrite any http probe result with this high-confidence data
                final_models[model_name] = {
                    "model": model_name,
                    "source": "docker_exec",
                    "container": container.get("names", cid),
                    "confidence": 1.0,
                    "CPU Util (%)": container.get("cpu_percent_normalized", 0),
                    "RAM (MB)": container.get("mem_mb", 0),
                    "VRAM (MB)": vram_mb,
                    "GPU Util (%)": container.get("gpu_util_percent", 0),
                    "Model Size (GB)": model_size_gb,
                    "Execution": execution_state
                }

    # 11. Final Fallback: If a single GPU model has 0 VRAM, and there's only one GPU, assign total usage.
    gpu_models_with_zero_vram = [
        m for m in final_models.values()
        if "GPU" in m.get("Execution", "") and m.get("VRAM (MB)", 0) == 0
    ]
    gpu_summary_list = gpu_data.get("summary", [])

    if len(gpu_models_with_zero_vram) == 1 and len(gpu_summary_list) == 1:
        model_to_update = gpu_models_with_zero_vram[0]
        gpu_summary = gpu_summary_list[0]
        
        # Update VRAM and GPU utilization from the overall system summary
        model_to_update["VRAM (MB)"] = gpu_summary.get("used_MB", 0)
        model_to_update["GPU Util (%)"] = gpu_summary.get("util_percent", 0)
        
        # Optionally, update the execution state to show this is an inferred value
        model_to_update["Execution"] = "GPU (System Total)"

    return {
        "usage": usage,
        "processes": procs,
        "containers": conts,
        "gpus": gpu_data.get("summary", []),
        "probes": probe_results,
        "models": list(final_models.values()),
        "scanned_at": datetime.now().isoformat()
    }

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="LLM Universal Scanner", layout="wide")
st.title("🧭 LLM / Embedding Universal Scanner (Local)")
st.markdown("Scan local machine for LLM runtimes, containers and local HTTP model endpoints. "
            "This tool only probes localhost and local resources (no external network calls).")

with st.sidebar:
    st.header("Settings")
    port_input = st.text_area("Extra ports (comma separated)", value="")
    if port_input.strip():
        try:
            extra_ports = [int(x.strip()) for x in port_input.split(",") if x.strip()]
        except Exception:
            extra_ports = []
    else:
        extra_ports = []
    scan_ports = PORTS_TO_SCAN + extra_ports
    st.write(f"Ports to scan: {len(scan_ports)} ports")

    if st.button("🔄 Run Full Scan now"):
        st.session_state["run_scan"] = True

# Run scan (immediately on first load or when triggered)
if "run_scan" not in st.session_state:
    st.session_state["run_scan"] = True

if st.session_state.get("run_scan"):
    with st.spinner("Running full scan (system, processes, containers, GPU, endpoints)..."):
        try:
            scan_result = run_full_scan(ports=scan_ports)
        except Exception as e:
            st.error(f"Scan failed: {e}")
            scan_result = {"usage": {}, "processes": [], "containers": [], "gpus": [], "models": [], "scanned_at": datetime.now().isoformat()}

    st.session_state["last_scan"] = scan_result
    st.session_state["run_scan"] = False

# Show results
res = st.session_state.get("last_scan", {})
st.subheader("System usage")
usage = res.get("usage", {})
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**CPU %:** {usage.get('cpu_percent', 0):.1f}%")
with col2:
    st.markdown(f"**Memory %:** {usage.get('mem_percent', 0):.1f}%")
with col3:
    st.markdown(f"**Total RAM (MB):** {usage.get('total_mem_MB', 0):.1f}")

st.subheader("Discovered Models and Resource Usage")
models = res.get("models", [])
if models:
    st.dataframe(models, width='stretch')
else:
    st.info("No model endpoints detected in scanned ports.")

st.subheader("GPU info (via nvidia-smi)")
gpus = res.get("gpus", [])
if gpus:
    st.dataframe(gpus, width='stretch')
else:
    st.info("No GPU info available or nvidia-smi not present.")

st.subheader("Detected Candidate Processes and Resources")
procs = res.get("processes", [])
# Add some default columns for better display if they are missing
if procs:
    # A bit of data cleaning for display
    for p in procs:
        p['gpu_mem_mb'] = p.get('gpu_mem_mb', 0)
    st.dataframe(procs, width='stretch')
else:
    st.info("No candidate processes found.")

st.subheader("Container Info and Resources")
conts = res.get("containers", [])
if conts:
    st.dataframe(conts, width='stretch')
else:
    st.info("No containers detected or Docker/Podman CLI not available.")

st.markdown("---")
st.write(f"Last scanned at: {res.get('scanned_at')}")
# allow manual re-run
if st.button("Run scan again"):
    st.session_state["run_scan"] = True
    st.rerun()
