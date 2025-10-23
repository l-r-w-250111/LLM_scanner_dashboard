# file: llm_scanner_dashboard.py
import streamlit as st
import psutil
import subprocess
import json
from datetime import datetime
from pathlib import Path


import yaml
from pathlib import Path


# =============================
# LLM / Embedding Detection Rules
# =============================
DEFAULT_RULES = {
    "rules": [
        {"name": "Ollama", "keywords": ["ollama"]},
        {"name": "vLLM", "keywords": ["vllm"]},
        {"name": "llama.cpp", "keywords": ["llama", "gguf"]},
        {"name": "FastChat", "keywords": ["fastchat"]},
        {"name": "TextGen WebUI", "keywords": ["server.py", "text-generation-webui"]},
        {"name": "Embedding", "keywords": ["embedding", "sentence_transformers"]},
        {"name": "Custom LLM Server", "keywords": ["generate", "fastapi"]},
    ]
}

rules_path = Path("scan_rules.yaml")
if rules_path.exists():
    with rules_path.open("r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
else:
    rules = DEFAULT_RULES


# =============================
# GPU Information (nvidia-smi subprocess)
# =============================
def scan_gpu_processes_nvsmi():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            index, name, used, total = line.split(", ")
            gpus.append({
                "index": int(index),
                "name": name,
                "used_MB": float(used),
                "total_MB": float(total)
            })
        return gpus
    except Exception as e:
        return [{"error": str(e)}]

# =============================
# Local Process Scan
# =============================
def scan_local_processes():
    results = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info.get('cmdline')
            if not cmdline:
                cmd = ""
            else:
                cmd = " ".join(map(str, cmdline))
            pid = proc.info['pid']
            started = datetime.fromtimestamp(proc.info['create_time']).strftime("%Y-%m-%d %H:%M:%S")
            for rule in rules["rules"]:
                if any(k in cmd for k in rule["keywords"]):
                    results.append({
                        "type": rule["name"],
                        "pid": pid,
                        "cmd": cmd,
                        "started": started
                    })
        except (psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return results

# =============================
# Docker / Podman Scan
# =============================
def scan_containers():
    containers = []
    for tool in ["docker", "podman"]:
        try:
            result = subprocess.run(
                [tool, "ps", "--format", "{{json .}}"],
                capture_output=True, text=True
            )
            for line in result.stdout.strip().splitlines():
                if not line:
                    continue
                data = json.loads(line)
                containers.append({
                    "engine": tool,
                    "id": data.get("ID", "")[:12],
                    "image": data.get("Image", ""),
                    "command": data.get("Command", ""),
                    "status": data.get("Status", ""),
                    "names": data.get("Names", "")
                })
        except FileNotFoundError:
            pass
    return containers

# =============================
# Integrated Scan
# =============================
def full_scan():
    local = scan_local_processes()
    gpu = scan_gpu_processes_nvsmi()
    containers = scan_containers()
    return local, gpu, containers

# =============================
# Streamlit Dashboard
# =============================
st.set_page_config(page_title="LLM Environment Scanner", layout="wide")
st.title("🧭 LLM / Embedding Environment Scanner")
st.caption("Automatically scans and estimates the operating status of LLM, Embedding, and Docker/Podman environments.")

local, gpu, containers = full_scan()

st.subheader("🧩 Local Process Detection Results")
if local:
    st.dataframe(local, width='stretch')
else:
    st.info("No relevant processes were detected.")

st.subheader("🧠 GPU Usage (nvidia-smi)")
if gpu:
    st.dataframe(gpu, width='stretch')
else:
    st.info("Could not retrieve GPU information.")

st.subheader("📦 Docker / Podman Containers")
if containers:
    st.dataframe(containers, width='stretch')
else:
    st.info("No running Docker / Podman containers found.")

# System Resource Usage
cpu = psutil.cpu_percent(interval=0.3)
mem = psutil.virtual_memory().percent
col1, col2 = st.columns(2)
col1.metric("CPU Usage", f"{cpu:.1f}%")
col2.metric("Memory Usage", f"{mem:.1f}%")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("🔄 Refresh"):
    st.rerun()
