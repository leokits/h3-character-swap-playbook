import json
import sys
import time
import urllib.error
import urllib.request

API = "http://127.0.0.1:8190"


def run(wf, wait=180):
    req = urllib.request.Request(f"{API}/prompt", data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            pid = json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        print("HTTP", e.code)
        print(e.read().decode()[:2000])
        return None
    print("pid", pid)
    t0 = time.time()
    while time.time() - t0 < wait:
        time.sleep(3)
        with urllib.request.urlopen(f"{API}/history/{pid}", timeout=20) as r:
            h = json.load(r)
        if pid in h:
            st = h[pid].get("status", {})
            print("status", st.get("status_str"))
            if st.get("status_str") == "error":
                for m in st.get("messages", []):
                    if m and m[0] == "execution_error":
                        print("ERR", json.dumps(m[1])[:800])
            print("outputs", json.dumps(h[pid].get("outputs", {}))[:800])
            return h[pid]
    print("timeout")
    return None
