"""Check the last steps before a training checkpoint for signs of a motor fault."""
import json, os, sys
import numpy as np

STUCK_LEAD_DEG, STUCK_DQ, BLOCK = 6.0, 0.03, 1000
ONE_JOINT_SHARE, MANY_JOINT_SHARE, MANY_JOINTS = 0.30, 0.15, 3
WINDOW = 5000


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(3)
    run, step = sys.argv[1], sys.argv[2]
    buf = os.path.join(run, "buffer" if step == "final" else f"buffer_step{step}")
    if not os.path.exists(buf):
        print(f"[check] no {buf}"); sys.exit(3)
    import torch
    b = torch.load(buf, map_location="cpu", weights_only=False)
    n, cur, m = int(b["num_in_buffer"]), int(b["current_idx"]), int(b["observation"].shape[0])
    if n <= 0:
        print("[check] empty buffer"); sys.exit(3)
    end = cur if (0 < cur < n or n == m) else n
    idx = (np.arange(end - min(WINDOW, n), end) % m) if n == m else np.arange(max(0, end - WINDOW), end)
    no = np.asarray(b["next_observation"])[idx].astype(np.float64)
    lead = np.degrees(no[:, 65:72]); dq = no[:, 7:14]
    stuck = (np.abs(lead) > STUCK_LEAD_DEG) & (np.abs(dq) < STUCK_DQ)
    bad, lines = [], []
    for s in range(0, len(idx), BLOCK):
        sh = stuck[s:s + BLOCK].mean(0)
        lines.append(f"    rows {s:5d}-{min(s + BLOCK, len(idx)):5d}: stuck " + " ".join(f"J{j+1} {100*sh[j]:3.0f}%" for j in range(7)))
        if sh.max() > ONE_JOINT_SHARE:
            bad.append(f"J{int(sh.argmax())+1} stuck behind its command in {100*sh.max():.0f}% of rows {s}-{s+BLOCK} of the window")
        if int((sh > MANY_JOINT_SHARE).sum()) >= MANY_JOINTS:
            bad.append(f"{int((sh > MANY_JOINT_SHARE).sum())} joints stuck in > {100*MANY_JOINT_SHARE:.0f}% of rows {s}-{s+BLOCK} (chain dropout signature)")
    ep_path = os.path.join(run, "episodes.jsonl")
    if os.path.exists(ep_path):
        E = [json.loads(l) for l in open(ep_path)]
        if step != "final":
            with open(os.path.join(run, "config.json")) as f:
                off = json.load(f)["agent"]["step_offset"]
            c = np.cumsum([e["l"] for e in E]); E = [e for e, cc in zip(E, c) if cc <= int(step) - off]
        if len(E) >= 30:
            last, prev = [e["min_dist"] for e in E[-10:]], [e["min_dist"] for e in E[-30:-10]]
            ml, mp = 1000 * float(np.median(last)), 1000 * float(np.median(prev))
            lines.append(f"    episodes: median min_dist last 10 = {ml:.0f} mm, the 20 before = {mp:.0f} mm")
            if ml > 100 and ml > 3 * mp:
                bad.append(f"closest approach collapsed: last 10 episodes {ml:.0f} mm vs {mp:.0f} mm before")
    print(f"[check] {os.path.basename(run)} step {step}: {len(idx)} most recent rows")
    print("\n".join(lines))
    if bad:
        print("[check] ABNORMAL:\n  - " + "\n  - ".join(bad)); sys.exit(2)
    print("[check] normal"); sys.exit(0)


if __name__ == "__main__":
    main()
