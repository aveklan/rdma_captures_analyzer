#!/usr/bin/env python3
"""
High-level Pipeline GPU Timeline Analyzer (RoCE pcap)
=====================================================

This script reconstructs bulk RDMA WRITE messages from pcap traffic and builds
GPU-level timelines per training step with forward/backward separation.

Outputs:
- tensor_transfer_table.csv: reconstructed bulk transfers (tensor-like units)
- gpu_step_timeline.csv: per-step, per-pass, per-GPU recv/compute/send windows
- gpu_step_summary.csv: aggregated communication/computation breakdown
- plots:
  - 01_gpu_timeline_step_<N>.png (Gantt-style, forward/backward)
  - 02_gpu_comm_vs_compute_breakdown.png
  - 03_pipeline_transfer_view.png

Requirements:
    pip install numpy pandas matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_pipeline_roce import (
    PacketParser,
    PcapReader,
    RDMA_WRITE_OPCODES,
    WRITE_FIRST_OPCODES,
    WRITE_LAST_OPCODES,
    WRITE_MIDDLE_OPCODES,
    WRITE_ONLY_OPCODES,
)


@dataclass
class Config:
    input_pcap: str
    output_dir: str
    start_ip: str = "192.168.2.1"
    end_ip: str = "192.168.2.7"
    window_mode: str = "steps"
    auto_window_sec: float = 4.0
    start_step: int = 1
    max_steps: int = 1
    active_bin_ms: float = 10.0
    bitrate_bin_ms: float = 1.0
    bulk_min_packets: int = 16
    bulk_min_bytes: int = 131072


class HighLevelPipelineAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _expand_ip_range(start_ip: str, end_ip: str) -> List[str]:
        sp = start_ip.split(".")
        ep = end_ip.split(".")
        if len(sp) != 4 or len(ep) != 4:
            return []
        if sp[:3] != ep[:3]:
            return []
        a, b = int(sp[3]), int(ep[3])
        if a > b:
            a, b = b, a
        prefix = ".".join(sp[:3])
        return [f"{prefix}.{i}" for i in range(a, b + 1)]

    @staticmethod
    def _safe_goodput(orig_len: float) -> float:
        return float(max(orig_len - 58.0, 0.0))

    @staticmethod
    def _stage_pairs(nodes: Sequence[str]) -> List[Tuple[str, str, int]]:
        out = []
        for i in range(len(nodes) - 1):
            out.append((nodes[i], nodes[i + 1], i + 1))
        return out

    def load_roce_packets(self) -> pd.DataFrame:
        rows = []
        for pkt_idx, (ts, orig_len, raw, link_type) in enumerate(
            PcapReader.read_pcap(self.cfg.input_pcap)
        ):
            info = PacketParser.parse_frame(raw, link_type=link_type)
            if not info.get("is_roce"):
                continue
            rows.append(
                {
                    "pkt_idx": pkt_idx,
                    "timestamp": float(ts),
                    "orig_len": float(orig_len),
                    "goodput_bytes_est": self._safe_goodput(float(orig_len)),
                    "src_ip": info.get("src_ip"),
                    "dst_ip": info.get("dst_ip"),
                    "opcode": info.get("opcode"),
                    "opcode_name": info.get("opcode_name"),
                    "dest_qp": info.get("dest_qp"),
                    "psn": info.get("psn"),
                }
            )

        if not rows:
            raise RuntimeError("No RoCE packets found in input pcap.")

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        df["rel_time"] = df["timestamp"] - float(df["timestamp"].iloc[0])
        df["direction"] = df["src_ip"].astype(str) + " -> " + df["dst_ip"].astype(str)
        return df

    def resolve_nodes(self, df: pd.DataFrame) -> Tuple[List[str], str]:
        req = self._expand_ip_range(self.cfg.start_ip, self.cfg.end_ip)
        if req:
            m = df["src_ip"].isin(req) & df["dst_ip"].isin(req)
            if int(m.sum()) > 100:
                return req, "requested_range"

        prefix = ".".join(self.cfg.start_ip.split(".")[:3])
        ip_counts = pd.concat([df["src_ip"], df["dst_ip"]]).value_counts()
        cand = [ip for ip in ip_counts.index if ip.startswith(prefix + ".")]
        chosen = sorted(cand[:7], key=lambda x: tuple(int(p) for p in x.split(".")))
        if len(chosen) < 2:
            raise RuntimeError("Cannot resolve pipeline nodes from capture.")
        return chosen, "auto_detected_range"

    def filter_pipeline_adjacent(self, df: pd.DataFrame, nodes: Sequence[str]) -> pd.DataFrame:
        pairs = self._stage_pairs(nodes)
        forward = {f"{a} -> {b}" for a, b, _ in pairs}
        backward = {f"{b} -> {a}" for a, b, _ in pairs}
        out = df[df["direction"].isin(forward | backward)].copy()

        node_idx = {ip: i for i, ip in enumerate(nodes)}

        def pass_dir(row: pd.Series) -> str:
            s = node_idx.get(str(row["src_ip"]), -1)
            d = node_idx.get(str(row["dst_ip"]), -1)
            if s >= 0 and d >= 0 and s < d:
                return "forward"
            if s >= 0 and d >= 0 and s > d:
                return "backward"
            return "unknown"

        out["pass_dir"] = out.apply(pass_dir, axis=1)
        return out

    def detect_active_window(self, df: pd.DataFrame) -> Tuple[float, float]:
        bin_w = self.cfg.active_bin_ms / 1000.0
        max_t = float(df["rel_time"].max())
        bins = np.arange(0.0, max_t + bin_w, bin_w)
        if len(bins) < 2:
            return 0.0, min(max_t, self.cfg.auto_window_sec)

        b, _ = np.histogram(df["rel_time"], bins=bins, weights=df["orig_len"])
        mbps = (b * 8.0) / (bin_w * 1e6)
        if len(mbps) == 0:
            return 0.0, min(max_t, self.cfg.auto_window_sec)

        thr = max(float(np.percentile(mbps, 90)), float(np.max(mbps)) * 0.15)
        active = mbps > thr
        start = 0.0
        for i, flag in enumerate(active):
            if flag:
                start = max(0.0, bins[i] - 0.2)
                break
        end = min(max_t, start + self.cfg.auto_window_sec)
        return start, end

    def detect_step_edges(self, df: pd.DataFrame, nodes: Sequence[str]) -> Tuple[List[float], str]:
        # Define one full training step as forward + backward passes.
        pairs = self._stage_pairs(nodes)
        if not pairs:
            s, e = self.detect_active_window(df)
            return [s, e], "time_fallback"

        fwd_ref = f"{pairs[0][0]} -> {pairs[0][1]}"
        bwd_ref = f"{pairs[0][1]} -> {pairs[0][0]}"

        fwd_times = (
            df[df["direction"] == fwd_ref]["rel_time"].sort_values().to_numpy(dtype=float)
        )
        bwd_times = (
            df[df["direction"] == bwd_ref]["rel_time"].sort_values().to_numpy(dtype=float)
        )
        if len(fwd_times) < 2:
            s, e = self.detect_active_window(df)
            return [s, e], "time_fallback"

        ref_df = df[df["direction"].isin([fwd_ref, bwd_ref])].sort_values("rel_time")
        if ref_df.empty:
            s, e = self.detect_active_window(df)
            return [s, e], "time_fallback"

        t = ref_df["rel_time"].to_numpy(dtype=float)
        if len(t) <= 1:
            s, e = self.detect_active_window(df)
            return [s, e], "time_fallback"

        dt = np.diff(t)
        gap_thr = max(0.2, float(np.percentile(dt, 99)) * 10.0)
        idx = [0]
        idx.extend((np.where(dt > gap_thr)[0] + 1).tolist())
        idx = np.asarray(idx, dtype=int)

        clusters: List[Dict[str, object]] = []
        for i, a_idx in enumerate(idx):
            b_idx = idx[i + 1] if i + 1 < len(idx) else len(ref_df)
            c = ref_df.iloc[a_idx:b_idx]
            if c.empty:
                continue
            f_n = int((c["direction"] == fwd_ref).sum())
            b_n = int((c["direction"] == bwd_ref).sum())
            clusters.append(
                {
                    "start": float(c["rel_time"].iloc[0]),
                    "end": float(c["rel_time"].iloc[-1]),
                    "dominant": "F" if f_n >= b_n else "B",
                }
            )

        f_clusters = [c for c in clusters if c["dominant"] == "F"]
        b_clusters = [c for c in clusters if c["dominant"] == "B"]
        if len(f_clusters) < 1 or len(b_clusters) < 1:
            s, e = self.detect_active_window(df)
            return [s, e], "time_fallback"

        f_starts = np.asarray([float(c["start"]) for c in f_clusters], dtype=float)

        cycle = (
            float(np.median(np.diff(f_starts)))
            if len(f_starts) >= 2
            else self.cfg.auto_window_sec
        )
        active_start, _ = self.detect_active_window(df)
        base_idx = int(np.searchsorted(f_starts, active_start, side="left"))
        if base_idx >= len(f_clusters):
            base_idx = len(f_clusters) - 1

        idx0 = base_idx + max(0, self.cfg.start_step - 1)
        if idx0 >= len(f_clusters):
            idx0 = len(f_clusters) - 1

        steps = max(1, self.cfg.max_steps)
        idx_last = min(idx0 + steps - 1, len(f_clusters) - 1)

        def step_end(i: int) -> float:
            s_i = float(f_clusters[i]["start"])
            if i + 1 < len(f_clusters):
                next_s = float(f_clusters[i + 1]["start"])
            else:
                next_s = float("inf")
            cand = [float(c["end"]) for c in b_clusters if float(c["start"]) >= s_i and float(c["start"]) < next_s]
            if cand:
                return float(max(cand))
            if np.isfinite(next_s):
                return float(next_s)
            return float(s_i + cycle)

        boundaries = [float(f_clusters[idx0]["start"])]
        boundaries.extend(float(f_clusters[i]["start"]) for i in range(idx0 + 1, idx_last + 1))
        boundaries.append(step_end(idx_last))

        max_t = float(df["rel_time"].max())
        boundaries[-1] = min(boundaries[-1], max_t)
        if boundaries[-1] <= boundaries[0]:
            s, e = self.detect_active_window(df)
            return [s, e], "time_fallback"

        return boundaries, "steps"

    def reconstruct_bulk_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        src = df[df["opcode"].isin(RDMA_WRITE_OPCODES)].sort_values("rel_time")
        if src.empty:
            return pd.DataFrame(
                columns=[
                    "msg_id",
                    "direction",
                    "pass_dir",
                    "src_ip",
                    "dst_ip",
                    "dest_qp",
                    "start",
                    "end",
                    "duration",
                    "packets",
                    "wire_bytes",
                    "goodput_bytes",
                ]
            )

        records: List[Dict[str, object]] = []
        msg_id = 0

        for direction, g in src.groupby("direction"):
            active: Dict[int, Dict[str, object]] = {}
            done: List[Dict[str, object]] = []

            for _, r in g.iterrows():
                op = int(r["opcode"])
                qp = int(r["dest_qp"]) if pd.notna(r["dest_qp"]) else -1
                t = float(r["rel_time"])

                def new_msg() -> Dict[str, object]:
                    return {
                        "start": t,
                        "end": t,
                        "packets": 0,
                        "wire_bytes": 0.0,
                        "goodput_bytes": 0.0,
                    }

                def add_pkt(m: Dict[str, object]) -> None:
                    m["end"] = t
                    m["packets"] = int(m["packets"]) + 1
                    m["wire_bytes"] = float(m["wire_bytes"]) + float(r["orig_len"])
                    m["goodput_bytes"] = float(m["goodput_bytes"]) + float(r["goodput_bytes_est"])

                if op in WRITE_ONLY_OPCODES:
                    m = new_msg()
                    add_pkt(m)
                    done.append(m)
                    continue

                if op in WRITE_FIRST_OPCODES:
                    m = new_msg()
                    add_pkt(m)
                    active[qp] = m
                    continue

                if op in WRITE_MIDDLE_OPCODES:
                    m = active.get(qp)
                    if m is None:
                        m = new_msg()
                        active[qp] = m
                    add_pkt(m)
                    continue

                if op in WRITE_LAST_OPCODES:
                    m = active.get(qp)
                    if m is None:
                        m = new_msg()
                    add_pkt(m)
                    done.append(m)
                    if qp in active:
                        del active[qp]

            done.extend(active.values())
            if not done:
                continue

            msg_df = pd.DataFrame(done)
            bulk = msg_df[
                (msg_df["packets"] >= int(self.cfg.bulk_min_packets))
                | (msg_df["wire_bytes"] >= float(self.cfg.bulk_min_bytes))
            ].copy()

            # Fallback to robust percentile cut if fixed thresholds are too strict.
            if bulk.empty and not msg_df.empty:
                pkt_thr = max(8.0, float(np.percentile(msg_df["packets"], 90)))
                byte_thr = max(4096.0, float(np.percentile(msg_df["wire_bytes"], 90)))
                bulk = msg_df[
                    (msg_df["packets"] >= pkt_thr) | (msg_df["wire_bytes"] >= byte_thr)
                ].copy()

            if bulk.empty:
                continue

            src_ip, dst_ip = [x.strip() for x in direction.split("->")]
            pass_dir = str(g["pass_dir"].iloc[0]) if "pass_dir" in g.columns else "unknown"
            qp_mode = int(g["dest_qp"].mode(dropna=True).iloc[0]) if g["dest_qp"].notna().any() else -1

            bulk.sort_values("start", inplace=True)
            for _, m in bulk.iterrows():
                msg_id += 1
                records.append(
                    {
                        "msg_id": msg_id,
                        "direction": direction,
                        "pass_dir": pass_dir,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "dest_qp": qp_mode,
                        "start": float(m["start"]),
                        "end": float(m["end"]),
                        "duration": max(float(m["end"]) - float(m["start"]), 0.0),
                        "packets": int(m["packets"]),
                        "wire_bytes": float(m["wire_bytes"]),
                        "goodput_bytes": float(m["goodput_bytes"]),
                    }
                )

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records).sort_values("start").reset_index(drop=True)

    @staticmethod
    def assign_step_ids(
        msgs: pd.DataFrame, step_edges: Sequence[float]
    ) -> pd.DataFrame:
        if msgs.empty:
            return msgs
        if len(step_edges) < 2:
            msgs = msgs.copy()
            msgs["step_id"] = 1
            return msgs
        bins = np.asarray(step_edges, dtype=float)
        step_id = np.digitize(msgs["start"].to_numpy(dtype=float), bins[1:], right=False) + 1
        msgs = msgs.copy()
        msgs["step_id"] = np.clip(step_id, 1, max(1, len(step_edges) - 1))
        return msgs

    @staticmethod
    def build_gpu_timeline(
        msgs: pd.DataFrame, nodes: Sequence[str], step_edges: Sequence[float]
    ) -> pd.DataFrame:
        if msgs.empty:
            return pd.DataFrame(
                columns=[
                    "step_id",
                    "pass_dir",
                    "gpu",
                    "step_start",
                    "step_end",
                    "recv_start",
                    "recv_end",
                    "recv_ms",
                    "compute_start",
                    "compute_end",
                    "compute_ms",
                    "send_start",
                    "send_end",
                    "send_ms",
                    "comm_ms",
                    "step_duration_ms",
                    "compute_ratio_pct",
                    "comm_ratio_pct",
                ]
            )

        rows = []
        max_step = int(msgs["step_id"].max()) if "step_id" in msgs.columns else 1
        pass_dirs = ["forward", "backward"]

        def safe_ms(a: Optional[float], b: Optional[float]) -> float:
            if a is None or b is None:
                return 0.0
            return max((float(b) - float(a)) * 1e3, 0.0)

        for step_id in range(1, max_step + 1):
            if len(step_edges) >= step_id + 1:
                step_start = float(step_edges[step_id - 1])
                step_end = float(step_edges[step_id])
            else:
                sub_s = msgs[msgs["step_id"] == step_id]["start"]
                sub_e = msgs[msgs["step_id"] == step_id]["end"]
                if sub_s.empty or sub_e.empty:
                    continue
                step_start = float(sub_s.min())
                step_end = float(sub_e.max())

            step_dur_ms = max((step_end - step_start) * 1e3, 1e-9)
            step_msgs = msgs[msgs["step_id"] == step_id]

            for pass_dir in pass_dirs:
                smsgs = step_msgs[step_msgs["pass_dir"] == pass_dir]
                for gpu in nodes:
                    incoming = smsgs[smsgs["dst_ip"] == gpu]
                    outgoing = smsgs[smsgs["src_ip"] == gpu]

                    recv_start = float(incoming["start"].min()) if not incoming.empty else None
                    recv_end = float(incoming["end"].max()) if not incoming.empty else None

                    send_source = outgoing
                    if recv_end is not None and not outgoing.empty:
                        after_recv = outgoing[outgoing["start"] >= recv_end]
                        if not after_recv.empty:
                            send_source = after_recv
                        else:
                            overlap = outgoing[outgoing["end"] > recv_end]
                            if not overlap.empty:
                                send_source = overlap

                    send_start = float(send_source["start"].min()) if not send_source.empty else None
                    send_end = float(send_source["end"].max()) if not send_source.empty else None

                    # Inferred compute window: post-recv to pre-send within the same step/pass.
                    comp_start = max(recv_end, step_start) if recv_end is not None else step_start
                    if send_start is not None:
                        comp_end = min(send_start, step_end)
                    else:
                        comp_end = step_end

                    if comp_end < comp_start:
                        comp_start, comp_end = None, None

                    recv_ms = safe_ms(recv_start, recv_end)
                    send_ms = safe_ms(send_start, send_end)
                    comp_ms = safe_ms(comp_start, comp_end)
                    comm_ms = recv_ms + send_ms

                    rows.append(
                        {
                            "step_id": step_id,
                            "pass_dir": pass_dir,
                            "gpu": gpu,
                            "step_start": step_start,
                            "step_end": step_end,
                            "recv_start": recv_start,
                            "recv_end": recv_end,
                            "recv_ms": recv_ms,
                            "compute_start": comp_start,
                            "compute_end": comp_end,
                            "compute_ms": comp_ms,
                            "send_start": send_start,
                            "send_end": send_end,
                            "send_ms": send_ms,
                            "comm_ms": comm_ms,
                            "step_duration_ms": step_dur_ms,
                            "compute_ratio_pct": comp_ms / step_dur_ms * 100.0,
                            "comm_ratio_pct": comm_ms / step_dur_ms * 100.0,
                        }
                    )

        return pd.DataFrame(rows)

    @staticmethod
    def summarize_gpu_timeline(gpu_tl: pd.DataFrame) -> pd.DataFrame:
        if gpu_tl.empty:
            return pd.DataFrame()

        agg = (
            gpu_tl.groupby(["gpu", "pass_dir"], as_index=False)
            .agg(
                steps=("step_id", "nunique"),
                recv_ms_mean=("recv_ms", "mean"),
                compute_ms_mean=("compute_ms", "mean"),
                send_ms_mean=("send_ms", "mean"),
                comm_ms_mean=("comm_ms", "mean"),
                step_duration_ms_mean=("step_duration_ms", "mean"),
                compute_ratio_pct_mean=("compute_ratio_pct", "mean"),
                comm_ratio_pct_mean=("comm_ratio_pct", "mean"),
            )
            .sort_values(["pass_dir", "gpu"])
        )
        return agg

    @staticmethod
    def _save_plot(fig: plt.Figure, path: Path) -> None:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def plot_gpu_timeline_per_step(
        self, gpu_tl: pd.DataFrame, nodes: Sequence[str], step_id: int, out_path: Path
    ) -> None:
        step_df = gpu_tl[gpu_tl["step_id"] == step_id]
        if step_df.empty:
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        pass_order = ["forward", "backward"]
        colors = {"recv": "#4C78A8", "compute": "#F58518", "send": "#54A24B"}
        ymap = {gpu: i for i, gpu in enumerate(nodes)}

        for ax, p in zip(axes, pass_order):
            sub = step_df[step_df["pass_dir"] == p]
            for _, r in sub.iterrows():
                y = ymap[r["gpu"]]
                for key, c in [
                    ("recv", colors["recv"]),
                    ("compute", colors["compute"]),
                    ("send", colors["send"]),
                ]:
                    s = r[f"{key}_start"]
                    e = r[f"{key}_end"]
                    if pd.notna(s) and pd.notna(e) and float(e) > float(s):
                        ax.broken_barh([(float(s), float(e) - float(s))], (y - 0.35, 0.7), facecolors=c)

            ax.set_yticks(list(ymap.values()))
            ax.set_yticklabels(nodes)
            ax.set_ylabel("GPU")
            ax.set_title(f"Step {step_id} - {p.capitalize()} pass")
            ax.grid(True, axis="x", alpha=0.3)

        axes[-1].set_xlabel("Time (s, selected-window relative)")
        handles = [
            plt.Line2D([0], [0], color=colors["recv"], lw=6),
            plt.Line2D([0], [0], color=colors["compute"], lw=6),
            plt.Line2D([0], [0], color=colors["send"], lw=6),
        ]
        axes[0].legend(handles, ["Receive", "Compute (inferred)", "Send"], loc="upper right")
        self._save_plot(fig, out_path)

    def plot_comm_vs_compute_breakdown(self, summary: pd.DataFrame, out_path: Path) -> None:
        if summary.empty:
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        pass_order = ["forward", "backward"]

        for ax, p in zip(axes, pass_order):
            sub = summary[summary["pass_dir"] == p].copy()
            if sub.empty:
                ax.set_axis_off()
                continue
            x = np.arange(len(sub))
            ax.bar(x, sub["compute_ms_mean"], label="Compute", color="#F58518")
            ax.bar(
                x,
                sub["comm_ms_mean"],
                bottom=sub["compute_ms_mean"],
                label="Communication",
                color="#4C78A8",
            )
            ax.set_ylabel("Mean time per step (ms)")
            ax.set_title(f"{p.capitalize()} pass")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["gpu"], rotation=20, ha="right")

        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("GPU")
        self._save_plot(fig, out_path)

    def plot_pipeline_transfer_view(
        self, msgs: pd.DataFrame, nodes: Sequence[str], step_edges: Sequence[float], out_path: Path
    ) -> None:
        if msgs.empty:
            return

        pairs = []
        for i in range(len(nodes) - 1):
            pairs.append(f"{nodes[i]} -> {nodes[i+1]}")
            pairs.append(f"{nodes[i+1]} -> {nodes[i]}")

        pair_set = set(pairs)
        sub = msgs[msgs["direction"].isin(pair_set)].copy()
        if sub.empty:
            return

        ymap = {p: i for i, p in enumerate(pairs)}
        fig, ax = plt.subplots(figsize=(15, 6))
        colors = {"forward": "#4C78A8", "backward": "#E45756", "unknown": "#999999"}

        for _, r in sub.iterrows():
            y = ymap[r["direction"]]
            x = float(r["start"])
            w = max(float(r["duration"]), 1e-6)
            c = colors.get(str(r["pass_dir"]), "#999999")
            ax.broken_barh([(x, w)], (y - 0.35, 0.7), facecolors=c, alpha=0.9)

        for b in step_edges:
            ax.axvline(float(b), color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(pairs)
        ax.set_xlabel("Time (s, selected-window relative)")
        ax.set_title("Pipeline transfer view (bulk tensor-like messages)")
        ax.grid(True, axis="x", alpha=0.3)

        handles = [
            plt.Line2D([0], [0], color="#4C78A8", lw=6),
            plt.Line2D([0], [0], color="#E45756", lw=6),
        ]
        ax.legend(handles, ["Forward", "Backward"], loc="upper right")
        self._save_plot(fig, out_path)

    def run(self) -> None:
        print(f"[*] Loading pcap: {self.cfg.input_pcap}")
        all_df = self.load_roce_packets()

        nodes, node_mode = self.resolve_nodes(all_df)
        print(f"[*] Nodes ({node_mode}): {', '.join(nodes)}")

        pipe_df = self.filter_pipeline_adjacent(all_df, nodes)
        if pipe_df.empty:
            raise RuntimeError("No adjacent pipeline traffic found in selected node range.")

        if self.cfg.window_mode == "steps":
            edges_abs, mode = self.detect_step_edges(pipe_df, nodes)
        else:
            s, e = self.detect_active_window(pipe_df)
            edges_abs, mode = [s, e], "time"

        win_start, win_end = float(edges_abs[0]), float(edges_abs[-1])
        sel = pipe_df[(pipe_df["rel_time"] >= win_start) & (pipe_df["rel_time"] <= win_end)].copy()
        if sel.empty:
            raise RuntimeError("Selected window has no packets.")

        sel["rel_time"] = sel["rel_time"] - win_start
        edges = [float(x - win_start) for x in edges_abs]

        print(
            f"[*] Window mode={mode}, start={win_start:.6f}s, end={win_end:.6f}s, "
            f"duration={win_end-win_start:.6f}s, packets={len(sel):,}"
        )

        msgs = self.reconstruct_bulk_messages(sel)
        if msgs.empty:
            raise RuntimeError("No bulk write messages reconstructed in selected window.")

        msgs = self.assign_step_ids(msgs, edges)
        msgs["duration_ms"] = msgs["duration"] * 1e3
        msgs["wire_mb"] = msgs["wire_bytes"] / 1e6
        msgs["goodput_mb"] = msgs["goodput_bytes"] / 1e6

        gpu_tl = self.build_gpu_timeline(msgs, nodes, edges)
        summary = self.summarize_gpu_timeline(gpu_tl)

        # Export CSVs.
        msgs.to_csv(self.out / "tensor_transfer_table.csv", index=False)
        gpu_tl.to_csv(self.out / "gpu_step_timeline.csv", index=False)
        summary.to_csv(self.out / "gpu_step_summary.csv", index=False)

        # Plots.
        for step_id in sorted(gpu_tl["step_id"].unique()) if not gpu_tl.empty else []:
            self.plot_gpu_timeline_per_step(
                gpu_tl,
                nodes,
                int(step_id),
                self.out / f"01_gpu_timeline_step_{int(step_id)}.png",
            )

        self.plot_comm_vs_compute_breakdown(
            summary, self.out / "02_gpu_comm_vs_compute_breakdown.png"
        )
        self.plot_pipeline_transfer_view(
            msgs, nodes, edges, self.out / "03_pipeline_transfer_view.png"
        )

        # Lightweight textual summary.
        with open(self.out / "gpu_timeline_summary.md", "w", encoding="utf-8") as f:
            f.write("# High-level GPU Timeline Summary\n\n")
            f.write(f"- Input pcap: {self.cfg.input_pcap}\n")
            f.write(f"- Node mode: {node_mode}\n")
            f.write(f"- Nodes: {', '.join(nodes)}\n")
            f.write(f"- Window mode: {mode}\n")
            f.write(f"- Window: [{win_start:.6f}, {win_end:.6f}] s\n")
            if len(edges) >= 2:
                f.write("- Step boundaries (window-relative s): " + ", ".join(f"{x:.6f}" for x in edges) + "\n")
            f.write(f"- Reconstructed bulk transfers: {len(msgs)}\n")
            if not summary.empty:
                f.write("\n## Mean per-step breakdown by GPU and pass\n\n")
                f.write(summary.to_string(index=False))
                f.write("\n")

        print(f"[+] Done. Outputs written to: {self.out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="High-level pipeline GPU timeline analyzer")
    p.add_argument("-i", "--input", required=True, help="Input pcap path")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--start-ip", default="192.168.2.1", help="First pipeline node IP")
    p.add_argument("--end-ip", default="192.168.2.7", help="Last pipeline node IP")
    p.add_argument("--window-mode", choices=["steps", "time"], default="steps")
    p.add_argument("--auto-window-sec", type=float, default=4.0)
    p.add_argument("--start-step", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--active-bin-ms", type=float, default=10.0)
    p.add_argument("--bitrate-bin-ms", type=float, default=1.0)
    p.add_argument("--bulk-min-packets", type=int, default=16)
    p.add_argument("--bulk-min-bytes", type=int, default=131072)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config(
        input_pcap=args.input,
        output_dir=args.output,
        start_ip=args.start_ip,
        end_ip=args.end_ip,
        window_mode=args.window_mode,
        auto_window_sec=args.auto_window_sec,
        start_step=args.start_step,
        max_steps=args.max_steps,
        active_bin_ms=args.active_bin_ms,
        bitrate_bin_ms=args.bitrate_bin_ms,
        bulk_min_packets=args.bulk_min_packets,
        bulk_min_bytes=args.bulk_min_bytes,
    )
    HighLevelPipelineAnalyzer(cfg).run()


if __name__ == "__main__":
    main()
