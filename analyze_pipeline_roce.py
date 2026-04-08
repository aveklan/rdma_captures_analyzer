#!/usr/bin/env python3
"""
RoCE v2 RDMA Pipeline Parallelism Analyzer
==========================================
Analyzes header-only pcap captures for naive pipeline parallel GPT training.

Key features:
- Detects ordered stage communications (PC1->PC2->...->PC7)
- Focuses analysis on a short active window (1-2 training steps)
- Produces grouped metrics and plots for throughput, latency, jitter,
  congestion indicators, and phase timing
- Keeps implementation modular so new metrics can be added easily

Requirements:
    pip install numpy pandas matplotlib

Usage:
    python analyze_pipeline_roce.py \
      -i /path/to/pipeline_parallelism.pcap \
      -o analysis_output_pp
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROCE_UDP_DST_PORT = 4791
ROCEV1_ETHERTYPE = 0x8915
PFC_ETHERTYPE = 0x8808
PFC_OPCODE = 0x0101

BTH_OPCODES = {
    0x00: "RC_SEND_FIRST",
    0x01: "RC_SEND_MIDDLE",
    0x02: "RC_SEND_LAST",
    0x03: "RC_SEND_LAST_IMM",
    0x04: "RC_SEND_ONLY",
    0x05: "RC_SEND_ONLY_IMM",
    0x06: "RC_RDMA_WRITE_FIRST",
    0x07: "RC_RDMA_WRITE_MIDDLE",
    0x08: "RC_RDMA_WRITE_LAST",
    0x09: "RC_RDMA_WRITE_LAST_IMM",
    0x0A: "RC_RDMA_WRITE_ONLY",
    0x0B: "RC_RDMA_WRITE_ONLY_IMM",
    0x0C: "RC_RDMA_READ_REQUEST",
    0x0D: "RC_RDMA_READ_RESP_FIRST",
    0x0E: "RC_RDMA_READ_RESP_MIDDLE",
    0x0F: "RC_RDMA_READ_RESP_LAST",
    0x10: "RC_RDMA_READ_RESP_ONLY",
    0x11: "RC_ACKNOWLEDGE",
    0x12: "RC_ATOMIC_ACKNOWLEDGE",
    0x19: "RC_SEND_LAST_INV",
    0x64: "UD_SEND_ONLY",
    0x65: "UD_SEND_ONLY_IMM",
    0x81: "CNP",
}

RDMA_WRITE_OPCODES = {0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B}
WRITE_FIRST_OPCODES = {0x06}
WRITE_MIDDLE_OPCODES = {0x07}
WRITE_LAST_OPCODES = {0x08, 0x09}
WRITE_ONLY_OPCODES = {0x0A, 0x0B}


@dataclass
class AnalyzerConfig:
    pcap_path: str
    output_dir: str
    start_ip: str = "192.168.2.101"
    end_ip: str = "192.168.2.107"
    auto_window_sec: float = 4.0
    max_steps: int = 2
    bitrate_bin_ms: float = 1.0
    active_bin_ms: float = 10.0
    min_phase_gap_us: float = 500.0
    top_flows: int = 24
    window_mode: str = "steps"
    start_step: int = 1
    link_capacity_gbps: float = 10.0
    bulk_min_packets: int = 128
    bulk_min_bytes: int = 1000000


@dataclass
class StepWindow:
    start: float
    end: float
    boundaries: List[float]
    reference_stage: str
    mode: str


@dataclass
class Phase:
    stage_label: str
    stage_pair: str
    direction: str
    phase_id: int
    start: float
    end: float
    duration: float
    packets: int
    bytes_wire: float
    bytes_goodput: float
    metric_mode: str = "all_opcodes"


class PcapReader:
    @staticmethod
    def read_pcap(path: str) -> Iterable[Tuple[float, int, bytes, int]]:
        """Yield (timestamp_sec, orig_len, raw_bytes, network_type) from pcap."""
        with open(path, "rb") as f:
            ghdr = f.read(24)
            if len(ghdr) < 24:
                raise ValueError("File too short for pcap global header")

            magic = struct.unpack("<I", ghdr[:4])[0]
            if magic == 0xA1B2C3D4:
                endian = "<"
                ts_format = "standard"
            elif magic == 0xD4C3B2A1:
                endian = ">"
                ts_format = "standard"
            elif magic == 0xA1B23C4D:
                endian = "<"
                ts_format = "nano"
            elif magic == 0x4D3CB2A1:
                endian = ">"
                ts_format = "nano"
            else:
                raise ValueError(f"Unknown pcap magic: {hex(magic)}")

            _vmj, _vmn, _tz, _sig, _snaplen, network = struct.unpack(
                endian + "HHiIII", ghdr[4:]
            )
            rec_fmt = endian + "IIII"

            while True:
                rhdr = f.read(16)
                if len(rhdr) < 16:
                    break
                ts_sec, ts_frac, incl_len, orig_len = struct.unpack(rec_fmt, rhdr)
                timestamp = ts_sec + ts_frac * (1e-9 if ts_format == "nano" else 1e-6)
                raw = f.read(incl_len)
                if len(raw) < incl_len:
                    break
                yield timestamp, orig_len, raw, network


class PacketParser:
    @staticmethod
    def _ipv4_from_mapped_gid(gid_bytes: bytes) -> Optional[str]:
        if len(gid_bytes) != 16:
            return None
        if gid_bytes[:10] == b"\x00" * 10 and gid_bytes[10:12] == b"\xff\xff":
            return ".".join(str(b) for b in gid_bytes[12:16])
        return None

    @staticmethod
    def parse_frame(raw: bytes, link_type: int = 1) -> Dict[str, object]:
        """Parse generic frame-level events (including PFC) and RoCE fields."""
        out: Dict[str, object] = {
            "is_roce": False,
            "is_pfc": False,
            "src_ip": None,
            "dst_ip": None,
            "ip_ecn": None,
            "ip_total_length": None,
            "udp_src_port": None,
            "udp_dst_port": None,
            "opcode": None,
            "opcode_name": None,
            "dest_qp": None,
            "psn": None,
            "roce_version": None,
        }

        if link_type != 1 or len(raw) < 14:
            return out

        ethertype = struct.unpack("!H", raw[12:14])[0]
        offset = 14
        if ethertype == 0x8100 and len(raw) >= 18:
            ethertype = struct.unpack("!H", raw[16:18])[0]
            offset = 18

        if ethertype == PFC_ETHERTYPE and len(raw) >= offset + 2:
            opcode = struct.unpack("!H", raw[offset : offset + 2])[0]
            if opcode == PFC_OPCODE:
                out["is_pfc"] = True
            return out

        if ethertype == ROCEV1_ETHERTYPE:
            out["is_roce"] = True
            out["roce_version"] = "v1"

            # RoCEv1 carries GRH(40B) + BTH; decode IPv4-mapped GIDs when present.
            if len(raw) >= offset + 40:
                grh = raw[offset : offset + 40]
                src_gid = grh[8:24]
                dst_gid = grh[24:40]
                out["src_ip"] = PacketParser._ipv4_from_mapped_gid(src_gid)
                out["dst_ip"] = PacketParser._ipv4_from_mapped_gid(dst_gid)
                bth_offset = offset + 40
            else:
                bth_offset = offset

            if len(raw) >= bth_offset + 12:
                bth = raw[bth_offset : bth_offset + 12]
                opcode = bth[0]
                out["opcode"] = opcode
                out["opcode_name"] = BTH_OPCODES.get(opcode, f"UNKNOWN_0x{opcode:02X}")
                out["dest_qp"] = struct.unpack("!I", b"\x00" + bth[4:7])[0]
                out["psn"] = struct.unpack("!I", b"\x00" + bth[8:11])[0]
            return out

        if ethertype != 0x0800 or len(raw) < offset + 20:
            return out

        ip = raw[offset : offset + 20]
        ihl = (ip[0] & 0x0F) * 4
        if len(raw) < offset + ihl:
            return out

        out["src_ip"] = f"{ip[12]}.{ip[13]}.{ip[14]}.{ip[15]}"
        out["dst_ip"] = f"{ip[16]}.{ip[17]}.{ip[18]}.{ip[19]}"
        out["ip_total_length"] = struct.unpack("!H", ip[2:4])[0]
        tos = ip[1]
        out["ip_ecn"] = tos & 0x03
        protocol = ip[9]

        if protocol != 17:
            return out

        offset += ihl
        if len(raw) < offset + 8:
            return out

        udp = raw[offset : offset + 8]
        src_port, dst_port = struct.unpack("!HH", udp[:4])
        out["udp_src_port"] = src_port
        out["udp_dst_port"] = dst_port

        if dst_port != ROCE_UDP_DST_PORT:
            return out

        out["is_roce"] = True
        out["roce_version"] = "v2"

        offset += 8
        if len(raw) < offset + 12:
            return out

        bth = raw[offset : offset + 12]
        opcode = bth[0]
        out["opcode"] = opcode
        out["opcode_name"] = BTH_OPCODES.get(opcode, f"UNKNOWN_0x{opcode:02X}")
        out["dest_qp"] = struct.unpack("!I", b"\x00" + bth[4:7])[0]
        out["psn"] = struct.unpack("!I", b"\x00" + bth[8:11])[0]
        return out


class PipelineAnalyzer:
    def __init__(self, cfg: AnalyzerConfig):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.plot_dirs = {
            "traffic": self.output_dir / "01_traffic_structure",
            "throughput": self.output_dir / "02_throughput",
            "latency": self.output_dir / "03_latency",
            "congestion": self.output_dir / "04_congestion",
        }
        for p in self.plot_dirs.values():
            p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_payload_est(orig_len: float) -> float:
        # Ethernet + IPv4 + UDP + BTH + ICRC approximation
        return float(max(orig_len - 58.0, 0.0))

    @staticmethod
    def export_per_step_pair_traffic_wire(df: pd.DataFrame, out_path: Path) -> None:
        """Export per-step bidirectional pair traffic using wire bytes (orig_len)."""
        if df.empty or "step_id" not in df.columns:
            return

        tmp = df.copy()
        tmp["pair"] = tmp.apply(
            lambda r: " <-> ".join(sorted([str(r["src_ip"]), str(r["dst_ip"])])), axis=1
        )
        agg = (
            tmp.groupby(["step_id", "pair"], as_index=False)["orig_len"]
            .sum()
            .rename(columns={"orig_len": "wire_bytes"})
            .sort_values(["step_id", "pair"])
        )
        agg["wire_mb"] = agg["wire_bytes"] / 1e6
        agg.to_csv(out_path, index=False)

    def load_packets(self) -> Tuple[pd.DataFrame, int]:
        rows: List[Dict[str, object]] = []
        pfc_count = 0

        for pkt_idx, (ts, orig_len, raw, link_type) in enumerate(
            PcapReader.read_pcap(self.cfg.pcap_path)
        ):
            info = PacketParser.parse_frame(raw, link_type=link_type)

            if info.get("is_pfc"):
                pfc_count += 1

            if not info.get("is_roce"):
                continue

            row = {
                "pkt_idx": pkt_idx,
                "timestamp": ts,
                "orig_len": float(orig_len),
                "goodput_bytes_est": self._safe_payload_est(float(orig_len)),
                "src_ip": info.get("src_ip"),
                "dst_ip": info.get("dst_ip"),
                "ip_ecn": info.get("ip_ecn"),
                "opcode": info.get("opcode"),
                "opcode_name": info.get("opcode_name"),
                "dest_qp": info.get("dest_qp"),
                "psn": info.get("psn"),
                "roce_version": info.get("roce_version"),
            }
            rows.append(row)

        if not rows:
            raise RuntimeError("No RoCE packets found in capture.")

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["rel_time"] = df["timestamp"] - float(df["timestamp"].iloc[0])
        df["direction"] = df["src_ip"].astype(str) + " -> " + df["dst_ip"].astype(str)
        return df, pfc_count

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

    def resolve_pipeline_nodes(self, df: pd.DataFrame) -> Tuple[List[str], str]:
        requested = self._expand_ip_range(self.cfg.start_ip, self.cfg.end_ip)
        if requested:
            mask_req = df["src_ip"].isin(requested) & df["dst_ip"].isin(requested)
            req_packets = int(mask_req.sum())
            if req_packets > 1000:
                return requested, "requested_range"

        # Fallback: detect the 7 most active nodes in the same /24 as start_ip.
        prefix = ".".join(self.cfg.start_ip.split(".")[:3])
        ip_counts = pd.concat([df["src_ip"], df["dst_ip"]]).value_counts()
        candidates = [ip for ip in ip_counts.index if ip.startswith(prefix + ".")]
        chosen = sorted(candidates[:7], key=lambda x: tuple(int(p) for p in x.split(".")))

        if len(chosen) < 2:
            raise RuntimeError(
                "Could not resolve pipeline nodes. Check IP range or capture contents."
            )
        return chosen, "auto_detected_range"

    @staticmethod
    def _stage_pairs(nodes: Sequence[str]) -> List[Tuple[str, str, str]]:
        pairs = []
        for i in range(len(nodes) - 1):
            src = nodes[i]
            dst = nodes[i + 1]
            stage_label = f"S{i+1}: {src}->{dst}"
            pairs.append((src, dst, stage_label))
        return pairs

    def filter_pipeline_traffic(
        self, df: pd.DataFrame, nodes: Sequence[str]
    ) -> pd.DataFrame:
        pairs = self._stage_pairs(nodes)
        dirs = {f"{a} -> {b}" for a, b, _ in pairs}
        rev_dirs = {f"{b} -> {a}" for a, b, _ in pairs}
        allowed = dirs | rev_dirs
        out = df[df["direction"].isin(allowed)].copy()

        stage_by_dir = {}
        for i, (a, b, stage_label) in enumerate(pairs, start=1):
            stage_by_dir[f"{a} -> {b}"] = (i, stage_label, "forward")
            stage_by_dir[f"{b} -> {a}"] = (i, stage_label, "backward")

        out["stage_id"] = out["direction"].map(lambda d: stage_by_dir[d][0])
        out["stage_label"] = out["direction"].map(lambda d: stage_by_dir[d][1])
        out["stage_dir"] = out["direction"].map(lambda d: stage_by_dir[d][2])
        return out

    def detect_analysis_window(self, df: pd.DataFrame) -> Tuple[float, float]:
        if df.empty:
            return 0.0, 0.0

        bin_width = self.cfg.active_bin_ms / 1000.0
        bins = np.arange(0.0, float(df["rel_time"].max()) + bin_width, bin_width)
        if len(bins) < 2:
            start = float(df["rel_time"].min())
            return start, start + self.cfg.auto_window_sec

        bytes_hist, _ = np.histogram(df["rel_time"], bins=bins, weights=df["orig_len"])
        bitrate = (bytes_hist * 8.0) / (bin_width * 1e6)

        p90 = float(np.percentile(bitrate, 90)) if len(bitrate) else 0.0
        threshold = max(p90, float(np.max(bitrate)) * 0.15)
        active = bitrate > threshold

        run_start_idx = 0
        best_start_time = 0.0
        found = False
        for i, flag in enumerate(active):
            if flag and not found:
                run_start_idx = i
                found = True
            if found and (not flag or i == len(active) - 1):
                run_end_idx = i if not flag else i + 1
                run_dur = (run_end_idx - run_start_idx) * bin_width
                if run_dur >= 0.05:
                    best_start_time = max(0.0, bins[run_start_idx] - 0.2)
                    break
                found = False

        start = best_start_time
        end = start + self.cfg.auto_window_sec
        max_rel = float(df["rel_time"].max())
        if end > max_rel:
            end = max_rel
            start = max(0.0, end - self.cfg.auto_window_sec)
        return start, end

    def _select_reference_stage(self, phases: pd.DataFrame) -> Optional[str]:
        if phases.empty:
            return None
        # Use the earliest stage in the chain as cycle reference.
        ordered = phases.copy()
        ordered["stage_num"] = ordered["stage_label"].str.extract(r"S(\d+)").astype(float)
        ordered = ordered.sort_values(["stage_num", "start"])
        if ordered.empty:
            return None
        return str(ordered["stage_pair"].iloc[0])

    @staticmethod
    def _macro_starts_from_times(times: np.ndarray) -> np.ndarray:
        if len(times) == 0:
            return np.array([], dtype=float)
        if len(times) == 1:
            return times.astype(float)

        diffs = np.diff(times)
        p90 = float(np.percentile(diffs, 90))
        gap_thr = max(0.2, p90 * 0.5)

        starts_idx = [0]
        starts_idx.extend((np.where(diffs > gap_thr)[0] + 1).tolist())
        return times[np.asarray(starts_idx, dtype=int)].astype(float)

    @staticmethod
    def _bursts_from_times(times: np.ndarray) -> List[Tuple[float, float]]:
        """Split a timestamp stream into macro-bursts and return (start, end)."""
        if len(times) == 0:
            return []
        if len(times) == 1:
            t = float(times[0])
            return [(t, t)]

        diffs = np.diff(times)
        p90 = float(np.percentile(diffs, 90))
        gap_thr = max(0.2, p90 * 0.5)

        starts = [0]
        starts.extend((np.where(diffs > gap_thr)[0] + 1).tolist())
        starts = np.asarray(starts, dtype=int)

        bursts: List[Tuple[float, float]] = []
        for i, a in enumerate(starts):
            b = starts[i + 1] if i + 1 < len(starts) else len(times)
            t0 = float(times[a])
            t1 = float(times[b - 1])
            bursts.append((t0, t1))
        return bursts

    @staticmethod
    def _reference_clusters(
        ref_df: pd.DataFrame, fwd_dir: str, bwd_dir: str
    ) -> List[Dict[str, object]]:
        """Cluster ref-edge packets and mark dominant direction per cluster."""
        if ref_df.empty:
            return []
        t = ref_df["rel_time"].to_numpy(dtype=float)
        if len(t) == 0:
            return []

        if len(t) == 1:
            r = ref_df.iloc[0]
            f = 1 if r["direction"] == fwd_dir else 0
            b = 1 if r["direction"] == bwd_dir else 0
            return [
                {
                    "start": float(t[0]),
                    "end": float(t[0]),
                    "fwd_packets": f,
                    "bwd_packets": b,
                    "dominant": "F" if f >= b else "B",
                }
            ]

        dt = np.diff(t)
        # Split only at macro idle gaps so tiny/long sub-bursts stay in one phase cluster.
        gap_thr = max(0.2, float(np.percentile(dt, 99)) * 10.0)
        starts = [0]
        starts.extend((np.where(dt > gap_thr)[0] + 1).tolist())
        starts = np.asarray(starts, dtype=int)

        clusters: List[Dict[str, object]] = []
        for i, a in enumerate(starts):
            b = starts[i + 1] if i + 1 < len(starts) else len(ref_df)
            c = ref_df.iloc[a:b]
            if c.empty:
                continue
            fwd_n = int((c["direction"] == fwd_dir).sum())
            bwd_n = int((c["direction"] == bwd_dir).sum())
            dom = "F" if fwd_n >= bwd_n else "B"
            clusters.append(
                {
                    "start": float(c["rel_time"].iloc[0]),
                    "end": float(c["rel_time"].iloc[-1]),
                    "fwd_packets": fwd_n,
                    "bwd_packets": bwd_n,
                    "dominant": dom,
                }
            )
        return clusters

    def detect_step_window(self, df: pd.DataFrame) -> StepWindow:
        """Detect a window containing exactly max_steps pipeline cycles when possible."""
        fallback_start, fallback_end = self.detect_analysis_window(df)
        fallback = StepWindow(
            start=fallback_start,
            end=fallback_end,
            boundaries=[fallback_start, fallback_end],
            reference_stage="n/a",
            mode="time_fallback",
        )

        if df.empty:
            return fallback

        # Reference edge is stage 1 (node1<->node2); step starts on F-dominant
        # ref cluster and ends on B-dominant ref cluster.
        ref_rows = df[(df["stage_id"] == 1)]
        if ref_rows.empty:
            phases_full = self.detect_phases(df)
            if phases_full.empty:
                return fallback
            ref_stage = self._select_reference_stage(phases_full)
            if ref_stage is None:
                return fallback
        else:
            ref_stage = str(
                ref_rows[ref_rows["stage_dir"] == "forward"]["direction"].iloc[0]
            )

        try:
            a, b = ref_stage.split(" -> ")
        except ValueError:
            return fallback

        rev_stage = f"{b} -> {a}"
        ref_df = df[df["direction"].isin([ref_stage, rev_stage])].sort_values("rel_time")
        if ref_df.empty:
            return fallback

        clusters = self._reference_clusters(ref_df, ref_stage, rev_stage)
        f_clusters = [c for c in clusters if c["dominant"] == "F"]
        b_clusters = [c for c in clusters if c["dominant"] == "B"]
        if len(f_clusters) < 1 or len(b_clusters) < 1:
            return fallback

        f_starts = np.asarray([float(c["start"]) for c in f_clusters], dtype=float)
        full_diffs = np.diff(f_starts)
        cycle = float(np.median(full_diffs)) if len(full_diffs) else self.cfg.auto_window_sec
        if cycle <= 0:
            cycle = self.cfg.auto_window_sec

        # Anchor near first high-activity interval to skip init traffic.
        active_start, _ = self.detect_analysis_window(df)
        base_idx = int(np.searchsorted(f_starts, active_start, side="left"))
        if base_idx >= len(f_clusters):
            base_idx = max(0, len(f_clusters) - 1)

        start_step = max(1, int(self.cfg.start_step))
        steps = max(1, int(self.cfg.max_steps))
        idx0 = base_idx + (start_step - 1)
        if idx0 >= len(f_clusters):
            idx0 = len(f_clusters) - 1

        idx_last = min(idx0 + steps - 1, len(f_clusters) - 1)

        # Start on selected F-dominant reference cluster.
        start = float(f_clusters[idx0]["start"])

        # Step end is the last backward burst (reverse reference edge) that occurs
        # after the selected forward burst and before the next selected forward burst.
        def step_end_from_ref(i: int) -> float:
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

        end = step_end_from_ref(idx_last)

        # Keep boundary vector compatible with downstream step-id assignment:
        # [start_of_selected_step_1, start_of_selected_step_2, ..., end_of_last_selected_step].
        boundaries = [start]
        boundaries.extend(float(f_clusters[i]["start"]) for i in range(idx0 + 1, idx_last + 1))
        boundaries.append(float(end))

        max_rel = float(df["rel_time"].max())
        if end > max_rel:
            end = max_rel
            boundaries[-1] = end

        if end <= start:
            return fallback

        out_mode = "steps"
        if len(boundaries) >= 3:
            step_lens = np.diff(np.asarray(boundaries, dtype=float))
            if len(step_lens) >= 2 and (np.max(step_lens) / max(np.min(step_lens), 1e-9)) > 1.8:
                out_mode = "steps_filtered"

        return StepWindow(
            start=start,
            end=end,
            boundaries=boundaries,
            reference_stage=ref_stage,
            mode=out_mode,
        )

    @staticmethod
    def _phase_gap_threshold(times: np.ndarray, min_phase_gap_us: float) -> float:
        if len(times) < 3:
            return max(min_phase_gap_us / 1e6, 0.0005)
        ipg = np.diff(times)
        med = float(np.median(ipg))
        return max(min_phase_gap_us / 1e6, med * 20.0)

    def detect_phases(self, df: pd.DataFrame, metric_mode: str = "all_opcodes") -> pd.DataFrame:
        phases: List[Phase] = []

        grouped = df[df["stage_dir"] == "forward"].sort_values("rel_time").groupby(
            "direction"
        )
        for direction, sub in grouped:
            if sub.empty:
                continue
            times = sub["rel_time"].to_numpy(dtype=float)
            gap_thr = self._phase_gap_threshold(times, self.cfg.min_phase_gap_us)

            boundaries = [0]
            dt = np.diff(times)
            boundaries.extend((np.where(dt > gap_thr)[0] + 1).tolist())
            boundaries.append(len(sub))

            stage_label = str(sub["stage_label"].iloc[0])
            pair = direction
            phase_id = 0

            for i in range(len(boundaries) - 1):
                a, b = boundaries[i], boundaries[i + 1]
                chunk = sub.iloc[a:b]
                if len(chunk) < 2:
                    continue
                start = float(chunk["rel_time"].iloc[0])
                end = float(chunk["rel_time"].iloc[-1])
                duration = max(end - start, 0.0)
                if duration <= 0.0:
                    continue
                phase_id += 1
                phases.append(
                    Phase(
                        stage_label=stage_label,
                        stage_pair=pair,
                        direction=direction,
                        phase_id=phase_id,
                        start=start,
                        end=end,
                        duration=duration,
                        packets=int(len(chunk)),
                        bytes_wire=float(chunk["orig_len"].sum()),
                        bytes_goodput=float(chunk["goodput_bytes_est"].sum()),
                        metric_mode=metric_mode,
                    )
                )

        if not phases:
            return pd.DataFrame(
                columns=[
                    "stage_label",
                    "stage_pair",
                    "direction",
                    "phase_id",
                    "start",
                    "end",
                    "duration",
                    "packets",
                    "bytes_wire",
                    "bytes_goodput",
                    "inter_phase_gap",
                    "metric_mode",
                ]
            )

        ph = pd.DataFrame([p.__dict__ for p in phases]).sort_values(["stage_pair", "start"])
        ph["inter_phase_gap"] = ph.groupby("stage_pair")["start"].diff() - ph.groupby(
            "stage_pair"
        )["duration"].shift(1)
        return ph

    def detect_bulk_write_phases(
        self, df: pd.DataFrame, metric_mode: str = "write_bulk_strict"
    ) -> pd.DataFrame:
        """Reconstruct write messages and keep only bulk ones.

        A message starts at WRITE_FIRST and ends at WRITE_LAST (or is WRITE_ONLY).
        Messages are tracked per direction and destination QP.
        """
        cols = [
            "stage_label",
            "stage_pair",
            "direction",
            "phase_id",
            "start",
            "end",
            "duration",
            "packets",
            "bytes_wire",
            "bytes_goodput",
            "inter_phase_gap",
            "metric_mode",
        ]

        src = df[
            (df["stage_dir"] == "forward")
            & (df["opcode"].isin(RDMA_WRITE_OPCODES))
        ].sort_values("rel_time")
        if src.empty:
            return pd.DataFrame(columns=cols)

        out_rows: List[Dict[str, object]] = []

        for direction, g in src.groupby("direction"):
            active: Dict[int, Dict[str, object]] = {}
            completed: List[Dict[str, object]] = []
            stage_label = str(g["stage_label"].iloc[0])

            def new_msg(row: pd.Series) -> Dict[str, object]:
                t = float(row["rel_time"])
                return {
                    "start": t,
                    "end": t,
                    "packets": 0,
                    "bytes_wire": 0.0,
                    "bytes_goodput": 0.0,
                }

            def add_pkt(msg: Dict[str, object], row: pd.Series) -> None:
                t = float(row["rel_time"])
                msg["end"] = t
                msg["packets"] = int(msg["packets"]) + 1
                msg["bytes_wire"] = float(msg["bytes_wire"]) + float(row["orig_len"])
                msg["bytes_goodput"] = float(msg["bytes_goodput"]) + float(
                    row["goodput_bytes_est"]
                )

            for _, row in g.iterrows():
                op = int(row["opcode"])
                qp = int(row["dest_qp"]) if pd.notna(row["dest_qp"]) else -1

                if op in WRITE_ONLY_OPCODES:
                    msg = new_msg(row)
                    add_pkt(msg, row)
                    completed.append(msg)
                    continue

                if op in WRITE_FIRST_OPCODES:
                    msg = new_msg(row)
                    add_pkt(msg, row)
                    active[qp] = msg
                    continue

                if op in WRITE_MIDDLE_OPCODES:
                    msg = active.get(qp)
                    if msg is None:
                        msg = new_msg(row)
                        active[qp] = msg
                    add_pkt(msg, row)
                    continue

                if op in WRITE_LAST_OPCODES:
                    msg = active.get(qp)
                    if msg is None:
                        msg = new_msg(row)
                    add_pkt(msg, row)
                    completed.append(msg)
                    if qp in active:
                        del active[qp]

            # Flush partial messages so they can still be considered for bulk if large.
            completed.extend(active.values())
            if not completed:
                continue

            msg_df = pd.DataFrame(completed)
            bulk = msg_df[
                (msg_df["packets"] >= int(self.cfg.bulk_min_packets))
                | (msg_df["bytes_wire"] >= float(self.cfg.bulk_min_bytes))
            ].copy()
            if bulk.empty and not msg_df.empty:
                # Fallback: keep the top bulk-like tail if absolute thresholds are too strict
                # for this capture window.
                pkt_thr = max(8.0, float(np.percentile(msg_df["packets"], 90)))
                byte_thr = max(4096.0, float(np.percentile(msg_df["bytes_wire"], 90)))
                bulk = msg_df[
                    (msg_df["packets"] >= pkt_thr) | (msg_df["bytes_wire"] >= byte_thr)
                ].copy()
            if bulk.empty:
                continue

            bulk.sort_values("start", inplace=True)
            bulk.reset_index(drop=True, inplace=True)
            bulk["phase_id"] = np.arange(1, len(bulk) + 1)
            bulk["duration"] = (bulk["end"] - bulk["start"]).clip(lower=0.0)
            bulk["inter_phase_gap"] = bulk["start"].diff() - bulk["duration"].shift(1)

            for _, r in bulk.iterrows():
                out_rows.append(
                    {
                        "stage_label": stage_label,
                        "stage_pair": direction,
                        "direction": direction,
                        "phase_id": int(r["phase_id"]),
                        "start": float(r["start"]),
                        "end": float(r["end"]),
                        "duration": float(r["duration"]),
                        "packets": int(r["packets"]),
                        "bytes_wire": float(r["bytes_wire"]),
                        "bytes_goodput": float(r["bytes_goodput"]),
                        "inter_phase_gap": float(r["inter_phase_gap"])
                        if pd.notna(r["inter_phase_gap"])
                        else np.nan,
                        "metric_mode": metric_mode,
                    }
                )

        if not out_rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(out_rows).sort_values(["stage_pair", "start"]).reset_index(
            drop=True
        )

    @staticmethod
    def estimate_retransmissions(df: pd.DataFrame) -> pd.DataFrame:
        events = []
        sub = df.dropna(subset=["dest_qp", "psn"]).sort_values("rel_time")
        for (direction, qp), grp in sub.groupby(["direction", "dest_qp"]):
            prev_psn = None
            for _, row in grp.iterrows():
                curr_psn = int(row["psn"])
                if prev_psn is not None:
                    if curr_psn == prev_psn:
                        ev = "duplicate_psn"
                    elif curr_psn < prev_psn and not (
                        prev_psn > 0xF00000 and curr_psn < 0x0FFFFF
                    ):
                        ev = "psn_regression"
                    else:
                        ev = None
                    if ev is not None:
                        events.append(
                            {
                                "direction": direction,
                                "dest_qp": int(qp),
                                "rel_time": float(row["rel_time"]),
                                "prev_psn": prev_psn,
                                "psn": curr_psn,
                                "event": ev,
                            }
                        )
                prev_psn = curr_psn
        return pd.DataFrame(events)

    @staticmethod
    def _save_plot(fig: plt.Figure, path: Path) -> None:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _annotate_step_boundaries(
        ax: plt.Axes, step_edges: Optional[Sequence[float]] = None
    ) -> None:
        if not step_edges or len(step_edges) < 2:
            return

        ymin, ymax = ax.get_ylim()
        yr = ymax - ymin if ymax > ymin else 1.0
        label_y = ymax - 0.03 * yr

        for i, edge in enumerate(step_edges):
            ax.axvline(edge, color="black", linestyle="--", linewidth=0.7, alpha=0.55)
            if i < len(step_edges) - 1:
                mid = 0.5 * (edge + step_edges[i + 1])
                ax.text(
                    mid,
                    label_y,
                    f"step {i+1}",
                    fontsize=8,
                    ha="center",
                    va="top",
                    bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
                )

    def plot_packet_timeline(
        self,
        df: pd.DataFrame,
        path: Path,
        step_edges: Optional[Sequence[float]] = None,
    ) -> None:
        stage_dirs = sorted(df["direction"].unique())
        fig, ax = plt.subplots(figsize=(15, 5))
        cmap = plt.get_cmap("tab20", len(stage_dirs))
        ymap = {d: i for i, d in enumerate(stage_dirs)}

        for i, d in enumerate(stage_dirs):
            sub = df[df["direction"] == d]
            ax.scatter(sub["rel_time"], [ymap[d]] * len(sub), s=0.4, alpha=0.5, color=cmap(i))

        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(stage_dirs)
        ax.set_xlabel("Time in selected window (s)")
        ax.set_title("1.1 Pipeline Packet Timeline (Stage-by-Stage)")
        ax.grid(True, alpha=0.3)
        self._annotate_step_boundaries(ax, step_edges)
        self._save_plot(fig, path)

    def plot_bitrate(
        self,
        df: pd.DataFrame,
        path: Path,
        step_edges: Optional[Sequence[float]] = None,
    ) -> None:
        bin_w = self.cfg.bitrate_bin_ms / 1000.0
        max_t = float(df["rel_time"].max())
        bins = np.arange(0.0, max_t + bin_w, bin_w)

        fig, ax = plt.subplots(figsize=(15, 5))
        for direction in sorted(df["direction"].unique()):
            sub = df[df["direction"] == direction]
            b, _ = np.histogram(sub["rel_time"], bins=bins, weights=sub["orig_len"])
            mbps = (b * 8.0) / (bin_w * 1e6)
            ax.plot(bins[:-1], mbps, linewidth=0.8, label=direction)

        ax.set_xlabel("Time in selected window (s)")
        ax.set_ylabel("Bitrate (Mbps)")
        ax.set_title("2.1 Bitrate per Stage-to-Stage Direction")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        self._annotate_step_boundaries(ax, step_edges)
        self._save_plot(fig, path)

    def plot_phase_durations(self, phases: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        grouped = [g["duration"].to_numpy() * 1e3 for _, g in phases.groupby("stage_pair")]
        labels = [k for k, _ in phases.groupby("stage_pair")]
        if grouped:
            ax.boxplot(grouped, tick_labels=labels, showfliers=False)
        else:
            ax.text(
                0.5,
                0.5,
                "No phases available with current filter",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self._save_plot(fig, path)
            return
        ax.set_ylabel("Transfer phase duration (ms)")
        ax.set_title("1.2 Stage-to-Stage Transfer Duration")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        self._save_plot(fig, path)

    def plot_inter_phase_gaps(self, phases: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        data = []
        labels = []
        for stage_pair, g in phases.groupby("stage_pair"):
            vals = g["inter_phase_gap"].dropna().to_numpy() * 1e3
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(stage_pair)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_ylabel("Inter-phase gap / compute pause (ms)")
        ax.set_title("1.3 Pause Between Communication Phases")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        self._save_plot(fig, path)

    def plot_goodput_raw(self, phases: pd.DataFrame, path: Path) -> None:
        agg = (
            phases.groupby("stage_pair")[["bytes_wire", "bytes_goodput", "duration"]]
            .sum()
            .reset_index()
        )
        agg["raw_mbps"] = (agg["bytes_wire"] * 8.0) / agg["duration"] / 1e6
        agg["goodput_mbps"] = (agg["bytes_goodput"] * 8.0) / agg["duration"] / 1e6

        x = np.arange(len(agg))
        w = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w / 2, agg["raw_mbps"], width=w, label="Raw throughput")
        ax.bar(x + w / 2, agg["goodput_mbps"], width=w, label="Goodput (est)")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["stage_pair"], rotation=20, ha="right")
        ax.set_ylabel("Mbps during active transfer time")
        ax.set_title("2.2 Effective Throughput per Stage (Raw vs Goodput)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        self._save_plot(fig, path)

    def plot_opcode_distribution(
        self,
        df: pd.DataFrame,
        path: Path,
        title: str,
        top_n: int = 8,
    ) -> None:
        if df.empty or "opcode_name" not in df.columns:
            return

        counts = df["opcode_name"].value_counts()
        total = float(counts.sum()) if len(counts) else 1.0
        pct = (counts / total * 100.0).sort_values(ascending=True)

        if len(pct) > top_n:
            top = pct.tail(top_n - 1)
            other = 100.0 - float(top.sum())
            pct = pd.concat([top, pd.Series({"Other": other})]).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.barh(pct.index, pct.values)
        ax.set_xlabel("Share of packets (%)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3)

        for idx, value in enumerate(pct.values):
            ax.text(value + 0.5, idx, f"{value:.1f}%", va="center", fontsize=8)

        self._save_plot(fig, path)

    def _ipg_series(self, df: pd.DataFrame, rdma_write_only: bool = True) -> pd.DataFrame:
        src = df
        if rdma_write_only:
            filt = src["opcode"].isin(RDMA_WRITE_OPCODES)
            if int(filt.sum()) > 10:
                src = src[filt]
        src = src.sort_values(["direction", "rel_time"]).copy()
        src["ipg"] = src.groupby("direction")["rel_time"].diff()
        return src.dropna(subset=["ipg"])

    def plot_interpacket_cdf(self, df: pd.DataFrame, path: Path) -> None:
        ipg_df = self._ipg_series(df, rdma_write_only=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        for direction, g in ipg_df.groupby("direction"):
            vals = np.sort(g["ipg"].to_numpy() * 1e6)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, linewidth=1.0, label=direction)
        ax.set_xscale("log")
        ax.set_xlabel("Inter-packet latency / gap (us)")
        ax.set_ylabel("CDF")
        ax.set_title("3.1 Inter-Packet Latency CDF (RDMA write traffic)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        self._save_plot(fig, path)

    def plot_jitter(self, df: pd.DataFrame, path: Path) -> None:
        ipg_df = self._ipg_series(df, rdma_write_only=True)
        jitter = ipg_df.groupby("direction")["ipg"].std().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(jitter.index, jitter.values * 1e6)
        ax.set_ylabel("Jitter (std(IPG), us)")
        ax.set_title("3.2 Jitter per Stage Direction")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        self._save_plot(fig, path)

    def plot_tail_latency(self, df: pd.DataFrame, phases: pd.DataFrame, path: Path) -> None:
        ipg_df = self._ipg_series(df, rdma_write_only=True)
        rows = []
        for direction, g in ipg_df.groupby("direction"):
            vals = g["ipg"].to_numpy() * 1e6
            rows.append((direction, "ipg_p99", np.percentile(vals, 99)))
            rows.append((direction, "ipg_p999", np.percentile(vals, 99.9)))

        for stage_pair, g in phases.groupby("stage_pair"):
            vals = g["duration"].to_numpy() * 1e3
            if len(vals) >= 2:
                rows.append((stage_pair, "fct_p99_ms", np.percentile(vals, 99)))

        tail = pd.DataFrame(rows, columns=["series", "metric", "value"])
        fig, ax = plt.subplots(figsize=(12, 5))
        for metric, g in tail.groupby("metric"):
            ax.plot(g["series"], g["value"], marker="o", linewidth=1, label=metric)
        ax.set_ylabel("Tail metric value")
        ax.set_title("3.3 Tail Latency (p99 / p99.9)")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.legend()
        self._save_plot(fig, path)

    def plot_ipg_timeseries(
        self,
        df: pd.DataFrame,
        path: Path,
        step_edges: Optional[Sequence[float]] = None,
    ) -> None:
        ipg_df = self._ipg_series(df, rdma_write_only=True)
        fig, ax = plt.subplots(figsize=(14, 5))
        for direction, g in ipg_df.groupby("direction"):
            ax.plot(g["rel_time"], g["ipg"] * 1e6, linewidth=0.5, alpha=0.8, label=direction)
        ax.set_yscale("log")
        ax.set_xlabel("Time in selected window (s)")
        ax.set_ylabel("IPG (us)")
        ax.set_title("3.4 Inter-Packet Gap (IPG) Timeline")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        self._annotate_step_boundaries(ax, step_edges)
        self._save_plot(fig, path)

    def plot_fct(self, phases: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        data = []
        labels = []
        for stage_pair, g in phases.groupby("stage_pair"):
            vals = g["duration"].to_numpy() * 1e3
            if len(vals) > 0:
                data.append(vals)
                labels.append(stage_pair)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_ylabel("Flow completion time (ms)")
        ax.set_title("3.5 Flow Completion Time (per phase, per stage)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        self._save_plot(fig, path)

    def plot_phase_durations_dual(
        self, phases_all: pd.DataFrame, phases_write: pd.DataFrame
    ) -> None:
        # Keep original filename as all-opcodes view for compatibility.
        self.plot_phase_durations(
            phases_all, self.plot_dirs["traffic"] / "02_stage_transfer_duration.png"
        )
        self.plot_phase_durations(
            phases_write,
            self.plot_dirs["traffic"] / "02_stage_transfer_duration_write_bulk_strict.png",
        )

    def plot_inter_phase_gaps_dual(
        self, phases_all: pd.DataFrame, phases_write: pd.DataFrame
    ) -> None:
        self.plot_inter_phase_gaps(
            phases_all, self.plot_dirs["traffic"] / "03_inter_phase_gaps.png"
        )
        self.plot_inter_phase_gaps(
            phases_write,
            self.plot_dirs["traffic"] / "03_inter_phase_gaps_write_bulk_strict.png",
        )

    def plot_fct_dual(self, phases_all: pd.DataFrame, phases_write: pd.DataFrame) -> None:
        self.plot_fct(phases_all, self.plot_dirs["latency"] / "11_fct.png")
        self.plot_fct(
            phases_write, self.plot_dirs["latency"] / "11_fct_write_bulk_strict.png"
        )

    def export_per_step_stage_bulk_summary(
        self, phases_write: pd.DataFrame, step_edges: Sequence[float]
    ) -> pd.DataFrame:
        cols = [
            "step_id",
            "stage_pair",
            "messages_per_step",
            "active_time_sec",
            "active_time_ms",
            "step_duration_sec",
            "duty_cycle_pct",
            "wire_bytes",
            "wire_mb",
            "avg_msg_wire_bytes",
        ]
        out_path = self.output_dir / "per_step_stage_bulk_summary.csv"

        if phases_write.empty or "step_id" not in phases_write.columns:
            pd.DataFrame(columns=cols).to_csv(out_path, index=False)
            return pd.DataFrame(columns=cols)

        agg = (
            phases_write.groupby(["step_id", "stage_pair"], as_index=False)
            .agg(
                messages_per_step=("phase_id", "count"),
                active_time_sec=("duration", "sum"),
                wire_bytes=("bytes_wire", "sum"),
                avg_msg_wire_bytes=("bytes_wire", "mean"),
            )
            .sort_values(["step_id", "stage_pair"])
        )
        agg["active_time_ms"] = agg["active_time_sec"] * 1e3
        agg["wire_mb"] = agg["wire_bytes"] / 1e6

        step_dur = {}
        if step_edges and len(step_edges) >= 2:
            for i in range(1, len(step_edges)):
                step_dur[i] = max(float(step_edges[i] - step_edges[i - 1]), 1e-12)
        agg["step_duration_sec"] = agg["step_id"].map(step_dur).fillna(np.nan)
        agg["duty_cycle_pct"] = (agg["active_time_sec"] / agg["step_duration_sec"]) * 100.0

        agg = agg[cols]
        agg.to_csv(out_path, index=False)
        return agg

    def plot_per_step_stage_bulk_summary(self, bulk_df: pd.DataFrame, path: Path) -> None:
        if bulk_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(
                0.5,
                0.5,
                "No strict bulk messages available in selected window",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self._save_plot(fig, path)
            return

        msg_pivot = bulk_df.pivot(index="stage_pair", columns="step_id", values="messages_per_step")
        act_pivot = bulk_df.pivot(index="stage_pair", columns="step_id", values="active_time_ms")

        msg_pivot = msg_pivot.sort_index()
        act_pivot = act_pivot.reindex(msg_pivot.index)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        im0 = axes[0].imshow(msg_pivot.values, aspect="auto", interpolation="nearest")
        axes[0].set_title("Messages per Step (Strict Bulk)")
        axes[0].set_ylabel("Stage pair")
        axes[0].set_yticks(np.arange(len(msg_pivot.index)))
        axes[0].set_yticklabels(msg_pivot.index)
        c0 = fig.colorbar(im0, ax=axes[0])
        c0.set_label("messages")

        im1 = axes[1].imshow(act_pivot.values, aspect="auto", interpolation="nearest")
        axes[1].set_title("Active Time per Step (Strict Bulk)")
        axes[1].set_ylabel("Stage pair")
        axes[1].set_xlabel("Step ID")
        axes[1].set_yticks(np.arange(len(act_pivot.index)))
        axes[1].set_yticklabels(act_pivot.index)
        axes[1].set_xticks(np.arange(len(act_pivot.columns)))
        axes[1].set_xticklabels([str(int(x)) for x in act_pivot.columns])
        c1 = fig.colorbar(im1, ax=axes[1])
        c1.set_label("active time (ms)")

        self._save_plot(fig, path)

    def plot_periodicity(self, phases: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))
        stage1 = phases.sort_values("start")
        if not stage1.empty:
            first_pair = stage1["stage_pair"].iloc[0]
            starts = stage1[stage1["stage_pair"] == first_pair]["start"].to_numpy()
            if len(starts) >= 2:
                cyc = np.diff(starts) * 1e3
                ax.plot(np.arange(1, len(cyc) + 1), cyc, marker="o")
                ax.set_ylabel("Cycle time estimate (ms)")
                ax.set_xlabel("Cycle index")
                ax.set_title("1.4 Periodicity & Cycle Time (from first stage)")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Not enough phases for cycle-time estimate", ha="center")
                ax.set_axis_off()
        self._save_plot(fig, path)

    def plot_congestion(self, df: pd.DataFrame, retrans: pd.DataFrame, pfc_count: int, path: Path) -> None:
        cnp = int((df["opcode"] == 0x81).sum())
        ecn_ce = int((df["ip_ecn"] == 3).sum())
        dup = int((retrans["event"] == "duplicate_psn").sum()) if not retrans.empty else 0
        reg = int((retrans["event"] == "psn_regression").sum()) if not retrans.empty else 0

        labels = ["CNP", "ECN-CE", "PFC", "Dup-PSN", "PSN-regression"]
        vals = [cnp, ecn_ce, pfc_count, dup, reg]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, vals)
        ax.set_ylabel("Event count")
        ax.set_title("4.1 Congestion / Reliability Indicators")
        ax.grid(True, axis="y", alpha=0.3)
        self._save_plot(fig, path)

    def write_summary(
        self,
        all_df: pd.DataFrame,
        df: pd.DataFrame,
        phases: pd.DataFrame,
        phases_write: pd.DataFrame,
        retrans: pd.DataFrame,
        pfc_count: int,
        nodes: Sequence[str],
        node_mode: str,
        win_start: float,
        win_end: float,
        step_edges: Sequence[float],
        segmentation_mode: str,
        segmentation_reference_stage: str,
    ) -> None:
        summary_path = self.output_dir / "summary_report.md"

        total_duration = float(all_df["rel_time"].max()) if not all_df.empty else 0.0
        sel_duration = max(win_end - win_start, 1e-12)

        bytes_sel = float(df["orig_len"].sum())
        avg_bitrate_sel = (bytes_sel * 8.0) / sel_duration / 1e6

        # Peak bitrate in 1 ms bins for each direction and aggregate window utilization.
        peak_bin_w = 0.001
        max_t = float(df["rel_time"].max()) if not df.empty else 0.0
        peak_bins = np.arange(0.0, max_t + peak_bin_w, peak_bin_w)
        if len(peak_bins) < 2:
            peak_bins = np.array([0.0, peak_bin_w])

        cap_mbps = max(self.cfg.link_capacity_gbps * 1000.0, 1e-9)
        util_rows = []

        if not phases.empty:
            tmp = phases.groupby("stage_pair")[["bytes_wire", "bytes_goodput", "duration"]].sum()
            tmp = tmp.copy()
            tmp["raw_mbps"] = (tmp["bytes_wire"] * 8.0) / tmp["duration"] / 1e6
            tmp["goodput_mbps"] = (tmp["bytes_goodput"] * 8.0) / tmp["duration"] / 1e6

            for stage_pair, r in tmp.iterrows():
                sub = df[df["direction"] == stage_pair]
                counts, _ = np.histogram(sub["rel_time"], bins=peak_bins, weights=sub["orig_len"])
                peak_mbps = float(np.max((counts * 8.0) / (peak_bin_w * 1e6))) if len(counts) else 0.0
                util_rows.append(
                    {
                        "stage_pair": stage_pair,
                        "raw_mbps": float(r["raw_mbps"]),
                        "goodput_mbps": float(r["goodput_mbps"]),
                        "peak_1ms_mbps": peak_mbps,
                        "avg_util_pct": float(r["raw_mbps"]) / cap_mbps * 100.0,
                        "peak_util_pct": peak_mbps / cap_mbps * 100.0,
                    }
                )
        else:
            tmp = pd.DataFrame()

        util_df = pd.DataFrame(util_rows)

        # Aggregate peak utilization across all traffic in selected window.
        agg_counts, _ = np.histogram(df["rel_time"], bins=peak_bins, weights=df["orig_len"])
        agg_peak_mbps = (
            float(np.max((agg_counts * 8.0) / (peak_bin_w * 1e6))) if len(agg_counts) else 0.0
        )
        avg_util_total_pct = avg_bitrate_sel / cap_mbps * 100.0
        peak_util_total_pct = agg_peak_mbps / cap_mbps * 100.0

        write_df = df[df["opcode"].isin(RDMA_WRITE_OPCODES)]
        if len(write_df) > 10:
            ipg = (
                write_df.sort_values(["direction", "rel_time"])
                .groupby("direction")["rel_time"]
                .diff()
                .dropna()
                .to_numpy()
                * 1e6
            )
        else:
            ipg = (
                df.sort_values(["direction", "rel_time"])
                .groupby("direction")["rel_time"]
                .diff()
                .dropna()
                .to_numpy()
                * 1e6
            )

        p99_ipg = float(np.percentile(ipg, 99)) if len(ipg) else float("nan")
        p999_ipg = float(np.percentile(ipg, 99.9)) if len(ipg) else float("nan")

        cnp_count = int((df["opcode"] == 0x81).sum())
        ecn_ce_count = int((df["ip_ecn"] == 3).sum())

        fct_stats = (
            phases.groupby("stage_pair")["duration"].agg(["count", "mean", "median", "max"])
            if not phases.empty
            else pd.DataFrame()
        )
        fct_stats_write = (
            phases_write.groupby("stage_pair")["duration"].agg(["count", "mean", "median", "max"])
            if not phases_write.empty
            else pd.DataFrame()
        )

        def _delta_pct(write_value: float, all_value: float) -> float:
            if not np.isfinite(write_value) or not np.isfinite(all_value) or all_value == 0:
                return float("nan")
            return (write_value - all_value) / all_value * 100.0

        delta_rows = []
        all_stages = sorted(set(phases["stage_pair"]).union(set(phases_write["stage_pair"])))
        for stage_pair in all_stages:
            all_sub = phases[phases["stage_pair"] == stage_pair]
            write_sub = phases_write[phases_write["stage_pair"] == stage_pair]

            all_dur_med = float(all_sub["duration"].median()) if not all_sub.empty else float("nan")
            write_dur_med = (
                float(write_sub["duration"].median()) if not write_sub.empty else float("nan")
            )
            all_gap_med = (
                float(all_sub["inter_phase_gap"].median())
                if not all_sub.empty and "inter_phase_gap" in all_sub.columns
                else float("nan")
            )
            write_gap_med = (
                float(write_sub["inter_phase_gap"].median())
                if not write_sub.empty and "inter_phase_gap" in write_sub.columns
                else float("nan")
            )
            all_fct_med = float(all_sub["duration"].median()) if not all_sub.empty else float("nan")
            write_fct_med = (
                float(write_sub["duration"].median()) if not write_sub.empty else float("nan")
            )

            delta_rows.append(
                {
                    "stage_pair": stage_pair,
                    "duration_all_ms": all_dur_med * 1e3,
                    "duration_write_ms": write_dur_med * 1e3,
                    "duration_delta_pct": _delta_pct(write_dur_med, all_dur_med),
                    "gap_all_ms": all_gap_med * 1e3,
                    "gap_write_ms": write_gap_med * 1e3,
                    "gap_delta_pct": _delta_pct(write_gap_med, all_gap_med),
                    "fct_all_ms": all_fct_med * 1e3,
                    "fct_write_ms": write_fct_med * 1e3,
                    "fct_delta_pct": _delta_pct(write_fct_med, all_fct_med),
                }
            )

        delta_df = pd.DataFrame(delta_rows)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Pipeline Parallelism RoCE Analysis\n\n")
            f.write("## Scope\n")
            f.write(f"- Input pcap: {self.cfg.pcap_path}\n")
            f.write(f"- Total RoCE capture duration: {total_duration:.3f} s\n")
            f.write(
                f"- Selected analysis window: [{win_start:.3f}, {win_end:.3f}] s (duration {sel_duration:.3f} s)\n"
            )
            f.write(f"- Pipeline nodes used ({node_mode}): {', '.join(nodes)}\n\n")
            f.write(f"- Segmentation mode: {segmentation_mode}\n")
            f.write(f"- Segmentation reference stage: {segmentation_reference_stage}\n")
            if segmentation_mode == "steps":
                end_step = self.cfg.start_step + self.cfg.max_steps - 1
                f.write(
                    f"- Analyzed full steps (forward+backward): {self.cfg.start_step}..{end_step}\n"
                )
            if len(step_edges) >= 2:
                f.write(
                    "- Step boundaries (window-relative s): "
                    + ", ".join(f"{x:.6f}" for x in step_edges)
                    + "\n"
                )
            f.write("\n")

            f.write("## 1) Traffic Structure Metrics (Pipeline-Aware)\n")
            f.write("- Stage ordering inferred from directed flows between consecutive nodes.\n")
            f.write(
                "- Phase views: all-opcodes and strict bulk-write (reconstructed WRITE messages with FIRST/MIDDLE/LAST/ONLY + bulk thresholds).\n"
            )
            if not phases.empty:
                num_phases = phases.groupby("stage_pair")["phase_id"].max().to_dict()
                f.write(f"- Phases detected per stage: {num_phases}\n")
            else:
                f.write("- No communication phases detected with current thresholds.\n")
            if not phases_write.empty:
                num_phases_w = phases_write.groupby("stage_pair")["phase_id"].max().to_dict()
                f.write(f"- Strict bulk-write phases detected per stage: {num_phases_w}\n")
            f.write("\n")

            f.write("## 2) Throughput Metrics\n")
            f.write(f"- Average bitrate in selected window: {avg_bitrate_sel:.2f} Mbps\n")
            f.write(
                f"- Link capacity reference: {self.cfg.link_capacity_gbps:.2f} Gbps ({cap_mbps:.0f} Mbps)\n"
            )
            f.write(
                f"- Aggregate utilization: avg={avg_util_total_pct:.2f}%, peak_1ms={peak_util_total_pct:.2f}% (peak={agg_peak_mbps:.2f} Mbps)\n"
            )
            if not util_df.empty:
                for _, r in util_df.iterrows():
                    f.write(
                        f"- {r['stage_pair']}: raw={r['raw_mbps']:.2f} Mbps, goodput={r['goodput_mbps']:.2f} Mbps, "
                        f"peak_1ms={r['peak_1ms_mbps']:.2f} Mbps, avg_util={r['avg_util_pct']:.2f}%, "
                        f"peak_util={r['peak_util_pct']:.2f}%\n"
                    )
            f.write("\n")

            f.write("## 3) Latency / Jitter Metrics\n")
            if len(ipg):
                f.write(f"- Inter-packet latency p99: {p99_ipg:.3f} us\n")
                f.write(f"- Inter-packet latency p99.9: {p999_ipg:.3f} us\n")
            else:
                f.write("- IPG stats unavailable (insufficient packets).\n")
            f.write("\n")

            f.write("## 4) Congestion / Reliability Signals\n")
            f.write(f"- CNP packets: {cnp_count}\n")
            f.write(f"- ECN-CE packets: {ecn_ce_count}\n")
            f.write(f"- PFC events: {pfc_count}\n")
            if retrans.empty:
                f.write("- PSN retransmission signals: none detected\n")
            else:
                rc = retrans["event"].value_counts().to_dict()
                f.write(f"- PSN retransmission signals: {rc}\n")
            f.write(
                "- NACK decoding is not reliable with header-only capture unless full AETH is preserved for all ACKs.\n\n"
            )

            f.write("## 5) Flow Completion Time (FCT)\n")
            if fct_stats.empty:
                f.write("- FCT unavailable (no phases).\n")
            else:
                f.write("- All-opcodes FCT:\n")
                for stage_pair, r in fct_stats.iterrows():
                    f.write(
                        f"- {stage_pair}: count={int(r['count'])}, mean={r['mean']*1e3:.3f} ms, "
                        f"median={r['median']*1e3:.3f} ms, max={r['max']*1e3:.3f} ms\n"
                    )
            if fct_stats_write.empty:
                f.write("- Strict bulk-write FCT unavailable (no bulk messages passed thresholds).\n")
            else:
                f.write("- Strict bulk-write FCT:\n")
                for stage_pair, r in fct_stats_write.iterrows():
                    f.write(
                        f"- {stage_pair}: count={int(r['count'])}, mean={r['mean']*1e3:.3f} ms, "
                        f"median={r['median']*1e3:.3f} ms, max={r['max']*1e3:.3f} ms\n"
                    )
            if not delta_df.empty:
                f.write("\n## 6) All-opcodes vs Strict Bulk-Write Delta\n")
                f.write(
                    "- Deltas are computed as (strict-bulk-write - all-opcodes) / all-opcodes. Negative means strict bulk is smaller.\n"
                )
                for _, r in delta_df.iterrows():
                    if not np.isfinite(float(r["duration_write_ms"])):
                        continue
                    f.write(
                        f"- {r['stage_pair']}: duration {r['duration_all_ms']:.3f}->{r['duration_write_ms']:.3f} ms "
                        f"({r['duration_delta_pct']:+.1f}%), gap {r['gap_all_ms']:.3f}->{r['gap_write_ms']:.3f} ms "
                        f"({r['gap_delta_pct']:+.1f}%), FCT {r['fct_all_ms']:.3f}->{r['fct_write_ms']:.3f} ms "
                        f"({r['fct_delta_pct']:+.1f}%)\n"
                    )
            f.write("\n")

            f.write("## Practical Advice\n")
            f.write("- Keep NIC and switch ECN/PFC counters synchronized with packet captures for stronger congestion attribution.\n")
            f.write("- Add host-side training step timestamps (or NCCL traces) to align communication phases with forward/backward boundaries.\n")
            f.write("- If possible, capture full RoCE headers including AETH to improve NACK/retry diagnosis.\n")
            f.write("- For cross-run comparisons, keep identical analysis window length and bin sizes.\n")

        # Export machine-readable metrics.
        if not phases.empty:
            phases.to_csv(self.output_dir / "phase_metrics.csv", index=False)

        if not phases_write.empty:
            phases_write.to_csv(
                self.output_dir / "phase_metrics_write_bulk_strict.csv", index=False
            )

        if not delta_df.empty:
            delta_df.to_csv(self.output_dir / "phase_metrics_delta_comparison.csv", index=False)

        if not retrans.empty:
            retrans.to_csv(self.output_dir / "retransmission_events.csv", index=False)

        if not util_df.empty:
            util_df.to_csv(self.output_dir / "throughput_utilization_metrics.csv", index=False)

        self.export_per_step_pair_traffic_wire(
            df, self.output_dir / "per_step_pair_traffic_wire.csv"
        )

    def run(self) -> None:
        print(f"[*] Loading capture: {self.cfg.pcap_path}")
        all_df, pfc_count = self.load_packets()

        nodes, node_mode = self.resolve_pipeline_nodes(all_df)
        print(f"[*] Pipeline nodes ({node_mode}): {', '.join(nodes)}")

        pipe_df = self.filter_pipeline_traffic(all_df, nodes)
        if pipe_df.empty:
            raise RuntimeError("No adjacent pipeline traffic found for selected nodes.")

        if self.cfg.window_mode == "steps":
            step_window = self.detect_step_window(pipe_df)
            win_start, win_end = step_window.start, step_window.end
            step_edges_abs = step_window.boundaries
            seg_mode = step_window.mode
            ref_stage = step_window.reference_stage
        else:
            win_start, win_end = self.detect_analysis_window(pipe_df)
            step_edges_abs = [win_start, win_end]
            seg_mode = "time"
            ref_stage = "n/a"

        sel = pipe_df[(pipe_df["rel_time"] >= win_start) & (pipe_df["rel_time"] <= win_end)].copy()
        sel["rel_time"] = sel["rel_time"] - win_start
        step_edges = [x - win_start for x in step_edges_abs]

        if len(step_edges) >= 2:
            bins = np.asarray(step_edges, dtype=float)
            sel["step_id"] = np.digitize(sel["rel_time"].to_numpy(), bins[1:], right=False) + 1
            max_step = max(1, len(step_edges) - 1)
            sel["step_id"] = sel["step_id"].clip(1, max_step)
        else:
            sel["step_id"] = 1

        print(
            f"[*] Selected window: [{win_start:.3f}, {win_end:.3f}] s "
            f"({win_end - win_start:.3f} s), packets={len(sel):,}, mode={seg_mode}"
        )
        if self.cfg.window_mode == "steps":
            print(f"[*] Step reference stage: {ref_stage}")

        phases = self.detect_phases(sel, metric_mode="all_opcodes")
        phases_write = self.detect_bulk_write_phases(sel, metric_mode="write_bulk_strict")
        if not phases.empty and len(step_edges) >= 2:
            bins = np.asarray(step_edges, dtype=float)
            phase_step = np.digitize(phases["start"].to_numpy(dtype=float), bins[1:], right=False) + 1
            phases["step_id"] = np.clip(phase_step, 1, max(1, len(step_edges) - 1))
        if not phases_write.empty and len(step_edges) >= 2:
            bins = np.asarray(step_edges, dtype=float)
            phase_step_w = np.digitize(
                phases_write["start"].to_numpy(dtype=float), bins[1:], right=False
            ) + 1
            phases_write["step_id"] = np.clip(
                phase_step_w, 1, max(1, len(step_edges) - 1)
            )
        retrans = self.estimate_retransmissions(sel)

        # Group 1: Traffic structure
        self.plot_packet_timeline(
            sel, self.plot_dirs["traffic"] / "01_packet_timeline.png", step_edges=step_edges
        )
        self.plot_phase_durations_dual(phases, phases_write)
        self.plot_inter_phase_gaps_dual(phases, phases_write)
        self.plot_periodicity(phases, self.plot_dirs["traffic"] / "04_periodicity_cycle_time.png")
        bulk_step_df = self.export_per_step_stage_bulk_summary(phases_write, step_edges)
        self.plot_per_step_stage_bulk_summary(
            bulk_step_df,
            self.plot_dirs["traffic"] / "05_messages_active_time_per_step_strict_bulk.png",
        )

        # Group 2: Throughput
        self.plot_bitrate(
            sel, self.plot_dirs["throughput"] / "05_bitrate.png", step_edges=step_edges
        )
        self.plot_goodput_raw(
            phases, self.plot_dirs["throughput"] / "06_raw_vs_goodput.png"
        )

        # Group 3: Latency
        self.plot_interpacket_cdf(
            sel, self.plot_dirs["latency"] / "07_interpacket_cdf.png"
        )
        self.plot_jitter(sel, self.plot_dirs["latency"] / "08_jitter.png")
        self.plot_tail_latency(
            sel, phases, self.plot_dirs["latency"] / "09_tail_latency.png"
        )
        self.plot_ipg_timeseries(
            sel, self.plot_dirs["latency"] / "10_ipg_timeseries.png", step_edges=step_edges
        )
        self.plot_fct_dual(phases, phases_write)

        # Group 4: Congestion and retransmissions
        self.plot_congestion(
            sel, retrans, pfc_count, self.plot_dirs["congestion"] / "12_congestion_events.png"
        )
        self.plot_opcode_distribution(
            sel,
            self.plot_dirs["congestion"] / "13_opcode_distribution_all.png",
            "4.2 Opcode Distribution (All packets)",
        )
        self.plot_opcode_distribution(
            sel[sel["opcode"].isin(RDMA_WRITE_OPCODES)].copy(),
            self.plot_dirs["congestion"] / "14_opcode_distribution_write_only.png",
            "4.3 Opcode Distribution (RDMA write only)",
        )

        self.write_summary(
            all_df=all_df,
            df=sel,
            phases=phases,
            phases_write=phases_write,
            retrans=retrans,
            pfc_count=pfc_count,
            nodes=nodes,
            node_mode=node_mode,
            win_start=win_start,
            win_end=win_end,
            step_edges=step_edges,
            segmentation_mode=seg_mode,
            segmentation_reference_stage=ref_stage,
        )

        print(f"[+] Analysis complete. Outputs saved to: {self.output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RoCE pipeline-parallelism analyzer")
    p.add_argument("-i", "--input", required=True, help="Input pcap path")
    p.add_argument("-o", "--output", required=True, help="Output directory")
    p.add_argument("--start-ip", default="192.168.2.101", help="First node IP")
    p.add_argument("--end-ip", default="192.168.2.107", help="Last node IP")
    p.add_argument(
        "--auto-window-sec",
        type=float,
        default=4.0,
        help="Duration of selected active window",
    )
    p.add_argument(
        "--bitrate-bin-ms", type=float, default=1.0, help="Bitrate bin size in ms"
    )
    p.add_argument(
        "--active-bin-ms",
        type=float,
        default=10.0,
        help="Coarser bin for active-window detection",
    )
    p.add_argument(
        "--min-phase-gap-us",
        type=float,
        default=500.0,
        help="Minimum gap used to split communication phases",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=2,
        help="Number of pipeline steps to include when --window-mode=steps",
    )
    p.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="1-based first full step index to analyze when --window-mode=steps",
    )
    p.add_argument(
        "--window-mode",
        choices=["steps", "time"],
        default="steps",
        help="Window selection mode: exact step-based window or fixed active time window",
    )
    p.add_argument(
        "--link-capacity-gbps",
        type=float,
        default=10.0,
        help="Reference link capacity used for utilization percentages",
    )
    p.add_argument(
        "--bulk-min-packets",
        type=int,
        default=128,
        help="Strict bulk detector threshold: minimum packets per reconstructed write message",
    )
    p.add_argument(
        "--bulk-min-bytes",
        type=int,
        default=1000000,
        help="Strict bulk detector threshold: minimum wire bytes per reconstructed write message",
    )
    p.add_argument(
        "--top-flows",
        type=int,
        default=24,
        help="Reserved for future flow-level expansions",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not os.path.isfile(args.input):
        print(f"[!] Input file not found: {args.input}")
        sys.exit(1)

    cfg = AnalyzerConfig(
        pcap_path=args.input,
        output_dir=args.output,
        start_ip=args.start_ip,
        end_ip=args.end_ip,
        auto_window_sec=args.auto_window_sec,
        max_steps=args.max_steps,
        bitrate_bin_ms=args.bitrate_bin_ms,
        active_bin_ms=args.active_bin_ms,
        min_phase_gap_us=args.min_phase_gap_us,
        top_flows=args.top_flows,
        window_mode=args.window_mode,
        start_step=args.start_step,
        link_capacity_gbps=args.link_capacity_gbps,
        bulk_min_packets=args.bulk_min_packets,
        bulk_min_bytes=args.bulk_min_bytes,
    )

    try:
        PipelineAnalyzer(cfg).run()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[!] Analysis failed: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
