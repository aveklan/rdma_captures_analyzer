#!/usr/bin/env python3
"""
RoCE v2 RDMA Traffic Analyzer for Distributed Data Parallel (DDP) Training
==========================================================================
Analyzes header-only pcap captures of RoCE v2 traffic between GPU nodes
during DDP training (forward, backward, allreduce phases).

Requirements:
    pip install scapy matplotlib numpy pandas

Usage:
    python analyze_roce.py

Input:  rdma_headers.pcap  (header-only capture, ~128 bytes per packet)
Output: Multiple PNG plots characterizing the RDMA flows.
"""

import os
import sys
import struct
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Configuration setup
# ---------------------------------------------------------------------------
PCAP_FILE = "rdma_headers.pcap"

# Node IP addresses
NODES = {
    "PC1 (master)": "192.168.2.100",
    "PC2": "192.168.2.101",
    "PC3": "192.168.2.102",
    "PC4": "192.168.2.103",
}

# Flows we expect to see (bidirectional)
EXPECTED_FLOWS = [
    ("192.168.2.100", "192.168.2.101"),
    ("192.168.2.101", "192.168.2.102"),
]

# RoCE v2 uses UDP destination port 4791
ROCE_UDP_DST_PORT = 4791
ROCEV1_ETHERTYPE = 0x8915

# Time-bin width for rate / throughput calculations (seconds)
BIN_WIDTH = 0.001  # 1 ms bins  — adjust to 0.01 or 0.1 for coarser view

# Original packet length heuristic: if the pcap snap length is short
# (header-only capture), the *wire length* stored in the pcap record header
# is still the real frame length.  We use that for throughput estimates.

# ---------------------------------------------------------------------------
# Low-level pcap parser  (avoids heavy scapy rdpcap on large files)
# ---------------------------------------------------------------------------


def read_pcap(path):
    """
    Yield (timestamp_sec, orig_len, raw_bytes) for every record in a
    pcap (little-endian or big-endian, both link types Ethernet=1).
    """
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

        _ver_maj, _ver_min, _thiszone, _sigfigs, snaplen, network = struct.unpack(
            endian + "HHiIII", ghdr[4:]
        )
        if network not in (1, 101):  # LINKTYPE_ETHERNET or LINKTYPE_RAW
            warnings.warn(f"Link-layer type {network} — assuming Ethernet anyway")

        rec_hdr_fmt = endian + "IIII"
        rec_hdr_size = 16

        while True:
            rhdr = f.read(rec_hdr_size)
            if len(rhdr) < rec_hdr_size:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(rec_hdr_fmt, rhdr)
            if ts_format == "nano":
                timestamp = ts_sec + ts_usec * 1e-9
            else:
                timestamp = ts_sec + ts_usec * 1e-6
            raw = f.read(incl_len)
            if len(raw) < incl_len:
                break
            yield timestamp, orig_len, raw, network


# ---------------------------------------------------------------------------
# Packet-level field extraction
# ---------------------------------------------------------------------------

# BTH opcode names (subset for RC transport — most common in NCCL)
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
    0x13: "RC_COMPARE_SWAP",
    0x14: "RC_FETCH_ADD",
    0x19: "RC_SEND_LAST_INV",
    # UD
    0x64: "UD_SEND_ONLY",
    0x65: "UD_SEND_ONLY_IMM",
    # CNP (Congestion Notification Packet)
    0x81: "CNP",
}


def _ipv4_from_mapped_gid(gid_bytes):
    """
    Decode IPv4 from IPv4-mapped IPv6 GID (::ffff:a.b.c.d), else return None.
    """
    if len(gid_bytes) != 16:
        return None
    if gid_bytes[:10] == b"\x00" * 10 and gid_bytes[10:12] == b"\xff\xff":
        return ".".join(str(b) for b in gid_bytes[12:16])
    return None


def parse_packet(raw, link_type=1):
    """
    Parse Ethernet / IPv4 / UDP / BTH from a (possibly truncated) frame.
    Returns a dict with extracted fields, or None if not a valid RoCE v2 pkt.
    """
    info = {}
    offset = 0

    if link_type != 1:
        return None

    # --- Ethernet ---
    if len(raw) < 14:
        return None
    ethertype = struct.unpack("!H", raw[12:14])[0]

    # Handle VLAN tag (802.1Q)
    if ethertype == 0x8100:
        if len(raw) < 18:
            return None
        ethertype = struct.unpack("!H", raw[16:18])[0]
        offset = 18
    else:
        offset = 14

    # --- RoCE v1 over Ethernet (Ethertype 0x8915) ---
    if ethertype == ROCEV1_ETHERTYPE:
        info["roce_version"] = "v1"

        # In RoCEv1, payload starts with GRH (40 bytes), then BTH.
        # If packet is truncated, keep partial information when possible.
        if len(raw) >= offset + 40:
            grh = raw[offset : offset + 40]
            src_gid = grh[8:24]
            dst_gid = grh[24:40]
            info["src_ip"] = _ipv4_from_mapped_gid(src_gid)
            info["dst_ip"] = _ipv4_from_mapped_gid(dst_gid)
            info["src_gid"] = src_gid.hex(":")
            info["dst_gid"] = dst_gid.hex(":")
            bth_offset = offset + 40
        else:
            info["src_ip"] = None
            info["dst_ip"] = None
            info["src_gid"] = None
            info["dst_gid"] = None
            bth_offset = offset

        if len(raw) < bth_offset + 12:
            info["opcode"] = None
            info["opcode_name"] = "UNKNOWN_TRUNCATED"
            info["dest_qp"] = None
            info["psn"] = None
            info["ack_req"] = None
            return info

        bth = raw[bth_offset : bth_offset + 12]
        opcode = bth[0]
        padcount = (bth[1] >> 4) & 0x03
        dest_qp = struct.unpack("!I", b"\x00" + bth[4:7])[0]
        psn = struct.unpack("!I", b"\x00" + bth[8:11])[0]  # lower 24 bits
        ack_req = (bth[11] >> 7) & 0x01

        info["opcode"] = opcode
        info["opcode_name"] = BTH_OPCODES.get(opcode, f"UNKNOWN_0x{opcode:02X}")
        info["padcount"] = padcount
        info["dest_qp"] = dest_qp
        info["psn"] = psn
        info["ack_req"] = ack_req
        return info

    # --- RoCE v2 over IPv4/UDP ---
    if ethertype != 0x0800:  # IPv4 only for RoCEv2
        return None

    # --- IPv4 ---
    if len(raw) < offset + 20:
        return None
    ip_hdr = raw[offset : offset + 20]
    ihl = (ip_hdr[0] & 0x0F) * 4
    total_length = struct.unpack("!H", ip_hdr[2:4])[0]
    protocol = ip_hdr[9]
    src_ip = f"{ip_hdr[12]}.{ip_hdr[13]}.{ip_hdr[14]}.{ip_hdr[15]}"
    dst_ip = f"{ip_hdr[16]}.{ip_hdr[17]}.{ip_hdr[18]}.{ip_hdr[19]}"

    info["roce_version"] = "v2"
    info["src_ip"] = src_ip
    info["dst_ip"] = dst_ip
    info["ip_total_length"] = total_length
    tos = ip_hdr[1]
    info["ip_dscp"] = (tos >> 2) & 0x3F
    info["ip_ecn"] = tos & 0x03

    if protocol != 17:  # UDP
        return None

    offset += ihl

    # --- UDP ---
    if len(raw) < offset + 8:
        return None
    udp_hdr = raw[offset : offset + 8]
    src_port, dst_port, udp_len = struct.unpack("!HHH", udp_hdr[:6])
    info["udp_src_port"] = src_port
    info["udp_dst_port"] = dst_port

    if dst_port != ROCE_UDP_DST_PORT:
        return None  # not RoCE v2

    offset += 8

    # --- BTH (Base Transport Header) — 12 bytes ---
    if len(raw) < offset + 12:
        # Still record as RoCE even without full BTH
        info["opcode"] = None
        info["dest_qp"] = None
        info["psn"] = None
        return info

    bth = raw[offset : offset + 12]
    opcode = bth[0]
    padcount = (bth[1] >> 4) & 0x03
    dest_qp = struct.unpack("!I", b"\x00" + bth[4:7])[0]
    psn = struct.unpack("!I", b"\x00" + bth[8:11])[0]  # lower 24 bits
    ack_req = (bth[11] >> 7) & 0x01

    info["opcode"] = opcode
    info["opcode_name"] = BTH_OPCODES.get(opcode, f"UNKNOWN_0x{opcode:02X}")
    info["padcount"] = padcount
    info["dest_qp"] = dest_qp
    info["psn"] = psn
    info["ack_req"] = ack_req

    return info


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def load_and_parse(pcap_path):
    """Read pcap, parse every packet, return a DataFrame."""
    records = []
    pkt_idx = 0
    for ts, orig_len, raw, link_type in read_pcap(pcap_path):
        info = parse_packet(raw, link_type=link_type)
        if info is not None:
            info["timestamp"] = ts
            info["orig_len"] = orig_len  # real wire length (even if truncated)
            info["pkt_idx"] = pkt_idx
            records.append(info)
        pkt_idx += 1

    if not records:
        print(
            "[!] No RoCE packets found. Check capture format, link-layer type, or snap length."
        )
        sys.exit(1)

    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # relative time from first packet
    t0 = df["timestamp"].iloc[0]
    df["rel_time"] = df["timestamp"] - t0
    return df


def identify_flow(row):
    """Return a canonical (sorted) flow key as 'A <-> B'."""
    src = row.get("src_ip")
    dst = row.get("dst_ip")
    if src is None or dst is None:
        src = row.get("src_gid") or "unknown_src"
        dst = row.get("dst_gid") or "unknown_dst"
    a, b = sorted([src, dst])
    return f"{a} <-> {b}"


def identify_direction(row):
    """Return directed flow as 'A -> B'."""
    src = row.get("src_ip")
    dst = row.get("dst_ip")
    if src is None or dst is None:
        src = row.get("src_gid") or "unknown_src"
        dst = row.get("dst_gid") or "unknown_dst"
    return f"{src} -> {dst}"


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _nice_time_axis(ax, xlabel="Time (s)"):
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=150)
    plt.close(fig)
    print(f"  [+] Saved {name}")


# ============================  PLOTS  =====================================


def plot_packet_timeline(df):
    """Scatter plot: packet arrival time vs flow, colored by direction."""
    fig, ax = plt.subplots(figsize=(14, 4))
    flows = df["flow"].unique()
    colors = plt.cm.tab10.colors

    for i, flow in enumerate(flows):
        sub = df[df["flow"] == flow]
        for j, direction in enumerate(sub["direction"].unique()):
            dsub = sub[sub["direction"] == direction]
            ax.scatter(
                dsub["rel_time"],
                [i] * len(dsub),
                s=0.3,
                alpha=0.5,
                label=direction,
                color=colors[(2 * i + j) % len(colors)],
            )

    ax.set_yticks(range(len(flows)))
    ax.set_yticklabels(flows)
    ax.set_title("Packet Arrival Timeline per Flow")
    _nice_time_axis(ax)
    ax.legend(loc="upper right", fontsize=7, markerscale=5)
    _save(fig, "01_packet_timeline.png")


def plot_bitrate(df, bin_width=BIN_WIDTH):
    """
    Bitrate (Mbps) over time per directed flow.
    Uses orig_len (the real wire length from the pcap record header).
    """
    fig, axes = plt.subplots(
        len(df["flow"].unique()),
        1,
        figsize=(14, 4 * len(df["flow"].unique())),
        sharex=True,
    )
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    max_t = df["rel_time"].max()
    bins = np.arange(0, max_t + bin_width, bin_width)

    for ax, flow in zip(axes, df["flow"].unique()):
        sub = df[df["flow"] == flow]
        for direction in sorted(sub["direction"].unique()):
            dsub = sub[sub["direction"] == direction]
            byte_counts, _ = np.histogram(
                dsub["rel_time"], bins=bins, weights=dsub["orig_len"]
            )
            bitrate_mbps = (byte_counts * 8) / (bin_width * 1e6)
            ax.plot(bins[:-1], bitrate_mbps, linewidth=0.6, label=direction)

        ax.set_ylabel("Bitrate (Mbps)")
        ax.set_title(f"Bitrate — {flow}")
        ax.legend(fontsize=8)
        _nice_time_axis(ax)

    _save(fig, "02_bitrate.png")


def plot_throughput(df, bin_width=BIN_WIDTH):
    """
    Throughput (MB/s) over time — aggregate and per flow.
    Throughput here = total payload bytes / time.
    Since we only have headers, we estimate payload from
    orig_len - 14 (Eth) - 20 (IP) - 8 (UDP) - 12 (BTH) - 4 (ICRC) = orig_len - 58
    """
    HEADER_OVERHEAD = 58  # Eth(14)+IP(20)+UDP(8)+BTH(12)+ICRC(4)

    fig, axes = plt.subplots(
        1 + len(df["flow"].unique()),
        1,
        figsize=(14, 4 * (1 + len(df["flow"].unique()))),
        sharex=True,
    )

    max_t = df["rel_time"].max()
    bins = np.arange(0, max_t + bin_width, bin_width)

    # Aggregate
    df["payload_est"] = (df["orig_len"] - HEADER_OVERHEAD).clip(lower=0)
    byte_counts, _ = np.histogram(df["rel_time"], bins=bins, weights=df["payload_est"])
    throughput_mbytes = byte_counts / (bin_width * 1e6)
    axes[0].plot(bins[:-1], throughput_mbytes, linewidth=0.6, color="black")
    axes[0].set_ylabel("Throughput (MB/s)")
    axes[0].set_title("Aggregate Estimated Throughput (all flows)")
    _nice_time_axis(axes[0])

    for ax, flow in zip(axes[1:], df["flow"].unique()):
        sub = df[df["flow"] == flow]
        for direction in sorted(sub["direction"].unique()):
            dsub = sub[sub["direction"] == direction]
            payload = (dsub["orig_len"] - HEADER_OVERHEAD).clip(lower=0)
            bc, _ = np.histogram(dsub["rel_time"], bins=bins, weights=payload)
            tp = bc / (bin_width * 1e6)
            ax.plot(bins[:-1], tp, linewidth=0.6, label=direction)
        ax.set_ylabel("Throughput (MB/s)")
        ax.set_title(f"Estimated Throughput — {flow}")
        ax.legend(fontsize=8)
        _nice_time_axis(ax)

    _save(fig, "03_throughput.png")


def plot_inter_packet_latency(df):
    """
    Inter-packet gap (latency proxy) per directed flow.
    Shows CDF + time-series.
    """
    directions = sorted(df["direction"].unique())
    fig, axes = plt.subplots(len(directions), 2, figsize=(14, 3.5 * len(directions)))
    if len(directions) == 1:
        axes = axes.reshape(1, -1)

    for i, direction in enumerate(directions):
        sub = df[df["direction"] == direction].sort_values("rel_time")
        ipg = sub["rel_time"].diff().dropna().values * 1e6  # µs

        # Time-series
        axes[i, 0].plot(sub["rel_time"].iloc[1:].values, ipg, linewidth=0.3, alpha=0.7)
        axes[i, 0].set_ylabel("Inter-packet gap (µs)")
        axes[i, 0].set_title(f"Inter-packet latency — {direction}")
        _nice_time_axis(axes[i, 0])

        # CDF
        sorted_ipg = np.sort(ipg)
        cdf = np.arange(1, len(sorted_ipg) + 1) / len(sorted_ipg)
        axes[i, 1].plot(sorted_ipg, cdf, linewidth=1)
        axes[i, 1].set_xlabel("Inter-packet gap (µs)")
        axes[i, 1].set_ylabel("CDF")
        axes[i, 1].set_title(f"CDF of inter-packet gap — {direction}")
        axes[i, 1].set_xscale("log")
        axes[i, 1].grid(True, alpha=0.3)

    _save(fig, "04_latency_inter_packet.png")


def plot_ack_latency(df):
    """
    Estimate RDMA round-trip from SEND/WRITE → ACK pairing per QP.
    For each ACK, find the most recent data packet on the reverse
    direction with the matching QP, and compute the time delta.
    """
    rtt_df = estimate_ack_rtt(df)

    if rtt_df.empty:
        print("  [!] No ACK-based RTT estimates found — skipping ACK latency plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time-series
    for flow in rtt_df["flow"].unique():
        sub = rtt_df[rtt_df["flow"] == flow]
        axes[0].scatter(sub["ack_time"], sub["rtt_us"], s=1, alpha=0.5, label=flow)
    axes[0].set_ylabel("Estimated RTT (µs)")
    axes[0].set_title("ACK-based RTT Estimation Over Time")
    axes[0].legend(fontsize=8, markerscale=5)
    _nice_time_axis(axes[0])

    # Histogram
    for flow in rtt_df["flow"].unique():
        sub = rtt_df[rtt_df["flow"] == flow]
        axes[1].hist(sub["rtt_us"], bins=100, alpha=0.6, label=flow)
    axes[1].set_xlabel("Estimated RTT (µs)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of ACK-based RTT")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    _save(fig, "05_ack_latency.png")


def estimate_ack_rtt(df):
    """
    Estimate RDMA round-trip from SEND/WRITE → ACK pairing per QP.
    For each ACK, find the most recent data packet on the reverse
    direction with the matching QP, and compute the time delta.
    Returns DataFrame(flow, direction, ack_time, rtt_us).
    """
    ack_opcodes = {0x11, 0x12}  # RC_ACKNOWLEDGE, RC_ATOMIC_ACKNOWLEDGE
    data_opcodes = set(range(0x00, 0x11)) - ack_opcodes

    results = []

    for flow in df["flow"].unique():
        sub = df[df["flow"] == flow].copy()
        for direction in sub["direction"].unique():
            dsub = sub[sub["direction"] == direction].sort_values("rel_time")
            # reverse direction
            rev = direction.split(" -> ")
            rev_dir = f"{rev[1]} -> {rev[0]}"
            rev_sub = sub[sub["direction"] == rev_dir].sort_values("rel_time")

            acks = dsub[dsub["opcode"].isin(ack_opcodes)]
            data_pkts = rev_sub[rev_sub["opcode"].isin(data_opcodes)]

            if acks.empty or data_pkts.empty:
                continue

            data_times = data_pkts["rel_time"].values
            for _, ack in acks.iterrows():
                # find closest preceding data packet
                idx = np.searchsorted(data_times, ack["rel_time"], side="right") - 1
                if idx >= 0:
                    rtt_est = ack["rel_time"] - data_times[idx]
                    if 0 < rtt_est < 0.1:  # sanity: < 100 ms
                        results.append(
                            {
                                "flow": flow,
                                "direction": direction,
                                "ack_time": ack["rel_time"],
                                "rtt_us": rtt_est * 1e6,
                            }
                        )
    if not results:
        return pd.DataFrame(columns=["flow", "direction", "ack_time", "rtt_us"])
    return pd.DataFrame(results)


def _is_psn_wrap(prev_psn, curr_psn):
    """Heuristic for 24-bit PSN wrap-around."""
    return prev_psn > 0xF00000 and curr_psn < 0x0FFFFF


def estimate_retransmission_signals(df):
    """
    Detect likely retransmission / reordering signals from non-monotonic PSN.
    Returns DataFrame(direction, flow, dest_qp, rel_time, prev_psn, psn, event).
    """
    cols = [
        "direction",
        "flow",
        "dest_qp",
        "rel_time",
        "prev_psn",
        "psn",
        "event",
    ]
    psn_df = df.dropna(subset=["dest_qp", "psn"]).copy()
    if psn_df.empty:
        return pd.DataFrame(columns=cols)

    events = []
    grouped = psn_df.sort_values("rel_time").groupby(["direction", "dest_qp"])
    for (direction, qp), sub in grouped:
        prev_psn = None
        flow = (
            sub["flow"].iloc[0]
            if "flow" in sub.columns and not sub.empty
            else "unknown"
        )
        for _, row in sub.iterrows():
            curr_psn = int(row["psn"])
            if prev_psn is not None:
                if curr_psn == prev_psn:
                    events.append(
                        {
                            "direction": direction,
                            "flow": flow,
                            "dest_qp": int(qp),
                            "rel_time": row["rel_time"],
                            "prev_psn": prev_psn,
                            "psn": curr_psn,
                            "event": "duplicate_psn",
                        }
                    )
                elif curr_psn < prev_psn and not _is_psn_wrap(prev_psn, curr_psn):
                    events.append(
                        {
                            "direction": direction,
                            "flow": flow,
                            "dest_qp": int(qp),
                            "rel_time": row["rel_time"],
                            "prev_psn": prev_psn,
                            "psn": curr_psn,
                            "event": "psn_regression",
                        }
                    )
            prev_psn = curr_psn

    if not events:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(events)


def print_congestion_assessment(df):
    """Print congestion indicators and an overall assessment."""
    cnp_count = int((df["opcode"] == 0x81).sum()) if "opcode" in df.columns else 0
    ce_count = int((df.get("ip_ecn", pd.Series(dtype=float)) == 3).sum())

    retrans_df = estimate_retransmission_signals(df)
    retrans_count = len(retrans_df)
    retrans_ratio = retrans_count / max(len(df), 1)

    ipg = df["rel_time"].diff().dropna().values * 1e6
    if len(ipg) > 0:
        ipg_median = float(np.median(ipg))
        ipg_p99 = float(np.percentile(ipg, 99))
        ipg_p999 = float(np.percentile(ipg, 99.9)) if len(ipg) > 1000 else np.nan
    else:
        ipg_median = np.nan
        ipg_p99 = np.nan
        ipg_p999 = np.nan

    rtt_df = estimate_ack_rtt(df)
    ack_rtt_p99 = (
        float(np.percentile(rtt_df["rtt_us"], 99)) if not rtt_df.empty else np.nan
    )

    score = 0
    if cnp_count > 0:
        score += 2
    if ce_count > 0:
        score += 2
    if retrans_ratio > 0.001:
        score += 1
    if np.isfinite(ipg_median) and np.isfinite(ipg_p99) and ipg_median > 0:
        if (ipg_p99 / ipg_median) > 20:
            score += 1
    if np.isfinite(ack_rtt_p99) and ack_rtt_p99 > 500:
        score += 1

    if score >= 4:
        verdict = "LIKELY CONGESTION"
    elif score >= 2:
        verdict = "POSSIBLE / INTERMITTENT CONGESTION"
    else:
        verdict = "NO STRONG CONGESTION SIGNAL"

    print()
    print("  Congestion assessment:")
    print("  " + "-" * 66)
    print(f"    Verdict                        : {verdict}")
    print(f"    CNP packets (opcode 0x81)      : {cnp_count:,}")
    print(f"    ECN-CE marked IP packets       : {ce_count:,}")
    print(
        f"    PSN non-monotonic events       : {retrans_count:,} ({retrans_ratio * 100:.4f}% of packets)"
    )
    if np.isfinite(ipg_median):
        print(f"    IPG median                     : {ipg_median:.3f} µs")
        print(f"    IPG p99                        : {ipg_p99:.3f} µs")
        if np.isfinite(ipg_p999):
            print(f"    IPG p99.9                      : {ipg_p999:.3f} µs")
    else:
        print("    IPG stats                      : unavailable")

    if np.isfinite(ack_rtt_p99):
        print(f"    ACK-based RTT p99              : {ack_rtt_p99:.3f} µs")
    else:
        print("    ACK-based RTT p99              : unavailable")

    if cnp_count > 0:
        per_dir_cnp = (
            df[df["opcode"] == 0x81]["direction"]
            .value_counts()
            .sort_values(ascending=False)
        )
        print("    CNP by direction:")
        for direction, count in per_dir_cnp.items():
            print(f"      {direction}: {count:,}")


def print_jumbo_frame_report(df, jumbo_threshold=1518):
    """
    Report whether jumbo frames are present using wire-length (`orig_len`).
    Threshold uses >1518 bytes (covers standard Ethernet + optional VLAN).
    """
    lengths = df["orig_len"].dropna().astype(int)
    if lengths.empty:
        print()
        print("  Jumbo-frame report:")
        print("  " + "-" * 66)
        print("    No packet length data available.")
        return

    jumbo = lengths[lengths > jumbo_threshold]
    jumbo_count = int(len(jumbo))
    total_count = int(len(lengths))
    jumbo_ratio = jumbo_count / max(total_count, 1)

    print()
    print("  Jumbo-frame report:")
    print("  " + "-" * 66)
    print(f"    Jumbo threshold (wire bytes)   : > {jumbo_threshold}")
    print(f"    Max wire length observed       : {int(lengths.max()):,}")
    print(
        f"    Jumbo packets                  : {jumbo_count:,} / {total_count:,} ({jumbo_ratio * 100:.3f}%)"
    )
    if jumbo_count > 0:
        print("    Verdict                        : Jumbo frames are in use")
        top_sizes = jumbo.value_counts().head(10)
        print("    Most common jumbo sizes:")
        for size, cnt in top_sizes.items():
            print(f"      {size:>6} bytes : {cnt:,}")
    else:
        print("    Verdict                        : No jumbo frames detected")


def plot_jitter(df, bin_width=BIN_WIDTH):
    """
    Jitter = variation of inter-packet gap over time.
    Computed as rolling std-dev of inter-packet deltas within bins.
    """
    directions = sorted(df["direction"].unique())
    fig, axes = plt.subplots(
        len(directions), 1, figsize=(14, 4 * len(directions)), sharex=True
    )
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    for ax, direction in zip(axes, directions):
        sub = df[df["direction"] == direction].sort_values("rel_time")
        ipg = sub["rel_time"].diff().dropna().values * 1e6  # µs
        times = sub["rel_time"].iloc[1:].values

        # Rolling jitter (window = 100 packets)
        window = min(100, len(ipg) // 2) if len(ipg) > 10 else len(ipg)
        if window < 2:
            continue
        ipg_series = pd.Series(ipg)
        rolling_std = ipg_series.rolling(window, center=True).std().values

        ax.plot(times, rolling_std, linewidth=0.5, alpha=0.8)
        ax.set_ylabel("Jitter (µs, rolling σ)")
        ax.set_title(
            f"Jitter (rolling std of inter-pkt gap, window={window}) — {direction}"
        )
        _nice_time_axis(ax)

    _save(fig, "06_jitter.png")


def plot_opcode_distribution(df):
    """Bar chart of BTH opcode distribution per flow."""
    flows = df["flow"].unique()
    fig, axes = plt.subplots(1, len(flows), figsize=(7 * len(flows), 5))
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    for ax, flow in zip(axes, flows):
        sub = df[df["flow"] == flow]
        counts = sub["opcode_name"].value_counts()
        counts.plot.barh(ax=ax)
        ax.set_xlabel("Packet count")
        ax.set_title(f"Opcode Distribution — {flow}")
        ax.grid(True, alpha=0.3)

    _save(fig, "07_opcode_distribution.png")


def plot_packet_size_distribution(df):
    """Histogram of orig_len (wire-length) per flow."""
    flows = df["flow"].unique()
    fig, axes = plt.subplots(1, len(flows), figsize=(7 * len(flows), 5))
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    for ax, flow in zip(axes, flows):
        sub = df[df["flow"] == flow]
        ax.hist(sub["orig_len"], bins=80, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Wire-length (bytes)")
        ax.set_ylabel("Count")
        ax.set_title(f"Packet Size Distribution — {flow}")
        ax.grid(True, alpha=0.3)

    _save(fig, "08_packet_size_distribution.png")


def plot_qp_activity(df):
    """Show QP (Queue Pair) usage over time — scatter colored by QP."""
    fig, ax = plt.subplots(figsize=(14, 5))
    qps = df["dest_qp"].dropna().unique()
    cmap = plt.cm.get_cmap("tab20", max(len(qps), 1))
    qp_to_idx = {qp: i for i, qp in enumerate(sorted(qps))}

    for qp in sorted(qps):
        sub = df[df["dest_qp"] == qp]
        ax.scatter(
            sub["rel_time"], sub["dest_qp"], s=0.3, alpha=0.5, color=cmap(qp_to_idx[qp])
        )

    ax.set_ylabel("Destination QP Number")
    ax.set_title("Queue Pair Activity Over Time")
    _nice_time_axis(ax)
    _save(fig, "09_qp_activity.png")


def plot_psn_progression(df):
    """PSN (Packet Sequence Number) over time per QP — helps spot retransmissions."""
    qps = df["dest_qp"].dropna().unique()
    # Limit to top-N busiest QPs
    top_n = 6
    qp_counts = df.groupby("dest_qp").size().sort_values(ascending=False)
    top_qps = qp_counts.head(top_n).index

    fig, axes = plt.subplots(
        min(len(top_qps), top_n),
        1,
        figsize=(14, 3 * min(len(top_qps), top_n)),
        sharex=True,
    )
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    for ax, qp in zip(axes, top_qps):
        sub = df[df["dest_qp"] == qp].sort_values("rel_time")
        ax.plot(sub["rel_time"], sub["psn"], linewidth=0.4)
        ax.set_ylabel("PSN")
        ax.set_title(f"PSN Progression — QP {qp:#x} ({len(sub)} pkts)")
        _nice_time_axis(ax)

    _save(fig, "10_psn_progression.png")


def print_summary(df):
    """Print key summary statistics."""
    duration = df["rel_time"].max() - df["rel_time"].min()
    total_bytes = df["orig_len"].sum()
    safe_duration = duration if duration > 0 else 1e-9

    print("\n" + "=" * 70)
    print("  RoCE v2 RDMA Traffic Summary")
    print("=" * 70)
    print(f"  Total packets (RoCE)    : {len(df):,}")
    print(f"  Capture duration        : {duration:.6f} s")
    print(
        f"  Total wire bytes        : {total_bytes:,.0f}  ({total_bytes / 1e6:.2f} MB)"
    )
    print(
        f"  Average bitrate         : {total_bytes * 8 / safe_duration / 1e6:.2f} Mbps"
    )
    if "roce_version" in df.columns:
        version_counts = df["roce_version"].value_counts()
        print(
            f"  RoCE version mix        : "
            + ", ".join(f"{k}={v:,}" for k, v in version_counts.items())
        )
    print()

    print("  Per-flow breakdown:")
    print("  " + "-" * 66)
    for flow in sorted(df["flow"].unique()):
        sub = df[df["flow"] == flow]
        fb = sub["orig_len"].sum()
        fd = sub["rel_time"].max() - sub["rel_time"].min()
        fd = fd if fd > 0 else 1e-9
        print(f"    {flow}")
        print(f"      Packets : {len(sub):,}")
        print(f"      Bytes   : {fb:,.0f}  ({fb / 1e6:.2f} MB)")
        print(f"      Duration: {fd:.6f} s")
        print(f"      Avg rate: {fb * 8 / fd / 1e6:.2f} Mbps")

        # Direction breakdown
        for direction in sorted(sub["direction"].unique()):
            dsub = sub[sub["direction"] == direction]
            db = dsub["orig_len"].sum()
            print(f"        {direction}: {len(dsub):,} pkts, {db:,.0f} bytes")

    print()

    # Opcode summary
    print("  Opcode summary:")
    print("  " + "-" * 66)
    opcode_counts = df["opcode_name"].value_counts()
    for op, cnt in opcode_counts.items():
        print(f"    {op:40s} : {cnt:>10,}")

    # QP summary
    print()
    print(f"  Unique Queue Pairs (dest_qp): {df['dest_qp'].nunique()}")

    # Inter-packet gap stats
    ipg = df["rel_time"].diff().dropna().values * 1e6
    print()
    print("  Inter-packet gap (µs):")
    print(f"    Mean   : {np.mean(ipg):.3f}")
    print(f"    Median : {np.median(ipg):.3f}")
    print(f"    Std    : {np.std(ipg):.3f}")
    print(f"    Min    : {np.min(ipg):.3f}")
    print(f"    Max    : {np.max(ipg):.3f}")

    # Congestion and jumbo reports
    print_congestion_assessment(df)
    print_jumbo_frame_report(df)

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    if not os.path.isfile(PCAP_FILE):
        print(f"[!] File not found: {PCAP_FILE}")
        print("    Place the pcap in the current directory or update PCAP_FILE.")
        sys.exit(1)

    print(f"[*] Loading {PCAP_FILE} ...")
    df = load_and_parse(PCAP_FILE)

    # Annotate flows
    df["flow"] = df.apply(identify_flow, axis=1)
    df["direction"] = df.apply(identify_direction, axis=1)

    # Print textual summary
    print_summary(df)

    # Generate all plots
    print("\n[*] Generating plots ...")
    plot_packet_timeline(df)
    plot_bitrate(df)
    plot_throughput(df)
    plot_inter_packet_latency(df)
    plot_ack_latency(df)
    plot_jitter(df)
    plot_opcode_distribution(df)
    plot_packet_size_distribution(df)
    plot_qp_activity(df)
    plot_psn_progression(df)

    print("\n[✓] Done. All plots saved as PNG in the current directory.")
    print(
        "    Tip: adjust BIN_WIDTH at the top of the script for finer/coarser time bins."
    )


if __name__ == "__main__":
    main()
