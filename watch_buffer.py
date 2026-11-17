"""
watch_buffer.py - Live Virtual Cart, Inflow & Memory Inspector for Each Lane.

Options:
  --memory, -m    Display detailed in-memory virtual tokens (vehicle ID, type, age, remaining TTL)
                  ensuring tokens do not exceed per-lane TTL.
"""

import argparse
import json
import os
import sys
import time

LOG_FILE = "live_counts.jsonl"


def clear_screen():
    print("\033[H\033[J", end="")


def main():
    parser = argparse.ArgumentParser(description="Live Lane Buffer, Inflow, and Memory TTL Inspector.")
    parser.add_argument(
        "--memory", "-m",
        action="store_true",
        help="Show detailed token memory (ID, type, age, and remaining TTL) for each lane's virtual cart",
    )
    args = parser.parse_args()

    if not os.path.exists(LOG_FILE):
        print(f"Waiting for {LOG_FILE} to be created by run_multi_camera.py...")
        while not os.path.exists(LOG_FILE):
            time.sleep(0.5)

    print("Connecting to live buffer telemetry...\n")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.15)
                continue
            try:
                data = json.loads(line)
                flow = data.get("payload", {}).get("traffic_flow", {})
                by_lane = flow.get("by_lane", {})
                alarms = flow.get("alarm_flags", {})
                ts = data.get("payload", {}).get("timestamp", "")

                if not by_lane:
                    continue

                clear_screen()
                print("=" * 95)
                print(f"   LIVE LANE BUFFER, INFLOW & CART MEMORY MONITOR (Timestamp: {ts})")
                print("=" * 95)
                print(f"{'LANE':<6} | {'FINAL':<6} = {'CAM1 VIS':<9} + {'CART BUFFER':<12} | {'INFLOW':<8} | {'IN-RATE':<10} | {'TTL STATE':<10} | {'HANDOVERS'}")
                print("-" * 95)

                for lid in sorted(by_lane.keys()):
                    l = by_lane[lid]
                    final_demand = l.get("demand_count", 0)
                    vis_cam1 = l.get("vis_cam1", 0)
                    cart_tokens = l.get("cart_tokens", 0)
                    inflow = l.get("inflow", 0)
                    inflow_rate = l.get("inflow_rate", 0.0)
                    is_frozen = l.get("is_frozen", False)
                    spillback = l.get("spillback", False)
                    exact_ho = l.get("handover_exact", 0)
                    borrowed_ho = l.get("handover_borrowed", 0)

                    state_str = "SPILLBACK" if spillback else ("FROZEN ❄" if is_frozen else "TICKING ⏱")

                    print(f"{lid:<6} | {final_demand:<6} = {vis_cam1:<9} + {cart_tokens:<12} | {inflow:<8} | {inflow_rate:<10.2f} | {state_str:<10} | {exact_ho} exact, {borrowed_ho} borrowed")

                if args.memory:
                    print("\n" + "=" * 95)
                    print("   [--memory] IN-MEMORY CART TOKENS & TTL TRACKING PER LANE (Zone B: 100m -> 55m)")
                    print("=" * 95)
                    has_tokens = False
                    for lid in sorted(by_lane.keys()):
                        l = by_lane[lid]
                        tokens = l.get("tokens_detail", [])
                        if tokens:
                            has_tokens = True
                            print(f"\n -> Lane [{lid}] ({len(tokens)} vehicle(s) in memory cart):")
                            for t in tokens:
                                bar_len = int((t['ttl_remaining_sec'] / max(1e-3, t['ttl_total_sec'])) * 20)
                                bar = "█" * bar_len + "░" * (20 - bar_len)
                                print(f"    Token #{t['id']:<4} [{t['v_type']:<7}] | Age: {t['age_sec']:>4.1f}s | TTL Rem: {t['ttl_remaining_sec']:>4.1f}s / {t['ttl_total_sec']:>4.1f}s [{bar}]")
                    if not has_tokens:
                        print(" (No tokens currently in flight between 100m and 55m. Inflow adds tokens when cars cross Cam 2)")

                print("-" * 95)
                print(f"Spillback Active: {alarms.get('emergency_spillback_override', False)} | Pass '--memory' to inspect individual token TTL bars")
                print("=" * 95)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting monitor.")
