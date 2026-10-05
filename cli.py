#!/usr/bin/env python3
"""
Enterprise Biometric System Control CLI
Management utility for dataset auditing, KNN calibration, and attendance reporting.
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dataset_manager import DatasetManager
from core.knn_engine import KNNEngine
from core.attendance_logger import AttendanceLogger


def main():
    parser = argparse.ArgumentParser(description="Enterprise Face Recognition System CLI Controller")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: list
    subparsers.add_parser("list", help="List all enrolled biometric subjects")

    # Subcommand: seed
    subparsers.add_parser("seed", help="Seed initial demo biometric profiles")

    # Subcommand: diagnose
    subparsers.add_parser("diagnose", help="Run KNN diagnostic audit on current dataset")

    # Subcommand: attendance
    att_parser = subparsers.add_parser("attendance", help="View or export attendance logs")
    att_parser.add_argument("--limit", type=int, default=20, help="Number of records to show")
    att_parser.add_argument("--export", type=str, default=None, help="File path to save CSV export")

    args = parser.parse_args()

    dm = DatasetManager()
    al = AttendanceLogger()

    if args.command == "list":
        subjects = dm.get_all_subjects()
        print(f"\n{'ID'.ljust(4)} {'NAME'.ljust(25)} {'EMP ID'.ljust(12)} {'DEPARTMENT'.ljust(22)} {'SAMPLES'.ljust(8)}")
        print("-" * 75)
        for s in subjects:
            print(f"{str(s['id']).ljust(4)} {s['full_name'].ljust(25)} {str(s['employee_id']).ljust(12)} {str(s['department']).ljust(22)} {str(s['sample_count']).ljust(8)}")
        print("-" * 75)
        print(f"Total Enrolled Subjects: {len(subjects)}\n")

    elif args.command == "seed":
        dm.seed_initial_demo_profiles()
        print("[SUCCESS] Pre-seeded demo biometric profiles.")

    elif args.command == "diagnose":
        X, y, names = dm.assemble_training_matrix()
        if len(X) == 0:
            print("[WARN] Dataset is empty.")
            return
        knn = KNNEngine(k=5, metric="euclidean", weights="distance")
        knn.fit(X, y, names)
        diag = knn.get_diagnostics()
        print("\n=== KNN BIOMETRIC ENGINE DIAGNOSTICS ===")
        print(json.dumps(diag, indent=2))
        print("=======================================\n")

    elif args.command == "attendance":
        if args.export:
            csv_data = al.export_csv()
            with open(args.export, "w", encoding="utf-8") as f:
                f.write(csv_data)
            print(f"[SUCCESS] Attendance log exported to {args.export}")
        else:
            logs = al.get_logs(limit=args.limit)
            summary = al.get_summary_metrics()
            print(f"\n=== ATTENDANCE SUMMARY TODAY ===")
            print(f"Total Check-ins: {summary['total_today']} | Verified: {summary['verified_today']} | Unknown Alerts: {summary['unknown_today']} | Avg Conf: {summary['avg_confidence']}%")
            print("-" * 80)
            print(f"{'TIMESTAMP'.ljust(20)} {'SUBJECT'.ljust(22)} {'EMP ID'.ljust(12)} {'CONF'.ljust(8)} {'STATUS'}")
            print("-" * 80)
            for l in logs:
                print(f"{l['timestamp'].ljust(20)} {l['person_name'].ljust(22)} {str(l['employee_id']).ljust(12)} {str(l['confidence_pct']).ljust(8)}% {l['status']}")
            print("-" * 80 + "\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
