"""Extract the purchase events and user information needed from a local Tianchi ZIP."""

import argparse
import csv
import gzip
import hashlib
import io
import shutil
from pathlib import Path
from zipfile import ZipFile

PURCHASE_LOG = "data_format1/user_log_format1.csv"
USER_INFO = "data_format1/user_info_format1.csv"
PAPER_ROWS = 3_292_144
PAPER_MD5 = "0dffb6ca97c6aa8ff340afe5951a0447"


def export_purchases(archive, path):
    digest = hashlib.md5(usedforsecurity=False)
    count = 0
    with archive.open(PURCHASE_LOG) as raw, path.open("xb") as output:
        rows = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        header = next(rows, [])
        if header != ["user_id", "item_id", "cat_id", "seller_id", "brand_id", "time_stamp", "action_type"]:
            raise ValueError("Unexpected Tianchi event-log column names or order")
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            for user, item, _, _, _, stamp, action in rows:
                if action != "2":
                    continue
                line = f"{user},{item},{stamp}\n".encode("utf-8")
                digest.update(line)
                compressed.write(line)
                count += 1
    return count, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path, help="New directory for the two extracted files")
    parser.add_argument("--verify-paper-input", action="store_true", help="Check the historical purchase file signature")
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        for member in (PURCHASE_LOG, USER_INFO):
            archive.getinfo(member)
        args.output_dir.mkdir(parents=True)
        purchase_path = args.output_dir / "tmall_buy.csv.gz"
        count, signature = export_purchases(archive, purchase_path)
        if args.verify_paper_input and (count != PAPER_ROWS or signature != PAPER_MD5):
            parser.exit(1, f"Paper input mismatch. Found {count:,} purchases and MD5 {signature}.\n")
        user_path = args.output_dir / "tmall_user_info.csv.gz"
        with archive.open(USER_INFO) as source, user_path.open("xb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
    print(f"{count:,} purchases. MD5 of the extracted purchase CSV: {signature}.")


if __name__ == "__main__":
    main()
