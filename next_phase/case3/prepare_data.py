"""
Step 0 -- fetch the real bandit logs (Open Bandit Dataset, ZOZOTOWN) and keep
only the columns this study needs.

The full Open Bandit Dataset zip is 413 MB compressed but 11.7 GB uncompressed
(the per-row user-item affinity columns make the CSVs huge), which does not fit
on this machine. We therefore never download or extract the whole archive:

  * the zip's central directory is read with HTTP range requests, and only the
    members we need are streamed and decompressed on the fly
    (random/{all,men,women}.csv = 41 MB compressed in total; the optional
    Bernoulli-Thompson-sampling log bts/all/all.csv is another 204 MB);
  * each CSV is parsed line by line and only timestamp, item_id, position,
    click and propensity_score are kept (no user features / affinities);
  * the result is written as a small .npz per log under data/.

If a full local copy of the zip is present at data/open_bandit_dataset.zip it
is used instead of the network.

Outputs (data/ is git-ignored):
  data/obd_random_<campaign>.npz   ts_us, item, pos, click, pscore  (per row)
  data/obd_bts_all_hourly.npz      per-hour x per-item impressions and clicks of
                                   the deployed Thompson-sampling policy (--bts)
  data/obd_meta.json               row counts, item counts, timing, source

Usage (from the repository root):
  python3 next_phase/case3/prepare_data.py                 # random logs, 3 campaigns
  python3 next_phase/case3/prepare_data.py --bts           # + the deployed BTS log

Data license: CC BY 4.0 (stated in the dataset's README). Cite Saito et al.,
"Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible
Off-Policy Evaluation", NeurIPS 2021 Datasets and Benchmarks (arXiv:2008.07146).
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
URL = "https://research.zozo.com/data_release/open_bandit_dataset.zip"
LOCAL_ZIP = DATA / "open_bandit_dataset.zip"


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file object backed by HTTP range requests, so that
    zipfile can read the central directory and individual members of a remote
    archive without downloading the whole thing. Blocks are cached (LRU-ish)."""

    def __init__(self, url: str, block: int = 4 << 20, max_cached: int = 4, retries: int = 5):
        super().__init__()
        self.url, self.block, self.max_cached, self.retries = url, block, max_cached, retries
        self.pos, self.cache, self.n_requests, self.n_bytes = 0, {}, 0, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _block(self, b: int) -> bytes:
        if b in self.cache:
            return self.cache[b]
        start = b * self.block
        end = min(self.size, start + self.block) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError("short read")
                break
            except Exception:  # transient network error: back off and retry
                if attempt == self.retries - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))
        self.n_requests += 1
        self.n_bytes += len(data)
        self.cache[b] = data
        while len(self.cache) > self.max_cached:
            self.cache.pop(next(iter(self.cache)))
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b = self.pos // self.block
            off = self.pos - b * self.block
            chunk = self._block(b)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def open_archive():
    """Local zip if present, otherwise the remote zip via range requests."""
    if LOCAL_ZIP.exists():
        return zipfile.ZipFile(LOCAL_ZIP), None
    f = HTTPRangeFile(URL)
    return zipfile.ZipFile(f), f


def _ts_to_us(s: str) -> int:
    # e.g. "2019-11-24 00:00:00.007365+00:00" -> microseconds since the epoch
    d = datetime.fromisoformat(s)
    return int(d.timestamp()) * 1_000_000 + d.microsecond


def parse_member(z: zipfile.ZipFile, name: str, keep_rows: bool = True, report_every: int = 2_000_000):
    """Stream one CSV member and keep timestamp, item_id, position, click and
    propensity_score. Columns (header): ,timestamp,item_id,position,click,
    propensity_score,user_feature_0..3,user-item_affinity_0..; the leading
    unnamed column is the row index. Only the first 6 fields are split off."""
    ts, item, pos, click, ps = [], [], [], [], []
    t0 = time.time()
    with z.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        header = text.readline().rstrip("\r\n").split(",", 6)[:6]
        assert header[1:6] == ["timestamp", "item_id", "position", "click", "propensity_score"], header
        for k, line in enumerate(text):
            f = line.split(",", 6)
            ts.append(f[1])
            item.append(int(f[2]))
            pos.append(int(f[3]))
            click.append(int(f[4]))
            ps.append(float(f[5]))
            if report_every and (k + 1) % report_every == 0:
                print(f"    {name}: {k + 1:,} rows, {time.time() - t0:.0f}s", flush=True)
    out = {
        "item": np.asarray(item, dtype=np.int16),
        "pos": np.asarray(pos, dtype=np.int8),
        "click": np.asarray(click, dtype=np.int8),
        "pscore": np.asarray(ps, dtype=np.float32),
    }
    # Timestamps: parse once per distinct second prefix would be faster, but
    # fromisoformat is ~1 microsecond per call, fine for <= 12M rows.
    out["ts_us"] = np.fromiter((_ts_to_us(s) for s in ts), dtype=np.int64, count=len(ts))
    return out


def prepare_random(z, campaigns, meta):
    for c in campaigns:
        dst = DATA / f"obd_random_{c}.npz"
        if dst.exists():
            print(f"  {dst.name} exists, skipping")
            continue
        name = f"open_bandit_dataset/random/{c}/{c}.csv"
        info = z.getinfo(name)
        print(f"  streaming {name} ({info.compress_size / 1e6:.1f} MB compressed, "
              f"{info.file_size / 1e6:.0f} MB uncompressed; only 5 columns kept)", flush=True)
        t0 = time.time()
        d = parse_member(z, name)
        np.savez_compressed(dst, **d)
        meta[f"random_{c}"] = {
            "member": name, "rows": int(len(d["item"])), "items": int(d["item"].max() + 1),
            "clicks": int(d["click"].sum()), "ctr": float(d["click"].mean()),
            "first_ts": datetime.fromtimestamp(d["ts_us"].min() / 1e6, timezone.utc).isoformat(),
            "last_ts": datetime.fromtimestamp(d["ts_us"].max() / 1e6, timezone.utc).isoformat(),
            "pscore_unique": sorted({float(x) for x in np.unique(d["pscore"])})[:5],
            "seconds": round(time.time() - t0, 1),
            "npz_MB": round(dst.stat().st_size / 1e6, 2),
        }
        print(f"    -> {dst.name}: {meta[f'random_{c}']}", flush=True)


def prepare_bts(z, meta, campaign="all"):
    """The deployed Bernoulli Thompson Sampling policy's log, aggregated to
    per-hour x per-item impressions and clicks (the row-level file would be
    ~200 MB even with 5 columns; we only need the allocation over time)."""
    dst = DATA / f"obd_bts_{campaign}_hourly.npz"
    if dst.exists():
        print(f"  {dst.name} exists, skipping")
        return
    name = f"open_bandit_dataset/bts/{campaign}/{campaign}.csv"
    info = z.getinfo(name)
    print(f"  streaming {name} ({info.compress_size / 1e6:.1f} MB compressed, "
          f"{info.file_size / 1e6:.0f} MB uncompressed) -- aggregating per hour", flush=True)
    t0 = time.time()
    K = 0
    counts, clicks, pssum = {}, {}, {}
    t_first = None
    with z.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        text.readline()
        hour_cache = {}
        for k, line in enumerate(text):
            f = line.split(",", 6)
            key = f[1][:13]                      # "YYYY-MM-DD HH" -> hour bucket
            h = hour_cache.get(key)
            if h is None:
                h = hour_cache[key] = len(hour_cache)
            i = int(f[2])
            K = max(K, i + 1)
            kk = (h, i)
            counts[kk] = counts.get(kk, 0) + 1
            if f[4] == "1":
                clicks[kk] = clicks.get(kk, 0) + 1
            pssum[kk] = pssum.get(kk, 0.0) + float(f[5])
            if (k + 1) % 2_000_000 == 0:
                print(f"    {k + 1:,} rows, {time.time() - t0:.0f}s", flush=True)
    H = len(hour_cache)
    n = np.zeros((H, K), dtype=np.int64)
    c = np.zeros((H, K), dtype=np.int64)
    s = np.zeros((H, K), dtype=np.float64)
    for (h, i), v in counts.items():
        n[h, i] = v
        s[h, i] = pssum[(h, i)]
    for (h, i), v in clicks.items():
        c[h, i] = v
    hours = np.array(sorted(hour_cache, key=hour_cache.get))
    np.savez_compressed(dst, n=n, clicks=c, pscore_sum=s, hours=hours)
    meta[f"bts_{campaign}"] = {
        "member": name, "rows": int(n.sum()), "items": int(K), "hours": int(H),
        "clicks": int(c.sum()), "ctr": float(c.sum() / n.sum()),
        "first_hour": str(hours[0]), "last_hour": str(hours[-1]),
        "seconds": round(time.time() - t0, 1), "npz_MB": round(dst.stat().st_size / 1e6, 2),
    }
    print(f"    -> {dst.name}: {meta[f'bts_{campaign}']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaigns", nargs="+", default=["all", "men", "women"])
    ap.add_argument("--bts", action="store_true", help="also aggregate the deployed BTS log (bts/all)")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    meta_path = DATA / "obd_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    t0 = time.time()
    z, remote = open_archive()
    print(f"archive: {'remote ' + URL if remote else LOCAL_ZIP}")
    with z:
        for nm in ("open_bandit_dataset/README", "open_bandit_dataset/VERSION"):
            (DATA / ("obd_" + nm.split("/")[-1] + ".txt")).write_bytes(z.read(nm))
        for c in args.campaigns:
            nm = f"open_bandit_dataset/random/{c}/item_context.csv"
            (DATA / f"obd_item_context_{c}.csv").write_bytes(z.read(nm))
        prepare_random(z, args.campaigns, meta)
        if args.bts:
            prepare_bts(z, meta)
    if remote is not None:
        meta["download"] = {"MB_transferred_this_run": round(remote.n_bytes / 1e6, 1),
                            "range_requests": remote.n_requests}
    meta["source"] = URL
    meta["license"] = "CC BY 4.0"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s; meta -> {meta_path}")


if __name__ == "__main__":
    main()
