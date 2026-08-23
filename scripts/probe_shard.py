"""Probe one parquet shard: row-group layout + audio encoding.

Determines whether we can fetch individual clips via targeted row-group reads
(parquet range requests) instead of streaming entire shards.
"""

import io

import fsspec
import pyarrow.parquet as pq
import soundfile as sf

DATASET_ID = "ghanaopendata/ghana-english-tts-filtered"
SHARD = "data/filtered-train-00000-of-00082.parquet"


def main() -> None:
    fs = fsspec.filesystem("hf")
    url = f"hf://datasets/{DATASET_ID}/{SHARD}"

    with fs.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        print(f"rows: {md.num_rows}")
        print(f"row groups: {md.num_row_groups}")
        print(f"columns: {md.num_columns}")
        print(f"schema: {pf.schema_arrow}")
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            size_mb = sum(rg.column(c).total_compressed_size for c in range(rg.num_columns)) / 1e6
            print(f"  row group {i}: {rg.num_rows} rows, {size_mb:.1f} MB compressed")

        # Audio probe: read row 0 only (first row group, audio column).
        table = pf.read_row_group(0, columns=["audio"])
        row = table.slice(0, 1).to_pylist()[0]
        audio = row["audio"]
        raw = audio["bytes"]
        print(f"\naudio bytes length: {len(raw)}")
        print(f"audio path field: {audio['path']!r}")
        print(f"magic header: {raw[:12]!r}")

        info = sf.info(io.BytesIO(raw))
        print(f"decoded: samplerate={info.samplerate}, channels={info.channels}, "
              f"frames={info.frames}, format={info.format}/{info.subtype}, "
              f"duration={info.duration:.2f}s")


if __name__ == "__main__":
    main()
