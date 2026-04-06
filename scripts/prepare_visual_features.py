"""
Convert pre-extracted visual region features to HDF5 format for VQADataset.

Expected output format:
    data/vqa/visual_features/{split}_features.h5
    Key:   image_id (string)
    Value: float32 array of shape [num_boxes, feature_dim]  (typically [36, 2048])

This script does NOT extract features from images. It only converts
already-extracted bottom-up attention features into the HDF5 format
expected by VQADataset.

DEVIATION FROM PAPER:
    The paper uses a specific object detector and feature extractor whose
    exact weights and pipeline are not publicly released. This script
    assumes you have already obtained bottom-up attention features from
    one of the following common sources:
      1. Anderson et al. CVPR 2018 official release (TSV format):
         https://github.com/peteanderson80/bottom-up-attention
      2. Pre-extracted .npy or .npz files (per image)
      3. An existing HDF5 file (re-key to match expected format)

    The expected feature dimension is 2048 (ResNet-based bottom-up features).
    If your features have a different dimension, update src/configs/model/vqa_gnn.yaml
    (d_visual) and src/configs/datasets/vqa.yaml (d_visual) accordingly.

Supported input formats:
    --format tsv    TSV file from the bottom-up-attention official release.
                    Each line: image_id <tab> num_boxes <tab> features_b64 [<tab> boxes_b64]
                    Features are base64-encoded float32 arrays.

    --format npy    Directory of per-image .npy files.
                    Files must be named {image_id}.npy (e.g. 12345.npy).
                    Each file should be float32 of shape [num_boxes, feature_dim].

    --format npz    Directory of per-image .npz files.
                    Files must be named {image_id}.npz.
                    Features are loaded from the 'features' key (configurable via --npz-key).

    --format h5     An existing HDF5 file. Re-keys entries to string image_ids.
                    Useful if you have features in a different HDF5 layout.

Usage:
    # From TSV (bottom-up attention official release):
    python scripts/prepare_visual_features.py \\
        --format tsv \\
        --input data/raw/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv \\
        --output data/vqa/visual_features/train_features.h5

    # From a directory of .npy files:
    python scripts/prepare_visual_features.py \\
        --format npy \\
        --input data/raw/train_npy/ \\
        --output data/vqa/visual_features/train_features.h5

    # From per-image .npz files:
    python scripts/prepare_visual_features.py \\
        --format npz \\
        --input data/raw/train_npz/ \\
        --output data/vqa/visual_features/train_features.h5 \\
        --npz-key fc6  # key inside each .npz file

    # Verify an existing HDF5 (no conversion, just check):
    python scripts/prepare_visual_features.py \\
        --format h5 \\
        --input data/vqa/visual_features/train_features.h5 \\
        --output data/vqa/visual_features/train_features.h5 \\
        --verify-only
"""

import argparse
import base64
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print("[ERROR] h5py is required. Install: pip install h5py", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------


def _read_tsv(tsv_path: Path, num_fixed_boxes: int):
    """
    Yield (image_id_str, features_np) from a bottom-up attention TSV file.

    TSV columns (from Anderson et al. bottom-up-attention repo):
        image_id, num_boxes, features (base64), [boxes (base64)]

    Features are base64-encoded float32 arrays of shape [num_boxes, 2048].

    If the file has more or fewer than num_fixed_boxes per image, features are
    truncated or zero-padded to num_fixed_boxes.
    """
    path = Path(tsv_path)
    if not path.exists():
        print(f"[ERROR] TSV file not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Count lines for tqdm
    total = None
    try:
        with open(path, "r") as f:
            total = sum(1 for _ in f)
    except Exception:
        pass

    with open(path, "r") as f:
        for line in tqdm(f, total=total, desc="Reading TSV"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            image_id = str(int(parts[0]))
            # num_boxes = int(parts[1])  # actual num boxes in file

            # Decode base64 features
            try:
                feat_bytes = base64.b64decode(parts[2])
                feats = np.frombuffer(feat_bytes, dtype=np.float32)
                feature_dim = feats.size // int(parts[1])
                feats = feats.reshape(int(parts[1]), feature_dim)
            except Exception as e:
                print(
                    f"[WARNING] Could not decode features for image_id={image_id}: {e}",
                    file=sys.stderr,
                )
                continue

            # Pad or truncate to num_fixed_boxes
            actual = feats.shape[0]
            if actual > num_fixed_boxes:
                feats = feats[:num_fixed_boxes]
            elif actual < num_fixed_boxes:
                pad = np.zeros((num_fixed_boxes - actual, feats.shape[1]), dtype=np.float32)
                feats = np.concatenate([feats, pad], axis=0)

            yield image_id, feats


def _read_npy_dir(npy_dir: Path, num_fixed_boxes: int, feature_dim: int):
    """
    Yield (image_id_str, features_np) from a directory of .npy files.

    File naming: {image_id}.npy (e.g. 12345.npy)
    Expected shape: [num_boxes, feature_dim]
    """
    npy_dir = Path(npy_dir)
    if not npy_dir.is_dir():
        print(f"[ERROR] npy directory not found: {npy_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(npy_dir.glob("*.npy"))
    if not files:
        print(f"[ERROR] No .npy files found in: {npy_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} .npy files.")

    for npy_file in tqdm(files, desc="Reading .npy files"):
        image_id = npy_file.stem  # filename without extension
        try:
            feats = np.load(npy_file).astype(np.float32)
        except Exception as e:
            print(
                f"[WARNING] Could not load {npy_file}: {e}",
                file=sys.stderr,
            )
            continue

        if feats.ndim == 1:
            # Single box stored as flat vector
            feats = feats.reshape(1, -1)

        if feats.ndim != 2:
            print(
                f"[WARNING] Unexpected shape {feats.shape} for {npy_file}, skipping.",
                file=sys.stderr,
            )
            continue

        actual = feats.shape[0]
        actual_dim = feats.shape[1]

        if actual_dim != feature_dim:
            print(
                f"[WARNING] Expected feature_dim={feature_dim} but got {actual_dim} "
                f"for {npy_file}. Using as-is; update --feature-dim if needed.",
                file=sys.stderr,
            )

        if actual > num_fixed_boxes:
            feats = feats[:num_fixed_boxes]
        elif actual < num_fixed_boxes:
            pad = np.zeros((num_fixed_boxes - actual, feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)

        yield image_id, feats


def _read_npz_dir(npz_dir: Path, num_fixed_boxes: int, feature_dim: int, npz_key: str):
    """
    Yield (image_id_str, features_np) from a directory of .npz files.

    File naming: {image_id}.npz
    Features extracted from key `npz_key` (default: 'features').
    """
    npz_dir = Path(npz_dir)
    if not npz_dir.is_dir():
        print(f"[ERROR] npz directory not found: {npz_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(npz_dir.glob("*.npz"))
    if not files:
        print(f"[ERROR] No .npz files found in: {npz_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} .npz files.")

    for npz_file in tqdm(files, desc="Reading .npz files"):
        image_id = npz_file.stem
        try:
            data = np.load(npz_file)
            if npz_key not in data:
                available = list(data.keys())
                print(
                    f"[WARNING] Key '{npz_key}' not found in {npz_file}. "
                    f"Available keys: {available}. Skipping.",
                    file=sys.stderr,
                )
                continue
            feats = data[npz_key].astype(np.float32)
        except Exception as e:
            print(f"[WARNING] Could not load {npz_file}: {e}", file=sys.stderr)
            continue

        if feats.ndim == 1:
            feats = feats.reshape(1, -1)

        actual = feats.shape[0]
        if actual > num_fixed_boxes:
            feats = feats[:num_fixed_boxes]
        elif actual < num_fixed_boxes:
            pad = np.zeros((num_fixed_boxes - actual, feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)

        yield image_id, feats


def _verify_h5(h5_path: Path, num_fixed_boxes: int, feature_dim: int, sample_size: int = 10):
    """
    Verify an existing HDF5 file has the expected structure.

    Checks:
    - File exists and opens
    - Sample of keys can be read
    - Feature shape matches expected [num_fixed_boxes, feature_dim]
    - dtype is float32
    """
    if not h5_path.exists():
        print(f"[ERROR] HDF5 file not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying HDF5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        print(f"  Number of entries: {len(keys)}")

        sample_keys = keys[:sample_size]
        errors = 0
        for k in sample_keys:
            arr = f[k][()]
            if arr.shape != (num_fixed_boxes, feature_dim):
                print(
                    f"  [WARNING] Key '{k}': shape={arr.shape}, "
                    f"expected ({num_fixed_boxes}, {feature_dim})",
                    file=sys.stderr,
                )
                errors += 1
            if arr.dtype != np.float32:
                print(
                    f"  [WARNING] Key '{k}': dtype={arr.dtype}, expected float32",
                    file=sys.stderr,
                )
                errors += 1

        if errors == 0:
            print(
                f"  OK: sample of {len(sample_keys)} entries verified "
                f"(shape={num_fixed_boxes}x{feature_dim}, dtype=float32)."
            )
        else:
            print(f"  {errors} warning(s) in sample of {len(sample_keys)} entries.")

    return True


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_to_h5(generator, output_path: Path, compression: str = "gzip"):
    """
    Write (image_id, features) pairs from generator to HDF5.

    Args:
        generator: iterable of (image_id_str, np.ndarray[float32]) pairs.
        output_path (Path): output .h5 file path.
        compression (str): HDF5 compression codec ('gzip', 'lzf', or None).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with h5py.File(output_path, "w") as f:
        for image_id, feats in generator:
            f.create_dataset(
                str(image_id),
                data=feats.astype(np.float32),
                compression=compression,
            )
            n_written += 1

    print(f"Written {n_written} entries to {output_path}")
    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert pre-extracted visual features to HDF5 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format",
        choices=["tsv", "npy", "npz", "h5"],
        required=True,
        help="Input format: tsv | npy | npz | h5",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Input path: TSV file, or directory of .npy/.npz files, or existing .h5.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output HDF5 path (e.g. data/vqa/visual_features/train_features.h5).",
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        default=36,
        help="Fixed number of region boxes per image (default: 36). "
             "Must match num_visual_nodes in dataset and model configs.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=2048,
        help="Feature dimension per box (default: 2048). "
             "Must match d_visual in dataset and model configs.",
    )
    parser.add_argument(
        "--npz-key",
        default="features",
        help="Key to read from .npz files (default: 'features'). "
             "Common alternatives: 'fc6', 'x', 'feat'.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression codec (default: gzip). Use 'none' to disable.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the --input HDF5 file structure, do not write output.",
    )

    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.verify_only:
        if args.format != "h5":
            print("[ERROR] --verify-only only works with --format h5.", file=sys.stderr)
            sys.exit(1)
        _verify_h5(input_path, args.num_boxes, args.feature_dim)
        return

    if args.format == "tsv":
        gen = _read_tsv(input_path, args.num_boxes)

    elif args.format == "npy":
        gen = _read_npy_dir(input_path, args.num_boxes, args.feature_dim)

    elif args.format == "npz":
        gen = _read_npz_dir(input_path, args.num_boxes, args.feature_dim, args.npz_key)

    elif args.format == "h5":
        # Re-key from existing HDF5 (e.g. integer keys → string keys)
        def _read_existing_h5(path, num_boxes, feature_dim):
            path = Path(path)
            if not path.exists():
                print(f"[ERROR] HDF5 not found: {path}", file=sys.stderr)
                sys.exit(1)
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
            print(f"Re-keying {len(keys)} entries from existing HDF5.")
            with h5py.File(path, "r") as f:
                for k in tqdm(keys, desc="Re-keying"):
                    feats = f[k][()].astype(np.float32)
                    if feats.ndim == 1:
                        feats = feats.reshape(1, -1)
                    actual = feats.shape[0]
                    if actual > num_boxes:
                        feats = feats[:num_boxes]
                    elif actual < num_boxes:
                        pad = np.zeros(
                            (num_boxes - actual, feats.shape[1]), dtype=np.float32
                        )
                        feats = np.concatenate([feats, pad], axis=0)
                    yield str(k), feats

        if input_path.resolve() == output_path.resolve():
            print(
                "[WARNING] Input and output are the same file. "
                "Loading all entries into memory first.",
                file=sys.stderr,
            )
            entries = list(_read_existing_h5(input_path, args.num_boxes, args.feature_dim))
            gen = iter(entries)
        else:
            gen = _read_existing_h5(input_path, args.num_boxes, args.feature_dim)
    else:
        print(f"[ERROR] Unknown format: {args.format}", file=sys.stderr)
        sys.exit(1)

    n = write_to_h5(gen, output_path, compression=compression)

    if n == 0:
        print("[WARNING] No entries were written. Check your input path.", file=sys.stderr)
        sys.exit(1)

    # Quick verification
    print(f"\nVerifying output...")
    _verify_h5(output_path, args.num_boxes, args.feature_dim, sample_size=min(n, 20))
    print("Done.")


if __name__ == "__main__":
    main()
