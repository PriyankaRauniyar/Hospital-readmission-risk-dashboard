"""Generate a source-free public export with the frozen ensemble, offline."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from predictor import EXCLUDED_DISCHARGE_CODES, load_bundle, predict
from public_predictions import BATCH_SIZE, PUBLIC_COLUMNS, load_public_predictions

ROOT = Path(__file__).resolve().parent


def generate(source: Path, models: Path, output: Path, chunk_size: int = BATCH_SIZE):
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive.")
    if source.resolve() == output.resolve():
        raise ValueError("Output cannot overwrite the source dataset.")
    bundle = load_bundle(models)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    total = excluded_count = source_count = 0
    try:
        for raw in pd.read_csv(source, chunksize=chunk_size, low_memory=False):
            source_count += len(raw)
            codes = pd.to_numeric(raw['discharge_disposition_id'], errors='coerce')
            excluded = codes.isin([int(code) for code in EXCLUDED_DISCHARGE_CODES])
            excluded_count += int(excluded.sum())
            eligible = raw.loc[~excluded]
            if eligible.empty:
                continue
            result = predict(eligible, bundle)
            result.insert(0, 'Display_row', np.arange(total + 1, total + len(result) + 1))
            # Explicit allowlist prevents source columns from entering the public file.
            result[PUBLIC_COLUMNS].to_csv(temporary, index=False, mode='w' if total == 0 else 'a',
                                         header=total == 0, float_format='%.12g',
                                         compression='gzip' if output.name.endswith('.gz') else None)
            total += len(result)
            print(f'Scored {total:,} eligible encounters', flush=True)
        reviewed = load_public_predictions(temporary, compression='gzip' if output.name.endswith('.gz') else None)
        assert len(reviewed) == total
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Source: {source_count:,}; excluded: {excluded_count:,}; eligible: {total:,}')
    print(f'Public export: {output} ({output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/diabetic_data.csv')
    parser.add_argument('--models', type=Path, default=ROOT / 'final_model_artifacts')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/public_predictions.csv.gz')
    parser.add_argument('--chunk-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    generate(args.source, args.models, args.output, args.chunk_size)
