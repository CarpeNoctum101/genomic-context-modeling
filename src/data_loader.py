import pandas as pd
import random
from collections import defaultdict
from pyfaidx import Fasta

def load_genome(fasta_path):
    return Fasta(fasta_path)

def load_peaks(bed_path):
    cols  = ['chrom', 'start', 'end', 'name', 'score', 'strand', 
             'signalValue', 'pValue', 'qValue', 'peak']
    
    df = pd.read_csv(bed_path, sep='\t', header=None)

    if df.shape[1] >= 10:
        df.columns = cols + [f'extra_{i}' for i in range(10, df.shape[1])]
    else:
        df.columns = cols[:df.shape[1]]
    
    return df

def extract_sequences(genome, chrom, start, end):
    try:
        return genome[chrom][start:end].seq.upper()
    except KeyError:
        print(f"Warning: {chrom}:{start}-{end} not found in genome.")
        return None

def build_peak_lookup(peaks_df):
    peak_lookup = defaultdict(list)
    for _, row in peaks_df.iterrows():
        peak_lookup[row["chrom"]].append((row["start"], row["end"]))

    return peak_lookup

def overlaps_peak(chrom, start, end, peak_lookup):
    if chrom not in peak_lookup:
        return False
    
    for peak_start, peak_end in peak_lookup[chrom]:
        if start < peak_end and end > peak_start:
            return True
    
    return False

def sample_negative_region(genome, length, peak_lookup, chrom=None, max_attempts=1000):
    for _ in range(max_attempts):
        if chrom is None:
            chrom = random.choice(list(genome.keys()))
        else:
            current_chrom = chrom
        
        chrom_length = len(genome[current_chrom])

        if chrom_length <= length:
            continue

        start = random.randint(0, chrom_length - length)

        end = start + length

        if not overlaps_peak(current_chrom, start, end, peak_lookup):
            return (current_chrom, start, end)
    
    return None

def build_dataset(peaks_df, genome, context_size=200, max_samples=None):
    rows = []

    peak_lookup = build_peak_lookup(peaks_df)

    if max_samples is not None:
        peaks_df = peaks_df.sample(
            n=min(max_samples, len(peaks_df)), random_state=42
        )
    
    for _, row in peaks_df.iterrows():
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]

        peak_len = end - start

        #positive
        peak_seq = extract_sequences(genome, chrom, start, end)
        ctx_start = max(0, start - context_size)
        ctx_end = min(len(genome[chrom]), end + context_size)

        full_seq = extract_sequences(genome, chrom, ctx_start, ctx_end)

        left_ctx = extract_sequences(genome, chrom, ctx_start, start)
        right_ctx = extract_sequences(genome, chrom, end, ctx_end)

        context_seq = (left_ctx or "") + (right_ctx or "")

        if peak_seq is not None and full_seq is not None:
            rows.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "label": 1,
                "peak_seq": peak_seq,
                "context_seq": context_seq,
                "full_seq": full_seq
            })
        
        #negative
        neg_region = sample_negative_region(genome, peak_len, peak_lookup, chrom=chrom)

        if neg_region is None:
            continue

        neg_chrom, neg_start, neg_end = neg_region

        neg_peak_seq = extract_sequences(genome, neg_chrom, neg_start, neg_end)
        neg_ctx_start = max(0, neg_start - context_size)
        neg_ctx_end = min(len(genome[neg_chrom]), neg_end + context_size)

        neg_full_seq = extract_sequences(genome, neg_chrom, neg_ctx_start, neg_ctx_end)
        neg_left_ctx = extract_sequences(genome, neg_chrom, neg_ctx_start, neg_start)
        neg_right_ctx = extract_sequences(genome, neg_chrom, neg_end, neg_ctx_end)

        neg_context_seq = (neg_left_ctx or "") + (neg_right_ctx or "")

        if neg_peak_seq is not None and neg_full_seq is not None:
            rows.append({
                "chrom": neg_chrom,
                "start": neg_start,
                "end": neg_end,
                "label": 0,
                "peak_seq": neg_peak_seq,
                "context_seq": neg_context_seq,
                "full_seq": neg_full_seq
            })
    
    return pd.DataFrame(rows)

def save_dataset(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}"
          f" with {len(df)} samples.")
