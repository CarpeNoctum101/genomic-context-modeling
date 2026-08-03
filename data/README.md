# Data

The datasets used in this project are too large to be hosted on GitHub.

## Required Files

```
data/
├── hg38.fa
├── hg38.fa.fai
├── peaks.bed
```

## 1. Human Reference Genome (GRCh38)

Download the FASTA from UCSC:

https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/

After downloading:

```
hg38.fa
```

Index the genome with samtools:

```bash
samtools faidx hg38.fa
```

This generates

```
hg38.fa.fai
```

## 2. ENCODE DNase-seq Peaks

Download the narrowPeak file used for this project from ENCODE.

Rename the downloaded file

```
peaks.bed
```

and place it inside the `data/` directory.

## Directory Layout

```
data/
├── hg38.fa
├── hg38.fa.fai
└── peaks.bed
```

The preprocessing scripts expect these filenames.