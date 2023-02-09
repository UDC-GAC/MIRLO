#!/usr/bin/env Rscript
# Extract features from RNA and protein sequences using LION
# Usage: LION_feature_extract.R rna.fa pro.fa output.txt

library(LION)
library(seqinr)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
    stop("Usage: LION_feature_extract.R rna.fa pro.fa output.txt")
}

rna_seqs <- seqinr::read.fasta(args[1])
pro_seqs <- seqinr::read.fasta(args[2])

names <- data.frame(
    RNAs = names(rna_seqs),
    Proteins = names(pro_seqs)
)
sequence_features <- run_LION(
    seqRNA = rna_seqs,
    seqPro = pro_seqs,
    mode = "feature",
    parallel.cores = -1
)

features <- cbind(
    names,
    sequence_features
)

write.table(
    features,
    file = args[3],
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
)
