# MethylBERT BAM Lists

The paper-style MethylBERT pipeline is configured through `configs/methylbert_oac_paper.env`.

Create these list files on the cluster, with one absolute BAM or CRAM path per line:

- `oac_dmr_tumour_bams.list`
- `oac_dmr_normal_bams.list`
- `oac_train_tumour_bams.list`
- `oac_train_control_cfdna_bams.list`
- `oac_bulk_cfdna_bams.list`

For the PAT/DSS DMR path, create these list files with one absolute `.pat.gz`
path per line:

- `oac_dmr_tumour_pats.list`
- `oac_dmr_normal_pats.list`

The DMR tumour and training tumour lists may be identical if the same tumour-tissue BAMs are used for DMR discovery and read-classifier fine-tuning.

The DMR normal list should contain the non-tumour/background samples used by DSS to define tumour-specific DMRs. The training control list should contain healthy-control cfDNA BAMs used as negative read examples for MethylBERT fine-tuning.

Use PATs that have already been flipped to bisulfite convention, where `C`
means methylated and `T` means unmethylated. If using raw unflipped TAPS PATs,
set `PAT_METHYLATED_CHAR=T` and `PAT_UNMETHYLATED_CHAR=C`.
