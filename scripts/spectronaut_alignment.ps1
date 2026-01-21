python ms_vs_diffusion_overlay.py \
  --ms_tsv /mnt/data/20250630_15188_YYang_Report_Prot.tsv \
  --string_edges /path/to/edges.merged.uniprot.tsv \
  --diff_edges /mnt/data/subgraph_edges.tsv \
  --sec2pri_tsv /mnt/data/uniprot_sec2pri_9606.tsv \
  --outdir /mnt/data/ms_vs_diffusion_out \
  --label_topn 50
