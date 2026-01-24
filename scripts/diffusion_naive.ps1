python .\candidate_diffusion.py `
  --graph .\inputs\pkl\ppi_subgraph.pkl `
  --seeds .\configs\seeds.yaml `
  --outdir .\outputs\diffusion `
  --alpha 0.7 0.8 0.85 `
  --topn 200 `
  --weight_attr weight `
  --qc_topk 50 `
  --max_iter 500 `
  --tol 1e-10