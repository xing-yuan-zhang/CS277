python .\candidate_diffusion_with_attributes.py `
  --graph .\inputs\pkl\ppi_subgraph.pkl `
  --seeds .\configs\seeds.yaml `
  --outdir .\outputs\diffusion `
  --alpha 0.70 0.80 0.85 `
  --weight_attr weight `
  --node_prior_mode target `
  --node_prior_col node_prior_mult
