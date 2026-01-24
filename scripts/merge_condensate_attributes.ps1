python .\merge_attributes.py `
  --run all `
  --nodes .\inputs\ppi\subgraph\subgraph_nodes.tsv `
  --interpro .\inputs\annotations\domains\interproscan_summary.tsv `
  --llpsdb_in .\inputs\annotations\LLPS\LLPSDB\unambiguous\Phase_separation\protein.xls `
  --phasepdb_in .\inputs\annotations\LLPS\PhaSepDB\PhaSepDB_human.tsv `
  --llpsdb_out .\inputs\annotations\LLPS\LLPSDB\unambiguous\Phase_separation\llpsdb_positive.tsv `
  --phasepdb_out .\inputs\annotations\LLPS\PhaSepDB\phasepdb_positive.tsv `
  --elm .\inputs\annotations\motifs\ELM\elm_instances_human.tsv `
  --out .\inputs\ppi\subgraph\subgraph_node_attributes.tsv
