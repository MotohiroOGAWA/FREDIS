// Mol Encoder architecture fields for fragment-tree training.
// [form key, CLI flag, label, default value, kind, config key, help]
module.exports = [
  ['nodeDim','node-dim','Node dimension',16,'integer','node_dim','Mol Encoder atom embedding width. Must be divisible by each attention head count.'],
  ['graphDim','graph-dim','Graph dimension',64,'integer','graph_dim','Mol Encoder graph embedding width. Must be a multiple of the node dimension.'],
  ['numLayers','num-layers','Layers',2,'integer','num_layers','Number of Graphormer layers.'],
  ['numHeads','num-heads','Attention heads',4,'integer','num_heads','Number of attention heads. Must divide the node dimension.'],
  ['maxDegree','max-degree','Maximum degree',4,'integer','max_degree','Maximum atom degree represented by the graph encoder.'],
  ['maxSpatialDist','max-spatial-dist','Maximum spatial distance',3,'integer','max_spatial_dist','Maximum shortest path distance represented by the graph encoder.'],
  ['maxEdgeDist','max-edge-dist','Maximum edge distance',3,'integer','max_edge_dist','Maximum bond path distance represented by the graph encoder.'],
  ['dropout','dropout','Dropout',0.5,'number','dropout','Dropout applied inside the molecular Graphormer encoder.'],
];
