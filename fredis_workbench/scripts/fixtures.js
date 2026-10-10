// Shared check fixtures: prepared-data chemistry and a trained model configuration.
const fs=require('fs'),path=require('path');
const fragmenter=JSON.parse(fs.readFileSync(path.join(__dirname,'../../fredis/presets/fragment/single_bond_cleavage_pos.fragmenter.pft'),'utf8'));
const symbols=['C','N','O','P','S','F','Cl','Br','I'];
const chemistry={fragmenter_params:fragmenter,symbols};
const model={architecture:'fragment-tree-physical-ion',fragmenter_params:fragmenter,
  mol_encoder_params:{symbols,node_dim:32,graph_dim:128,num_layers:4,num_heads:8,max_degree:4,max_spatial_dist:3,max_edge_dist:3,descriptors:['FormalCharge']},
  reaction_model_params:{hidden_dim:128,branch_main_adduct_dim:128,num_heads:4,state_num_layers:2},
  spectrum_model_params:{hidden_dim:128,num_layers:2,num_heads:4},
  generation_params:{},adduct_type_strs:['[M+H]+','[M+Na]+','[M+H4N]+','[2M+H]+']};
// The Data Preparation form state for the same chemistry.
const dataConfig={fragmenterParams:fragmenter,symbols};
module.exports={chemistry,model,dataConfig};
