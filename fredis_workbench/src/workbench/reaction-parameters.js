// Expand the Workbench's shared controls into an explicit architecture per reaction.
const fields=['hidden_dim','graph_dim','num_layers','num_heads','neighborhood_hops','neighborhood_hidden_dim'];
function resolvePatternArchitecture(config){
 const architecture=config.patternArchitecture;
 if(!architecture||architecture.mode!=='shared')return config;
 const frozen=new Set((architecture.frozenPatternIds||[]).map(String));
 const patterns=config.modelConfig?.reaction_model_params?.pattern_model_params||{};
 for(const [pid,params]of Object.entries(patterns)){
  if(frozen.has(String(pid)))continue;
  for(const [key,value]of Object.entries(architecture.shared||{})){
   if(['hidden_dim','graph_dim','num_layers','num_heads','neighborhood_hops','neighborhood_hidden_dim'].includes(key))params[key]=value;
  }
 }
 return config;
}
module.exports={fields,resolvePatternArchitecture};
