// Display-only interpolation. Stored points and global_step remain untouched.
function createEpochAxis(boundaries) {
  const byStep=new Map([[0,0]]);
  for(const item of boundaries || []) {
    const step=Number(item.step),epoch=Number(item.epoch);
    if(Number.isFinite(step)&&Number.isFinite(epoch)&&step>=0&&epoch>=0)byStep.set(step,epoch);
  }
  const ordered=[...byStep].sort((a,b)=>a[0]-b[0]);
  const anchors=[];
  for(const [step,epoch] of ordered)if(!anchors.length||epoch>anchors.at(-1).epoch)anchors.push({step,epoch});
  const at=step=>{
    step=Number(step);
    if(!Number.isFinite(step)||step<0)return null;
    const exact=anchors.find(point=>point.step===step);
    if(exact)return exact.epoch;
    if(anchors.length<2)return null;
    let right=anchors.findIndex(point=>point.step>step);
    if(right<0)right=anchors.length-1;
    const a=anchors[right-1],b=anchors[right];
    return a.epoch+(step-a.step)/(b.step-a.step)*(b.epoch-a.epoch);
  };
  return {at,estimated:step=>anchors.length>1&&Number(step)>anchors.at(-1).step,
    format:epoch=>Number(epoch).toFixed(4).replace(/0+$/,'').replace(/\.$/,'')};
}
module.exports={createEpochAxis};
