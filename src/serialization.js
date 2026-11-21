// Explicit, versioned JSON interchange for finite value objects.
import {Vector4,Covector4,Matrix4,Metric4,Tensor4,Bivector4,LeviCivita4,LorentzTransform,PoincareTransform} from "./algebra.js";
import {FourVelocity,FourMomentum,FourCurrent,EMFieldTensor,MaxwellEquations} from "./mechanics.js";
import {Event4,SpacetimeInterval,LightCone,SpatialSlice,InertialFrame,WorldlineSegment,Worldline} from "./spacetime.js";
import {RelativityContext} from "./context.js";

function eventFromData(data){
  if(!data||typeof data!=="object"||Array.isArray(data)||Object.keys(data).length!==3||!["position","frame","c"].every(key=>Object.hasOwn(data,key)))throw new TypeError("invalid serialized event fields");
  if(data.frame!==null&&typeof data.frame!=="string")throw new TypeError("event frame must be null or a string for serialization");
  return new Event4(data.position,data.frame,data.c);
}

function segmentFromData(data){
  const keys=["start","end","u0","u1","span"];
  if(!data||typeof data!=="object"||Array.isArray(data)||Object.keys(data).length!==keys.length||!keys.every(key=>Object.hasOwn(data,key)))throw new TypeError("invalid serialized worldline segment fields");
  return new WorldlineSegment(eventFromData(data.start),eventFromData(data.end),data.u0,data.u1,data.span);
}
function worldlineFromData(data){
  if(!Array.isArray(data.segments)||data.segments.length<1||data.segments.length>65536)throw new RangeError("serialized worldline must have 1 to 65536 segments");
  return new Worldline(data.segments.map(segmentFromData));
}

function contextFromData(data){
  if(typeof data.device!=="string")throw new TypeError("context device label must be a string for serialization");
  if(typeof data.epsilon0!=="number"||!Number.isFinite(data.epsilon0)||data.epsilon0<=0)throw new RangeError("context epsilon0 must be positive and finite");
  return new RelativityContext({metric:new Metric4(data.metricComponents),c:data.c,units:data.units,epsilon0:data.epsilon0,device:data.device});
}
function contextData(value){
  if(!(value.metric instanceof Metric4))throw new TypeError("context metric must be a Metric4");
  return {metricComponents:value.metric.components,c:value.c,units:value.units,epsilon0:value.epsilon0,device:value.device};
}

const formats = [
  ["Vector4",Vector4,["components"], data=>new Vector4(data.components)],
  ["Covector4",Covector4,["components"], data=>new Covector4(data.components)],
  ["Matrix4",Matrix4,["components"], data=>new Matrix4(data.components)],
  ["Metric4",Metric4,["components"], data=>new Metric4(data.components)],
  ["Tensor4",Tensor4,["components","indexTypes"], data=>new Tensor4(data.components,data.indexTypes)],
  ["Bivector4",Bivector4,["components"], data=>new Bivector4(data.components)],
  ["LeviCivita4",LeviCivita4,["metricComponents","orientation"], data=>new LeviCivita4(new Metric4(data.metricComponents),data.orientation), value=>({metricComponents:value.metric.components,orientation:value.orientation})],
  ["PoincareTransform",PoincareTransform,["matrix","translation"], data=>new PoincareTransform(new LorentzTransform(data.matrix,{validate:true}),data.translation), value=>({matrix:value.lorentz._checkedMatrix(),translation:value.translation})],
  ["LorentzTransform",LorentzTransform,["matrix"], data=>new LorentzTransform(data.matrix,{validate:true})],
  ["Event4",Event4,["position","frame","c"], eventFromData],
  ["SpacetimeInterval",SpacetimeInterval,["displacement","c","atol"], data=>new SpacetimeInterval(data.displacement,{c:data.c,atol:data.atol})],
  ["LightCone",LightCone,["apex","branch","c","atol"], data=>new LightCone(eventFromData(data.apex),{branch:data.branch,c:data.c,atol:data.atol})],
  ["SpatialSlice",SpatialSlice,["origin","normal","basis","c","atol"], data=>new SpatialSlice(eventFromData(data.origin),data.normal,{basis:data.basis,c:data.c,atol:data.atol}), value=>{value._validateShape();return {origin:new Event4(value.originPosition,value.originFrame,value.originC),normal:value.normal,basis:value.basis,c:value.c,atol:value.atol};}],
  ["WorldlineSegment",WorldlineSegment,["start","end","u0","u1","span"],segmentFromData],
  ["Worldline",Worldline,["segments"],worldlineFromData],
  ["InertialFrame",InertialFrame,["origin","orientation","relativeVelocity","c"], data=>new InertialFrame({origin:eventFromData(data.origin),orientation:data.orientation,relativeVelocity:data.relativeVelocity,c:data.c})],
  ["RelativityContext",RelativityContext,["metricComponents","c","units","epsilon0","device"],contextFromData,contextData],
  ["MaxwellEquations",MaxwellEquations,["c","epsilon0"], data=>new MaxwellEquations({c:data.c,epsilon0:data.epsilon0})],
  ["FourVelocity",FourVelocity,["components","c"], data=>new FourVelocity(data.components,{c:data.c})],
  ["FourMomentum",FourMomentum,["components","c"], data=>new FourMomentum(data.components,{c:data.c})],
  ["FourCurrent",FourCurrent,["components","c"], data=>new FourCurrent(data.components,{c:data.c})],
  ["EMFieldTensor",EMFieldTensor,["components","c","epsilon0"], data=>new EMFieldTensor(data.components,{c:data.c,epsilon0:data.epsilon0})],
];
const MAX_JSON_LENGTH=32_000_000;

/** Return a versioned JSON string for a supported, currently valid value object. */
export function serializeRelativity(value){
  const format=formats.find(([,type])=>value instanceof type);
  if(!format)throw new TypeError("unsupported relativity value");
  const [type,,fields,construct,extract]=format;
  const checked=construct(extract?extract(value):value),data=extract?extract(checked):Object.fromEntries(fields.map(field=>[field,checked[field]]));
  const json=JSON.stringify({format:"relativity-value",version:1,type,data});
  if(json.length>MAX_JSON_LENGTH)throw new RangeError("serialized value exceeds the length limit");
  return json;
}

/** Restore a supported value and validate its data using the public constructors. */
export function deserializeRelativity(json){
  if(typeof json!=="string")throw new TypeError("serialized value must be a JSON string");
  if(json.length>MAX_JSON_LENGTH)throw new RangeError("serialized value exceeds the length limit");
  const record=JSON.parse(json);
  if(!record||typeof record!=="object"||Array.isArray(record)||record.format!=="relativity-value"||record.version!==1)throw new TypeError("unsupported relativity serialization format");
  const format=formats.find(([name])=>name===record.type);
  if(!format)throw new TypeError("unsupported relativity value type");
  const data=record.data,fields=format[2];
  if(!data||typeof data!=="object"||Array.isArray(data)||Object.keys(data).length!==fields.length||fields.some(field=>!Object.hasOwn(data,field)))throw new TypeError("invalid serialized value fields");
  return format[3](data);
}
