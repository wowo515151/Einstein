import {C,finiteNumber,vector} from "./array.js";
import {Vector4,Covector4,Metric4,Tensor4,LorentzTransform} from "./algebra.js";
import {Event4,SpacetimeInterval,InertialFrame,LightCone,SpatialSlice} from "./spacetime.js";
import {FourVelocity,FourMomentum,FourCurrent,EMFieldTensor,MaxwellEquations,RetardedField} from "./mechanics.js";

// Lib4 constructs a new metric wrapper; unchanged-device tensor storage can remain shared.
function contextMetric(metric){if(metric==null)return Metric4.minkowski();if(!(metric instanceof Metric4))return new Metric4(metric);metric._checkedComponents();const result=new Metric4(metric.components);result.components=metric.components;return result;}
function contextOptions(options,allowed,label){if(options===null||typeof options!=="object"||Array.isArray(options))throw new TypeError(`${label} options must be an object`);for(const key of Object.keys(options))if(!allowed.includes(key))throw new TypeError(`unknown ${label} option ${JSON.stringify(key)}`);return options;}

export class RelativityContext {
  constructor(options={}){const {metric=null,c=C,units="SI",epsilon0=null,device="cpu"}=contextOptions(options,["metric","c","units","epsilon0","device"],"RelativityContext");finiteNumber(c,"c");if(!["SI","natural"].includes(units))throw new TypeError("units must be SI or natural");if(units==="natural"&&c!==1)throw new RangeError("natural units require c=1");if(c<=0)throw new RangeError("c must be positive");this.c=c;this.units=units;this.epsilon0=finiteNumber(epsilon0??(units==="natural"?1/(4*Math.PI):8.8541878128e-12),"epsilon0");if(this.epsilon0<=0)throw new RangeError("epsilon0 must be positive");this.metric=contextMetric(metric);this.device=device;}
  createVector(components){return new Vector4(components);}
  createEvent(position){return new Event4(position,null,this.c);}
  createCovector(components){return new Covector4(components);}
  createTensor(components,indexTypes){return new Tensor4(components,indexTypes);}
  createSpacetimeInterval(displacement){return new SpacetimeInterval(displacement,{c:this.c});}
  createInertialFrame(origin,options={}){const {orientation=null,relativeVelocity=[0,0,0]}=contextOptions(options,["orientation","relativeVelocity"],"createInertialFrame");return new InertialFrame({origin,orientation,relativeVelocity,c:this.c});}
  createLightCone(apex,options={}){const {branch="both",atol=1e-10}=contextOptions(options,["branch","atol"],"createLightCone");return new LightCone(apex,{branch,atol,c:this.c});}
  createSpatialSlice(origin,normal,basis=null,options={}){return new SpatialSlice(origin,normal,{...contextOptions(options,["atol"],"createSpatialSlice"),basis,c:this.c});}
  createFourVelocity(components){return new FourVelocity(components,{c:this.c});}
  fourVelocityFromVelocity3(velocity){return FourVelocity.fromVelocity3(velocity,this.c);}
  createFourMomentum(components){return new FourMomentum(components,{c:this.c});}
  fourMomentumFromMassVelocity(mass,velocity){return FourMomentum.fromMassVelocity(mass,velocity,this.c);}
  createEMFieldTensor(electric,magnetic){return EMFieldTensor.fromEB(electric,magnetic,{c:this.c,epsilon0:this.epsilon0});}
  createFourCurrent(rho,current){return FourCurrent.fromChargeCurrent(rho,current,{c:this.c});}
  createMaxwellEquations(){return new MaxwellEquations({c:this.c,epsilon0:this.epsilon0});}
  createRetardedField(worldline,velocity,charge,maxSpeed,options={}){const config=contextOptions(options,["c","epsilon0","maxIterations","tolerance"],"RetardedField");if(Object.hasOwn(config,"c"))throw new TypeError("c is supplied by RelativityContext and cannot be overridden");if(Object.hasOwn(config,"epsilon0"))throw new TypeError("epsilon0 is supplied by RelativityContext and cannot be overridden");return new RetardedField(worldline,velocity,charge,maxSpeed,{...config,c:this.c,epsilon0:this.epsilon0});}
  toDevice(device){return new RelativityContext({metric:this.metric,c:this.c,units:this.units,epsilon0:this.epsilon0,device});}
  setUnits(units,options={}){const {c=null,epsilon0=null}=contextOptions(options,["c","epsilon0"],"setUnits");return new RelativityContext({metric:this.metric,c:c??(units==="natural"?1:C),units,epsilon0,device:this.device});}
}
