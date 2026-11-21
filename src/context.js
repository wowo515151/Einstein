import {C,finiteNumber,vector} from "./array.js";
import {Vector4,Covector4,Metric4,Tensor4,LorentzTransform} from "./algebra.js";
import {Event4,SpacetimeInterval,InertialFrame,LightCone,SpatialSlice} from "./spacetime.js";
import {FourVelocity,FourMomentum,FourCurrent,EMFieldTensor,MaxwellEquations,RetardedField} from "./mechanics.js";

export class RelativityContext {
  constructor({metric=null,c=C,units="SI",epsilon0=null,device="cpu"}={}){finiteNumber(c,"c");if(!["SI","natural"].includes(units))throw new TypeError("units must be SI or natural");if(units==="natural"&&c!==1)throw new RangeError("natural units require c=1");if(c<=0)throw new RangeError("c must be positive");this.c=c;this.units=units;this.epsilon0=finiteNumber(epsilon0??(units==="natural"?1/(4*Math.PI):8.8541878128e-12),"epsilon0");if(this.epsilon0<=0)throw new RangeError("epsilon0 must be positive");this.metric=metric==null?Metric4.minkowski():metric instanceof Metric4?metric:new Metric4(metric);this.device=device;}
  createVector(components){return new Vector4(components);}
  createEvent(position){return new Event4(position,null,this.c);}
  createCovector(components){return new Covector4(components);}
  createTensor(components,indexTypes){return new Tensor4(components,indexTypes);}
  createSpacetimeInterval(displacement){return new SpacetimeInterval(displacement,{c:this.c});}
  createInertialFrame(origin,{orientation=null,relativeVelocity=[0,0,0]}={}){return new InertialFrame({origin,orientation,relativeVelocity,c:this.c});}
  createLightCone(apex,{branch="both",atol=1e-10}={}){return new LightCone(apex,{branch,atol,c:this.c});}
  createSpatialSlice(origin,normal,basis=null,options={}){return new SpatialSlice(origin,normal,{...options,basis,c:this.c});}
  createFourVelocity(components){return new FourVelocity(components,{c:this.c});}
  fourVelocityFromVelocity3(velocity){return FourVelocity.fromVelocity3(velocity,this.c);}
  createFourMomentum(components){return new FourMomentum(components,{c:this.c});}
  fourMomentumFromMassVelocity(mass,velocity){return FourMomentum.fromMassVelocity(mass,velocity,this.c);}
  createEMFieldTensor(electric,magnetic){return EMFieldTensor.fromEB(electric,magnetic,{c:this.c,epsilon0:this.epsilon0});}
  createFourCurrent(rho,current){return FourCurrent.fromChargeCurrent(rho,current,{c:this.c});}
  createMaxwellEquations(){return new MaxwellEquations({c:this.c,epsilon0:this.epsilon0});}
  createRetardedField(worldline,velocity,charge,maxSpeed,options={}){return new RetardedField(worldline,velocity,charge,maxSpeed,{...options,c:this.c,epsilon0:this.epsilon0});}
  toDevice(device){return new RelativityContext({metric:this.metric,c:this.c,units:this.units,epsilon0:this.epsilon0,device});}
  setUnits(units,{c=units==="natural"?1:C,epsilon0=null}={}){return new RelativityContext({metric:this.metric,c,units,epsilon0,device:this.device});}
}
