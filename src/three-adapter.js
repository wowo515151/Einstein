import {Matrix4 as RelativityMatrix4,Vector4} from "./algebra.js";

function finiteComponents(value,length,name){
  if(!(Array.isArray(value)||ArrayBuffer.isView(value)&&!(value instanceof DataView))||value.length!==length)throw new TypeError(`${name} must contain ${length} finite numbers`);
  const components=Array.from(value);
  if(components.some(component=>typeof component!=="number"||!Number.isFinite(component)))throw new TypeError(`${name} must contain ${length} finite numbers`);
  return components;
}
function runtimeConstructor(THREE,name){
  if(typeof THREE?.[name]!=="function")throw new TypeError(`Three.js runtime must provide ${name}`);
  return THREE[name];
}
function readXYZ(value,name){
  if(!value||typeof value!=="object")throw new TypeError(`${name} must be a Three.js-compatible value`);
  return finiteComponents([value.x,value.y,value.z],3,name);
}
function readXYZW(value,name){
  if(!value||typeof value!=="object")throw new TypeError(`${name} must be a Three.js-compatible value`);
  return finiteComponents([value.x,value.y,value.z,value.w],4,name);
}

/** Copy a Three.js Vector3 into a plain spatial component triple. */
export function fromThreeVector3(value){return Object.freeze(readXYZ(value,"Vector3"));}
/** Construct a Three.js Vector3 from a spatial component triple. */
export function toThreeVector3(value,THREE){const Vector3=runtimeConstructor(THREE,"Vector3");return new Vector3(...finiteComponents(value,3,"spatial vector"));}
/** Copy Three.js Vector4 slots directly into a relativity Vector4. */
export function fromThreeVector4(value){return new Vector4(readXYZW(value,"Vector4"));}
/** Construct a Three.js Vector4 with slots in direct x/y/z/w order. */
export function toThreeVector4(value,THREE){const Vector4Class=runtimeConstructor(THREE,"Vector4"),components=finiteComponents(value instanceof Vector4?value.components:value,4,"four-vector");return new Vector4Class(...components);}
/** Convert Three.js's column-major Matrix4.elements into row-major relativity rows. */
export function fromThreeMatrix4(value){
  if(!value||typeof value!=="object")throw new TypeError("Matrix4 must be a Three.js-compatible value");
  const elements=finiteComponents(value.elements,16,"Matrix4.elements");
  return new RelativityMatrix4(Array.from({length:4},(_,row)=>Array.from({length:4},(_,column)=>elements[column*4+row])));
}
/** Construct a Three.js Matrix4; its set() method receives row-major arguments. */
export function toThreeMatrix4(value,THREE){
  const Matrix4Class=runtimeConstructor(THREE,"Matrix4"),relativityMatrix=new RelativityMatrix4(value instanceof RelativityMatrix4?value.components:value);
  if(relativityMatrix.batchRank!==0)throw new TypeError("matrix must be a single 4 by 4 value");
  const components=relativityMatrix.components;
  const matrix=new Matrix4Class();
  if(typeof matrix.set!=="function")throw new TypeError("Three.js Matrix4 runtime must provide set()");
  return matrix.set(...components.flat());
}
