import assert from "node:assert/strict";
import test from "node:test";
import { createViewerWithRuntime } from "../src/three-viz/core/mount.js";
import { createResourceRegistry } from "../src/three-viz/core/resource-registry.js";
import { pointCloudPlotter } from "../src/three-viz/plotters/point-cloud-plotter.js";

class FakeVector3 {
  constructor(x=0,y=0,z=0){this.set(x,y,z);}
  set(x,y,z){this.x=x;this.y=y;this.z=z;return this;}
  copy(value){return this.set(value.x,value.y,value.z);}
  clone(){return new FakeVector3(this.x,this.y,this.z);}
  distanceTo(value){return Math.hypot(this.x-value.x,this.y-value.y,this.z-value.z);}
}

class FakeCamera {
  constructor(){this.position=new FakeVector3();this.up=new FakeVector3(0,1,0);this.projections=0;}
  lookAt(target){this.target=target.clone();}
  updateProjectionMatrix(){this.projections+=1;}
}

function makeRuntime(){
  const state={renders:0,rendererDisposals:0,loops:[],scene:null};
  class FakeScene { constructor(){this.children=[];state.scene=this;} add(...items){this.children.push(...items);} }
  class FakeRenderer {
    constructor(){this.domElement={parentNode:null};}
    setPixelRatio(value){this.pixelRatio=value;}
    setSize(width,height){this.size=[width,height];}
    render(){state.renders+=1;}
    setAnimationLoop(callback){state.loops.push(callback);}
    dispose(){state.rendererDisposals+=1;}
  }
  const THREE={Scene:FakeScene,WebGLRenderer:FakeRenderer,PerspectiveCamera:FakeCamera,Vector3:FakeVector3};
  const runtime={THREE};
  return {runtime,state};
}

function makeContainer(){
  const container={clientWidth:640,clientHeight:480,children:[],ownerDocument:{defaultView:{devicePixelRatio:1}}};
  container.appendChild=element=>{element.parentNode=container;container.children.push(element);};
  container.removeChild=element=>{container.children.splice(container.children.indexOf(element),1);element.parentNode=null;};
  return container;
}

test("ThreeViz viewer mounts, updates, and disposes its renderer and canvas once",async()=>{
  const {runtime,state}=makeRuntime(),container=makeContainer();
  let plotterDisposals=0,updates=0,context;
  const definition={configuration:{},plotter:{capabilities:[],async create(value){context=value;return {async update(){updates+=1;},dispose(){plotterDisposals+=1;}};}}};
  const viewer=await createViewerWithRuntime(container,definition,runtime);
  assert.equal(container.children.length,1);assert.equal(state.renders,1);assert.equal(context.scene,state.scene);
  await viewer.update({sample:1},{time:.5});
  assert.equal(updates,1);assert.equal(state.renders,2);
  viewer.dispose();viewer.dispose();
  assert.equal(plotterDisposals,1);assert.equal(state.rendererDisposals,1);assert.equal(container.children.length,0);
  assert.throws(()=>viewer.getCameraPose(),/disposed/);
});

test("ThreeViz viewer cleans initialized core resources when plotter setup fails",async()=>{
  const {runtime,state}=makeRuntime(),container=makeContainer(),setupFailure=new Error("synthetic plotter setup failure");
  const definition={configuration:{},plotter:{create(){throw setupFailure;}}};
  await assert.rejects(createViewerWithRuntime(container,definition,runtime),error=>error.code==="MOUNT_FAILED"&&error.cause===setupFailure);
  assert.equal(state.rendererDisposals,1);assert.equal(container.children.length,0);
});

test("selection passes overlay hits and reports the first selectable point",async()=>{
  const {runtime}=makeRuntime(),container=makeContainer(),listeners={},documentListeners={},selections=[];
  container.ownerDocument.addEventListener=(name,callback)=>{documentListeners[name]=callback;};
  container.ownerDocument.removeEventListener=name=>{delete documentListeners[name];};
  const Renderer=runtime.THREE.WebGLRenderer;
  runtime.THREE.WebGLRenderer=class extends Renderer {
    constructor(){super();this.domElement.addEventListener=(name,callback)=>{listeners[name]=callback;};this.domElement.removeEventListener=name=>{delete listeners[name];};this.domElement.getBoundingClientRect=()=>({left:0,top:0,width:100,height:100});}
  };
  runtime.THREE.Vector2=class {};
  runtime.THREE.Raycaster=class {setFromCamera(){} intersectObjects(){return [{object:{kind:"overlay"}},{object:{kind:"point"},index:0}];}};
  const definition={configuration:{},callbacks:{onSelection(value){selections.push(value);}},plotter:{capabilities:["selection"],create(){return {update(){},dispose(){},describeSelection(hit){return hit.object.kind==="point"?{label:"sample"}:undefined;}};}}};
  const viewer=await createViewerWithRuntime(container,definition,runtime);
  listeners.pointerdown({pointerId:1,pointerType:"mouse",button:0,clientX:50,clientY:50});
  listeners.pointerup({pointerId:1,clientX:50,clientY:50});
  assert.deepEqual(selections,[{label:"sample"}]);
  listeners.pointerdown({pointerId:2,pointerType:"mouse",button:0,clientX:50,clientY:50});
  listeners.pointerup({pointerId:2,clientX:70,clientY:50});
  listeners.pointerdown({pointerId:3,pointerType:"mouse",button:2,clientX:50,clientY:50});
  listeners.pointerup({pointerId:3,clientX:50,clientY:50});
  listeners.pointerdown({pointerId:4,pointerType:"touch",isPrimary:true,clientX:50,clientY:50});
  listeners.pointercancel({pointerId:4});
  listeners.pointerup({pointerId:4,clientX:50,clientY:50});
  listeners.pointerdown({pointerId:5,pointerType:"mouse",button:0,clientX:50,clientY:50});
  documentListeners.pointermove({pointerId:5,clientX:70,clientY:50});
  listeners.pointerup({pointerId:5,clientX:50,clientY:50});
  listeners.pointerdown({pointerId:6,pointerType:"touch",isPrimary:true,clientX:50,clientY:50});
  listeners.pointerdown({pointerId:7,pointerType:"touch",isPrimary:false,clientX:60,clientY:50});
  listeners.pointerup({pointerId:6,clientX:50,clientY:50});
  listeners.pointerdown({pointerId:8,pointerType:"touch",isPrimary:true,clientX:50,clientY:50});
  documentListeners.pointerdown({pointerId:9,pointerType:"touch",isPrimary:false,clientX:200,clientY:50});
  listeners.pointerup({pointerId:8,clientX:50,clientY:50});
  assert.equal(selections.length,1);
  listeners.pointerdown({pointerId:10,pointerType:"touch",isPrimary:true,clientX:50,clientY:50});
  listeners.pointerup({pointerId:10,clientX:50,clientY:50});
  assert.equal(selections.length,2);
  viewer.dispose();
  assert.equal(listeners.pointerup,undefined);
  assert.equal(listeners.pointerdown,undefined);
  assert.equal(documentListeners.pointermove,undefined);
  assert.equal(documentListeners.pointerup,undefined);
  assert.equal(documentListeners.pointerdown,undefined);
  assert.equal(listeners.pointercancel,undefined);
});

test("point cloud selects only its own points, even when an overlay has an index",()=>{
  class Geometry {setAttribute(){} dispose(){}}
  class Material {dispose(){}}
  class Points {constructor(geometry,material){this.geometry=geometry;this.material=material;}}
  const THREE={BufferGeometry:Geometry,Float32BufferAttribute:class {},PointsMaterial:Material,Points};
  const registry=createResourceRegistry(),resources=registry.createOwner("points"),scene={add(value){this.points=value;},remove(){}};
  const plotter=pointCloudPlotter.create({THREE,scene,resources,requestRender(){}},{configuration:{}},[{id:"a",label:"First",x:1,y:2,z:3}]);
  assert.equal(plotter.describeSelection({object:{},index:0}),undefined);
  assert.deepEqual(plotter.describeSelection({object:scene.points,index:0}),{id:"a",label:"First",values:{id:"a",label:"First",x:1,y:2,z:3}});
  plotter.dispose();registry.disposeAll();
});

test("point cloud restores the previous sample and geometry when an update fails",()=>{
  const geometries=[];
  class Geometry {constructor(){this.disposed=0;geometries.push(this);}setAttribute(){}dispose(){this.disposed++;}}
  class Color {constructor(value){this.value=value;}clone(){return new Color(this.value);}copy(other){this.value=other.value;}set(value){this.value=value;}}
  class Material {constructor({color}){this.color=new Color(color);}dispose(){}}
  class Points {constructor(geometry,material){this.geometry=geometry;this.material=material;}}
  const THREE={BufferGeometry:Geometry,Float32BufferAttribute:class {},PointsMaterial:Material,Points},registry=createResourceRegistry(),resources=registry.createOwner("points");
  const scene={add(value){this.points=value;},remove(){}},old=[{id:"old",x:1,y:2,z:3}],next=[{id:"new",x:4,y:5,z:6}];
  let fail=true;
  const plotter=pointCloudPlotter.create({THREE,scene,resources,requestRender(){if(fail)throw new Error("render failed");}},{configuration:{pointCloud:{color:1}}},old);
  assert.throws(()=>plotter.update(next,{pointCloud:{color:2}}),/render failed/);
  assert.equal(scene.points.geometry,geometries[0]);assert.equal(scene.points.material.color.value,1);
  assert.equal(plotter.describeSelection({object:scene.points,index:0}).id,"old");
  assert.equal(geometries[0].disposed,0);assert.equal(geometries[1].disposed,1);
  fail=false;plotter.update(next,{pointCloud:{color:2}});
  assert.equal(plotter.describeSelection({object:scene.points,index:0}).id,"new");
  assert.equal(scene.points.material.color.value,2);assert.equal(geometries[0].disposed,1);
  plotter.dispose();registry.disposeAll();
});

test("ThreeViz resource registry retains shared resources until the last owner releases them",()=>{
  const registry=createResourceRegistry();let disposals=0;
  const resource={dispose(){disposals+=1;}};
  const first=registry.createOwner("first"),second=registry.createOwner("second");
  first.track(resource);second.track(resource);
  assert.deepEqual(registry.stats,{owners:2,resources:1,disposed:false});
  first.dispose();assert.equal(disposals,0);assert.equal(registry.stats.resources,1);
  second.release(resource);second.release(resource);
  assert.equal(disposals,1);assert.equal(registry.stats.resources,0);
  registry.disposeAll();registry.disposeAll();
  assert.equal(disposals,1);assert.equal(registry.stats.disposed,true);
});
