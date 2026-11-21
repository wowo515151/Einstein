import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { runInNewContext } from "node:vm";
import { buildSceneData, fieldWorldlines, scenes, selectedWorldlineEvents } from "./scene-models.js";
import { describeSample } from "./sample-inspection.js";
import { createResourceRegistry } from "../src/three-viz/core/resource-registry.js";
import { pointCloudPlotter } from "../src/three-viz/plotters/point-cloud-plotter.js";

class Element {
  constructor(tag="div") { this.tag=tag;this.children=[];this.dataset={};this.attributes={};this.listeners={};this.textContent="";this.value="";this.hidden=false; }
  append(...items){this.children.push(...items);}
  appendChild(item){item.parentNode=this;this.children.push(item);return item;}
  removeChild(item){const index=this.children.indexOf(item);if(index>=0)this.children.splice(index,1);item.parentNode=null;return item;}
  replaceChildren(...items){this.children=items;}
  setAttribute(name,value){this.attributes[name]=String(value);}
  getAttribute(name){return this.attributes[name]??null;}
  removeAttribute(name){delete this.attributes[name];}
  addEventListener(name,callback){(this.listeners[name]??=[]).push(callback);}
  removeEventListener(name,callback){this.listeners[name]=(this.listeners[name]??[]).filter(item=>item!==callback);}
  focus(){this.focused=true;}
  scrollIntoView(){this.scrolledIntoView=true;}
  getBoundingClientRect(){return {left:0,top:0,width:640,height:480};}
  getContext(kind){return kind==="2d"?{measureText:()=>({width:16}),fillRect(){},strokeRect(){},fillText(){}}:null;}
  emit(name,extra={}){for(const callback of this.listeners[name]??[])callback({target:this,currentTarget:this,preventDefault(){},...extra});}
  querySelectorAll(tag){return this.children.flatMap(child=>[...(child.tag===tag?[child]:[]),...child.querySelectorAll(tag)]);}
}

test("application fallback supports keyboard sample inspection and scene changes",async()=>{
  const html=await readFile(new URL("./index.html",import.meta.url),"utf8"),elements=new Map();
  for(const match of html.matchAll(/\bid="([^"]+)"/g))elements.set(match[1],new Element());
  const byId=id=>{const element=elements.get(id);assert.ok(element,`missing element ${id}`);return element;};
  byId("speed-slider").value="0.6";byId("time-slider").value="0";byId("sample-number").value="1";
  byId("observer-frame").options=[new Element("option"),new Element("option")];
  const windowStub=new Element(),locationStub={hash:""};let clock=0,scheduled;
  const replacements={document:{getElementById:byId,createElement:tag=>new Element(tag)},window:windowStub,location:locationStub,history:{replaceState(_a,_b,url){locationStub.hash=url;}},matchMedia:()=>({matches:false,addEventListener(){}}),performance:{now:()=>clock},setInterval(callback){scheduled=callback;return 1;},clearInterval(){scheduled=undefined;}};
  const previous=new Map(Object.keys(replacements).map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const runtimeDescriptor=Object.getOwnPropertyDescriptor(globalThis,"EINSTEIN_THREE_RUNTIME");
  try {
    for(const [name,value] of Object.entries(replacements))Object.defineProperty(globalThis,name,{configurable:true,writable:true,value});
    delete globalThis.EINSTEIN_THREE_RUNTIME;
    const app=await import("./visualizations.js?fallback-integration");
    assert.equal(byId("renderer-fallback").hidden,false);
    assert.match(byId("scene-status").textContent,/unavailable/);
    assert.equal(byId("scene-list").querySelectorAll("button").length,10);
    byId("play-button").emit("click");assert.equal(byId("play-button").getAttribute("aria-pressed"),"true");
    clock=1000;scheduled();assert.equal(Number(byId("time-slider").value),.125);
    byId("speed-slider").value="0.8";byId("speed-slider").emit("input");
    byId("observer-frame").value="moving";byId("observer-frame").emit("change");
    byId("reset-button").emit("click");
    assert.equal(byId("play-button").getAttribute("aria-pressed"),"false");assert.equal(scheduled,undefined);
    assert.equal(Number(byId("time-slider").value),0);assert.equal(Number(byId("speed-slider").value),.6);assert.equal(byId("observer-frame").value,"lab");
    const rows=buildSceneData("minkowski",{time:0,beta:.6,observerFrame:"lab"});
    byId("sample-number").value="2";byId("inspect-sample").emit("click");
    assert.equal(byId("selection-readout").textContent,describeSample(rows,2));
    assert.equal(Number(byId("sample-number").max),rows.length);
    byId("time-slider").value="0.5";byId("time-slider").emit("input");
    assert.match(byId("selection-readout").textContent,/sample inspector/);
    assert.match(byId("scene-readout").textContent,/Moving marker: s² = 0\.0000; coincident with the origin/);
    byId("time-slider").value="0.25";byId("time-slider").emit("input");
    assert.match(byId("scene-readout").textContent,/Moving marker: s² = -0\.6400; timelike/);
    byId("observer-frame").value="moving";byId("observer-frame").emit("change");
    assert.match(byId("scene-readout").textContent,/Moving marker: s² = -0\.6400; timelike/);
    byId("observer-frame").value="lab";byId("observer-frame").emit("change");
    byId("time-slider").value="0.5";byId("time-slider").emit("input");
    byId("inspect-sample").emit("click");
    assert.equal(byId("selection-readout").textContent,describeSample(buildSceneData("minkowski",{time:.5,beta:.6,observerFrame:"lab"}),2));
    byId("sample-number").value=String(rows.length+1);byId("inspect-sample").emit("click");
    assert.match(byId("selection-readout").textContent,/Choose a sample/);
    const buttons=byId("scene-list").querySelectorAll("button");
    buttons.find(button=>button.dataset.scene==="light-cone").emit("click");
    byId("time-slider").value="0.75";byId("time-slider").emit("input");
    assert.match(byId("scene-readout").textContent,/future surface samples;.*surface null; r = 0 timelike, r = 2\|ct\| spacelike/);
    byId("time-slider").value="0.25";byId("time-slider").emit("input");
    assert.match(byId("scene-readout").textContent,/past surface samples;.*surface null; r = 0 timelike, r = 2\|ct\| spacelike/);
    byId("time-slider").value="0.50000001";byId("time-slider").emit("input");
    assert.match(byId("scene-readout").textContent,/ct = 5\.000e-8, surface null; r = 0 null, r = 2\|ct\| null/);
    byId("time-slider").value="0.5";byId("time-slider").emit("input");
    assert.match(byId("scene-readout").textContent,/apex is coincident.*r = 1 is spacelike/);
    buttons.find(button=>button.dataset.scene==="twins").emit("click");
    assert.equal(locationStub.hash,"#twins");assert.match(byId("scene-title").textContent,/Twin/);
    let skipDefaultPrevented=false;
    byId("skip-to-scene").emit("click",{preventDefault(){skipDefaultPrevented=true;}});
    assert.equal(skipDefaultPrevented,true);assert.equal(locationStub.hash,"#twins");
    assert.equal(byId("scene").focused,true);assert.equal(byId("scene").scrolledIntoView,true);
    locationStub.hash="#scene";windowStub.emit("hashchange");
    assert.match(byId("scene-title").textContent,/Twin/);
    locationStub.hash="#particle";windowStub.emit("hashchange");
    assert.equal(byId("scene-title").textContent,scenes.find(scene=>scene.id==="particle").title);
    buttons.find(button=>button.dataset.scene==="twins").emit("click");
    const twinRows=buildSceneData("twins",{time:0,beta:.6,observerFrame:"lab"});
    byId("sample-number").value="1";byId("inspect-sample").emit("click");
    assert.equal(byId("selection-readout").textContent,describeSample(twinRows,1));
    for (const scene of scenes) {
      const button=buttons.find(item=>item.dataset.scene===scene.id);
      button.emit("click");
      assert.equal(byId("scene-title").textContent,scene.title);
      assert.equal(button.getAttribute("aria-current"),"true");
      assert.ok(byId("scene-readout").textContent.length>0);
      byId("time-slider").value="0.35";byId("time-slider").emit("input");
      const sampleRows=buildSceneData(scene.id,{time:.35,beta:.6,observerFrame:"lab"});
      assert.equal(Number(byId("sample-number").max),sampleRows.length);
      byId("sample-number").value=String(sampleRows.length);byId("inspect-sample").emit("click");
      assert.equal(byId("selection-readout").textContent,describeSample(sampleRows,sampleRows.length));
    }
    byId("sample-number").value="99999";
    buttons.find(button=>button.dataset.scene==="particle").emit("click");
    assert.equal(byId("sample-number").value,"1");
    byId("inspect-sample").emit("click");
    assert.match(byId("selection-readout").textContent,/Selected sample 1 of/);
    byId("time-slider").value="1";byId("time-slider").emit("input");
    byId("sample-number").value=byId("sample-number").max;
    assert.ok(Number(byId("sample-number").value)>1);
    byId("time-slider").value="0";byId("time-slider").emit("input");
    assert.equal(byId("sample-number").value,"1");
    assert.equal(byId("sample-number").max,"1");
    byId("inspect-sample").emit("click");
    assert.match(byId("selection-readout").textContent,/Selected sample 1 of 1/);
    const vendor=await readFile(new URL("./vendor/three-r128.min.js",import.meta.url),"utf8"),sandbox={console};
    runInNewContext(vendor,sandbox,{filename:"local-three-r128.min.js"});
    const THREE=sandbox.THREE,registry=createResourceRegistry(),resources=registry.createOwner("scene geometry"),colors=Object.fromEntries(scenes.map(scene=>[scene.id,0x55b9ff]));
    assert.equal(THREE.REVISION,"128");
    for(const model of scenes)for(const beta of [.05,.95])for(const time of [0,.5,1]){
      const scene=new THREE.Scene(),rows=buildSceneData(model.id,{time,beta,observerFrame:"lab"});
      const release=app.drawSceneOverlays({THREE,scene,resources},model.id,rows,colors,{time,beta});
      assert.ok(scene.children.length>0,`${model.id} has scene overlays`);
      for(const child of scene.children){
        const positions=child.geometry?.getAttribute("position");
        assert.ok(positions,`${model.id} overlay has position data`);
        assert.ok(Array.from(positions.array).every(Number.isFinite),`${model.id} overlay positions are finite`);
        const normals=child.geometry.getAttribute("normal");
        if(normals)assert.ok(Array.from(normals.array).every(Number.isFinite),`${model.id} overlay normals are finite`);
      }
      if(model.id==="particle"||model.id==="geodesic"){
        const endpoints=new Map();for(const row of rows)endpoints.set(model.id==="particle"?"particle":row.label,row);
        assert.equal(endpoints.size,model.id==="particle"?1:8);
        [...endpoints.values()].forEach((row,index)=>{
          const marker=scene.getObjectByName(`${model.id}-selected-endpoint-${index}`);
          assert.ok(marker);assert.equal(marker.position.x,row.x);assert.equal(marker.position.y,row.y);assert.equal(marker.position.z,row.z);
        });
      }
      if(model.id==="field-slices")fieldWorldlines(beta,time).forEach((line,index)=>{
        const marker=scene.getObjectByName(`field-observer-${index}`);
        assert.ok(marker);assert.equal(marker.position.x,line.current.x);assert.equal(marker.position.y,line.current.y);assert.equal(marker.position.z,line.current.z);
      });
      release();assert.equal(scene.children.length,0);assert.equal(registry.stats.resources,0);
      const points=pointCloudPlotter.create({THREE,scene,resources,requestRender(){}},{configuration:{}},rows);
      const object=scene.children[0],positions=object.geometry.getAttribute("position");
      assert.equal(positions.count,rows.length);
      assert.ok(Array.from(positions.array).every(Number.isFinite));
      assert.equal(points.describeSelection({object,index:0}).id,rows[0].id);
      points.dispose();assert.equal(scene.children.length,0);assert.equal(registry.stats.resources,0);
    }
    for(const id of ["minkowski","lorentz"])for(const frame of ["lab","moving"]){
      const scene=new THREE.Scene(),rows=buildSceneData(id,{time:.25,beta:.6,observerFrame:frame});
      const release=app.drawSceneOverlays({THREE,scene,resources},id,rows,colors,{time:.25,beta:.6,observerFrame:frame});
      const expected=selectedWorldlineEvents(id,.6,.25,frame);
      for(let i=0;i<2;i++){
        const marker=scene.getObjectByName(`${id}-selected-event-${i}`);
        assert.ok(marker);assert.ok(Math.abs(marker.position.x-expected[i].x)<1e-12);assert.ok(Math.abs(marker.position.y-expected[i].y)<1e-12);
      }
      release();assert.equal(registry.stats.resources,0);
    }
    const isolatedScene=new THREE.Scene();let rejectRender=false;
    const initialRows=buildSceneData("minkowski",{time:.25,beta:.2,observerFrame:"moving"});
    const isolated=app.makeVisualizationPlotter().create({THREE,scene:isolatedScene,resources,renderer:null,callbacks:{onError(){}},requestRender(){if(rejectRender)throw new Error("render failed");}},{configuration:{}},{sceneId:"minkowski",time:.25,beta:.2,observerFrame:"moving",rows:initialRows});
    const polished=isolatedScene.getObjectByName("Einstein polished reference marker"),oldPoints=isolatedScene.children.find(child=>child.isPoints),oldGeometry=oldPoints.geometry;
    assert.ok(Math.abs(polished.rotation.y-Math.PI/2)<1e-12);
    rejectRender=true;assert.throws(()=>isolated.setTime(.5),/render failed/);
    assert.equal(oldPoints.geometry,oldGeometry);assert.ok(Math.abs(polished.rotation.y-Math.PI/2)<1e-12);
    rejectRender=false;isolated.setTime(.75);
    const expectedMoving=selectedWorldlineEvents("minkowski",.2,.75,"moving")[1],movingMarker=isolatedScene.getObjectByName("minkowski-selected-event-1");
    assert.ok(Math.abs(movingMarker.position.x-expectedMoving.x)<1e-12);assert.ok(Math.abs(movingMarker.position.y-expectedMoving.y)<1e-12);
    const priorSlider=byId("time-slider").value;rejectRender=true;assert.throws(()=>isolated.updateFrame({elapsedSeconds:4}),/render failed/);
    assert.ok(Math.abs(polished.rotation.y-1.5*Math.PI)<1e-12);assert.equal(byId("time-slider").value,priorSlider);
    isolated.dispose();resources.dispose();
    registry.disposeAll();
    windowStub.emit("beforeunload");
  } finally {
    for(const [name,descriptor] of previous)descriptor?Object.defineProperty(globalThis,name,descriptor):delete globalThis[name];
    if(runtimeDescriptor)Object.defineProperty(globalThis,"EINSTEIN_THREE_RUNTIME",runtimeDescriptor);else delete globalThis.EINSTEIN_THREE_RUNTIME;
  }
});

test("application mounts local Three.js and OrbitControls with a simulated renderer",async()=>{
  const html=await readFile(new URL("./index.html",import.meta.url),"utf8"),elements=new Map();
  for(const match of html.matchAll(/\bid="([^"]+)"/g))elements.set(match[1],new Element());
  const byId=id=>{const element=elements.get(id);assert.ok(element,`missing element ${id}`);return element;};
  byId("speed-slider").value="0.6";byId("time-slider").value="0";byId("sample-number").value="1";
  byId("observer-frame").options=[new Element("option"),new Element("option")];
  const windowStub=new Element(),locationStub={hash:""},documentStub=Object.assign(new Element(),{getElementById:byId,createElement:tag=>new Element(tag),defaultView:windowStub});
  byId("canvas-host").ownerDocument=documentStub;windowStub.devicePixelRatio=1;
  const vendor=await readFile(new URL("./vendor/three-r128.min.js",import.meta.url),"utf8"),sandbox={console,document:documentStub,window:windowStub};
  runInNewContext(vendor,sandbox,{filename:"local-three-r128.min.js"});
  const controlsScript=await readFile(new URL("./vendor/OrbitControls-r128.js",import.meta.url),"utf8");
  runInNewContext(controlsScript,sandbox,{filename:"local-orbit-controls-r128.js"});
  let renderer,pickIndex=0,failNextGeometry=false;
  class FakeRenderer {
    constructor(){renderer=this;this.domElement=new Element("canvas");this.domElement.ownerDocument=documentStub;this.domElement.clientWidth=640;this.domElement.clientHeight=480;this.renders=0;this.disposed=false;}
    setPixelRatio(){} setSize(){} setAnimationLoop(){}
    render(scene,camera){const points=scene.children.find(child=>child.isPoints);if(this.rejectPointX!==undefined&&points&&Math.abs(points.geometry.getAttribute("position").array[300]-this.rejectPointX)<1e-5)throw new Error("simulated renderer failure");this.scene=scene;this.camera=camera;this.renders++;}
    dispose(){this.disposed=true;}
  }
  const runtime={THREE:{...sandbox.THREE,WebGLRenderer:FakeRenderer,PMREMGenerator:undefined},OrbitControls:sandbox.THREE.OrbitControls};
  runtime.THREE.SphereGeometry=class extends sandbox.THREE.SphereGeometry {constructor(...args){if(failNextGeometry){failNextGeometry=false;throw new Error("simulated geometry failure");}super(...args);}};
  runtime.THREE.Raycaster=class extends sandbox.THREE.Raycaster {intersectObjects(objects){const points=objects.find(object=>object.isPoints);return points?[{object:points,index:pickIndex}]:[];}};
  const replacements={document:documentStub,window:windowStub,location:locationStub,history:{replaceState(_a,_b,url){locationStub.hash=url;}},matchMedia:()=>({matches:false,addEventListener(){}}),EINSTEIN_THREE_RUNTIME:runtime};
  const previous=new Map(Object.keys(replacements).map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  try {
    for(const [name,value] of Object.entries(replacements))Object.defineProperty(globalThis,name,{configurable:true,writable:true,value});
    await import("./visualizations.js?mounted-integration");
    assert.match(byId("scene-status").textContent,/3D scene ready/);
    assert.equal(byId("renderer-fallback").hidden,true);
    assert.equal(byId("canvas-host").children.length,1);
    assert.ok(renderer.scene.getObjectByName("minkowski-selected-event-1"));
    const canvas=renderer.domElement,original=renderer.camera.position.clone();
    canvas.emit("pointerdown",{pointerType:"mouse",pointerId:1,button:0,clientX:100,clientY:100});
    canvas.emit("pointermove",{pointerType:"mouse",pointerId:1,clientX:140,clientY:115});
    documentStub.emit("pointermove",{pointerType:"mouse",pointerId:1,clientX:140,clientY:115});
    canvas.emit("pointerup",{pointerType:"mouse",pointerId:1,clientX:140,clientY:115});
    documentStub.emit("pointerup",{pointerType:"mouse",pointerId:1,clientX:140,clientY:115});
    assert.ok(renderer.camera.position.distanceTo(original)>0);
    assert.match(byId("selection-readout").textContent,/sample inspector/);
    canvas.emit("pointerdown",{pointerType:"mouse",pointerId:2,button:0,clientX:100,clientY:100});
    documentStub.emit("pointermove",{pointerType:"mouse",pointerId:2,clientX:180,clientY:100});
    canvas.emit("pointerup",{pointerType:"mouse",pointerId:2,clientX:100,clientY:100});
    documentStub.emit("pointerup",{pointerType:"mouse",pointerId:2,clientX:100,clientY:100});
    assert.match(byId("selection-readout").textContent,/sample inspector/);
    byId("play-button").emit("click");assert.equal(byId("play-button").getAttribute("aria-pressed"),"true");
    canvas.emit("pointerdown",{pointerType:"mouse",pointerId:3,button:0,clientX:100,clientY:100});
    canvas.emit("pointerup",{pointerType:"mouse",pointerId:3,clientX:100,clientY:100});
    assert.equal(byId("play-button").getAttribute("aria-pressed"),"false");
    assert.match(byId("selection-readout").textContent,/Selected sample 1 of/);
    pickIndex=100;failNextGeometry=true;byId("speed-slider").value="0.9";byId("speed-slider").emit("input");
    await new Promise(resolve=>setTimeout(resolve,0));assert.match(byId("scene-status").textContent,/Scene update failed/);
    canvas.emit("pointerdown",{pointerType:"mouse",pointerId:4,button:0,clientX:100,clientY:100});
    canvas.emit("pointerup",{pointerType:"mouse",pointerId:4,clientX:100,clientY:100});
    assert.match(byId("selection-readout").textContent,/displayed point does not match the current sample data/);
    assert.match(byId("selection-readout").textContent,/\(-1\.200, -2\.000, 0\.000\)/);
    byId("speed-slider").value="0.6";byId("speed-slider").emit("input");pickIndex=0;
    await new Promise(resolve=>setTimeout(resolve,0));assert.match(byId("scene-status").textContent,/3D scene ready/);
    pickIndex=100;renderer.rejectPointX=-1.8;byId("speed-slider").value="0.9";byId("speed-slider").emit("input");
    canvas.emit("pointerdown",{pointerType:"mouse",pointerId:5,button:0,clientX:100,clientY:100});
    canvas.emit("pointerup",{pointerType:"mouse",pointerId:5,clientX:100,clientY:100});
    assert.match(byId("selection-readout").textContent,/displayed point does not match the current sample data/);
    assert.match(byId("selection-readout").textContent,/\(-1\.200, -2\.000, 0\.000\)/);
    renderer.rejectPointX=undefined;byId("speed-slider").value="0.6";byId("speed-slider").emit("input");pickIndex=0;
    byId("time-slider").value="0.25";byId("time-slider").emit("input");
    const polished=renderer.scene.getObjectByName("Einstein polished reference marker");
    assert.ok(Math.abs(polished.rotation.y-Math.PI/2)<1e-12);
    renderer.rejectPointX=-1.2;byId("time-slider").value="0.5";byId("time-slider").emit("input");
    assert.ok(Math.abs(polished.rotation.y-Math.PI/2)<1e-12);
    renderer.rejectPointX=undefined;
    const before=renderer.camera.position.clone();
    byId("scene").emit("keydown",{key:"ArrowRight"});
    assert.ok(renderer.camera.position.distanceTo(before)>0);
    byId("time-slider").value="0.5";byId("time-slider").emit("input");
    const marker=renderer.scene.getObjectByName("minkowski-selected-event-1");
    assert.ok(Math.abs(marker.position.x)<1e-12&&Math.abs(marker.position.y)<1e-12);
    byId("observer-frame").value="moving";byId("observer-frame").emit("change");
    byId("speed-slider").value="0.8";byId("speed-slider").emit("input");
    byId("play-button").emit("click");assert.equal(byId("play-button").getAttribute("aria-pressed"),"true");
    byId("reset-button").emit("click");
    assert.equal(byId("play-button").getAttribute("aria-pressed"),"false");
    assert.equal(Number(byId("time-slider").value),0);assert.equal(Number(byId("speed-slider").value),.6);assert.equal(byId("observer-frame").value,"lab");
    assert.ok(renderer.camera.position.distanceTo(original)<1e-12);
    const resetMarker=renderer.scene.getObjectByName("minkowski-selected-event-1"),expectedReset=selectedWorldlineEvents("minkowski",.6,0,"lab")[1];
    assert.ok(Math.abs(resetMarker.position.x-expectedReset.x)<1e-12&&Math.abs(resetMarker.position.y-expectedReset.y)<1e-12);
    byId("scene-list").querySelectorAll("button").find(button=>button.dataset.scene==="particle").emit("click");
    assert.ok(renderer.scene.getObjectByName("particle-selected-endpoint-0"));
    for(const model of scenes){
      byId("scene-list").querySelectorAll("button").find(button=>button.dataset.scene===model.id).emit("click");
      byId("time-slider").value="0.5";byId("time-slider").emit("input");
      const points=renderer.scene.children.find(child=>child.isPoints);
      const expected=buildSceneData(model.id,{time:.5,beta:.6,observerFrame:"lab"});
      assert.equal(points.geometry.getAttribute("position").count,expected.length,model.id);
      assert.ok(renderer.scene.children.length<1000,`${model.id} has bounded scene objects`);
      const objectCount=renderer.scene.children.length;
      byId("time-slider").emit("input");
      assert.equal(renderer.scene.children.length,objectCount,`${model.id} repeated updates release old objects`);
    }
    assert.ok(renderer.renders>1);
    windowStub.emit("beforeunload");
    assert.equal(renderer.disposed,true);assert.equal(byId("canvas-host").children.length,0);
    assert.equal(canvas.listeners.pointerdown.length,0);assert.equal(documentStub.listeners.pointermove.length,0);
  } finally {
    for(const [name,descriptor] of previous)descriptor?Object.defineProperty(globalThis,name,descriptor):delete globalThis[name];
  }
});
