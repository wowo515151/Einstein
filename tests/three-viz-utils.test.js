import assert from "node:assert/strict";
import test from "node:test";
import { generateRainbowColors } from "../src/three-viz/utils/colors.js";
import { generateColorShades } from "../src/three-viz/utils/shades.js";
import { createCarPaintMaterial, retainCarPaintEnvironment } from "../src/three-viz/plotters/shared/materials.js";

test("ThreeViz palette helpers return bounded, deterministic color sequences",()=>{
  const rainbow=generateRainbowColors(8),shades=generateColorShades(0x336699,5);
  assert.equal(rainbow.length,8);assert.equal(new Set(rainbow).size,8);
  assert.equal(shades.length,5);assert.equal(new Set(shades).size,5);
  assert.ok([...rainbow,...shades].every(color=>Number.isSafeInteger(color)&&color>=0&&color<=0xffffff));
  assert.deepEqual(generateRainbowColors(0),[]);assert.deepEqual(generateColorShades(0x336699,0),[]);
  assert.throws(()=>generateRainbowColors(-1),/count/);
  assert.throws(()=>generateColorShades(0x336699,1.5),/count/);
  assert.throws(()=>generateColorShades(0x1000000,2),/color/);
});

test("ThreeViz car-paint materials keep the glossy defaults and caller overrides",()=>{
  class PhysicalMaterial { constructor(options){this.options=options;} }
  class StandardMaterial { constructor(options){this.options=options;} }
  const material=createCarPaintMaterial({MeshPhysicalMaterial:PhysicalMaterial,MeshStandardMaterial:StandardMaterial},0x123456,{roughness:.3,clearcoat:.7});
  assert.ok(material instanceof PhysicalMaterial);
  assert.equal(material.options.color,0x123456);assert.equal(material.options.metalness,1);
  assert.equal(material.options.roughness,.3);assert.equal(material.options.clearcoat,.7);
  const fallback=createCarPaintMaterial({MeshStandardMaterial:StandardMaterial},0xabcdef);
  assert.ok(fallback instanceof StandardMaterial);assert.equal(fallback.options.metalness,1);
});

test("ThreeViz reflection environment is shared and released after the last consumer",()=>{
  let fromSceneCalls=0,targetDisposals=0,generatorDisposals=0,geometryDisposals=0,materialDisposals=0;
  const texture={};
  const target={texture,dispose(){targetDisposals+=1;}};
  class FakeScene {
    constructor(){this.isScene=true;this.parent=null;this.children=[];}
    add(...items){this.children.push(...items);}
  }
  class FakeGeometry { dispose(){geometryDisposals+=1;} }
  class FakeMaterial { constructor(options){this.options=options;} dispose(){materialDisposals+=1;} }
  class FakeMesh {
    constructor(geometry,material){this.geometry=geometry;this.material=material;this.position={set(){}};this.rotation={set(){}};}
  }
  class FakePMREMGenerator {
    constructor(renderer){assert.equal(renderer,renderTarget);}
    compileEquirectangularShader(){}
    fromScene(){fromSceneCalls+=1;return target;}
    dispose(){generatorDisposals+=1;}
  }
  const THREE={Scene:FakeScene,PlaneGeometry:FakeGeometry,MeshBasicMaterial:FakeMaterial,Mesh:FakeMesh,PMREMGenerator:FakePMREMGenerator,DoubleSide:2};
  const scene=new FakeScene(),renderTarget={},releaseFirst=retainCarPaintEnvironment(THREE,scene,renderTarget),releaseSecond=retainCarPaintEnvironment(THREE,scene,renderTarget);
  assert.equal(fromSceneCalls,1);assert.equal(scene.environment,texture);assert.equal(scene.children.length,0);
  assert.equal(geometryDisposals,4);assert.equal(materialDisposals,1);assert.equal(generatorDisposals,1);
  releaseFirst();releaseFirst();assert.equal(targetDisposals,0);assert.equal(scene.environment,texture);
  releaseSecond();assert.equal(targetDisposals,1);assert.equal(scene.environment,null);
});
