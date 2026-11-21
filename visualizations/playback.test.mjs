import test from "node:test";
import assert from "node:assert/strict";
import {createPlaybackController} from "./playback.js";

function fakeClock(options={}){
  let time=0,tick,scheduledDelay,cancelled=0;const frames=[],changes=[];
  const controller=createPlaybackController({
    ...options,
    now:()=>time,
    schedule(callback,delay){tick=callback;scheduledDelay=delay;return 17;},
    cancel(handle){assert.equal(handle,17);cancelled++;},
    onFrame(value){frames.push(value);options.onFrame?.(value);},
    onChange(value){changes.push(value);options.onChange?.(value);},
  });
  return {controller,frames,changes,setTime(value){time=value;},tick(){tick?.();},get scheduledDelay(){return scheduledDelay;},get cancelled(){return cancelled;}};
}

test("playback starts, advances cyclic time, toggles off, and disposes its timer",()=>{
  const clock=fakeClock();
  assert.equal(clock.controller.play(),"started");assert.equal(clock.controller.playing,true);assert.deepEqual(clock.changes,[true]);assert.equal(clock.scheduledDelay,100);
  clock.setTime(2000);clock.tick();assert.deepEqual(clock.frames,[.25]);
  clock.setTime(9000);clock.tick();assert.deepEqual(clock.frames,[.25,.125]);
  assert.equal(clock.controller.toggle(),"stopped");assert.equal(clock.controller.playing,false);assert.equal(clock.cancelled,1);assert.deepEqual(clock.changes,[true,false]);
  assert.equal(clock.controller.play(),"started");clock.controller.dispose();assert.equal(clock.controller.disposed,true);assert.equal(clock.controller.playing,false);assert.equal(clock.cancelled,2);
  assert.equal(clock.controller.play(),"disposed");clock.controller.dispose();assert.equal(clock.cancelled,2);
});

test("playback respects reduced motion and renderer availability",()=>{
  let reduce=true,available=true;const clock=fakeClock({reducedMotion:()=>reduce,available:()=>available});
  assert.equal(clock.controller.play(),"reduced-motion");assert.equal(clock.scheduledDelay,undefined);
  reduce=false;available=false;assert.equal(clock.controller.play(),"unavailable");assert.equal(clock.scheduledDelay,undefined);
  available=true;assert.equal(clock.controller.play(),"started");reduce=true;clock.tick();assert.equal(clock.controller.playing,false);assert.equal(clock.cancelled,1);assert.deepEqual(clock.changes,[true,false]);
});

test("playback stops on lost availability and frame errors",()=>{
  let available=true;const unavailable=fakeClock({available:()=>available});
  unavailable.controller.play();available=false;unavailable.tick();
  assert.equal(unavailable.controller.playing,false);assert.equal(unavailable.cancelled,1);assert.deepEqual(unavailable.changes,[true,false]);
  const failure=new Error("frame failed"),errors=[];
  const broken=fakeClock({onFrame:()=>{throw failure;},onError:error=>errors.push(error)});
  broken.controller.play();broken.tick();
  assert.equal(broken.controller.playing,false);assert.equal(broken.cancelled,1);assert.deepEqual(broken.changes,[true,false]);assert.deepEqual(errors,[failure]);
});

test("playback resumes from the selected position and honors a seek while playing",()=>{
  const clock=fakeClock();
  clock.controller.seek(.5);clock.controller.play();
  clock.setTime(1000);clock.tick();assert.equal(clock.frames.at(-1),.625);
  clock.controller.seek(.2);clock.setTime(1800);clock.tick();assert.ok(Math.abs(clock.frames.at(-1)-.3)<1e-12);
  clock.controller.stop();clock.setTime(3000);clock.controller.play();clock.setTime(3800);clock.tick();assert.ok(Math.abs(clock.frames.at(-1)-.4)<1e-12);
  assert.throws(()=>clock.controller.seek(1.1),/position/);
});

test("playback handles immediate stops and scheduler failures",()=>{
  let controller,scheduled=0;
  controller=createPlaybackController({onChange:playing=>{if(playing)controller.stop();},schedule:()=>{scheduled++;return 17;}});
  assert.equal(controller.play(),"stopped");assert.equal(controller.playing,false);assert.equal(scheduled,0);
  const changes=[],failure=createPlaybackController({onChange:playing=>changes.push(playing),schedule:()=>{throw new Error("scheduler failed");}});
  assert.throws(()=>failure.play(),/scheduler failed/);assert.equal(failure.playing,false);assert.deepEqual(changes,[true,false]);
  let cancelled=0,synchronous;
  synchronous=createPlaybackController({now:()=>0,onFrame:()=>synchronous.stop(),schedule:callback=>{callback();return 42;},cancel:handle=>{assert.equal(handle,42);cancelled++;}});
  assert.equal(synchronous.play(),"stopped");assert.equal(synchronous.playing,false);assert.equal(cancelled,1);
});
test("queued callbacks from stopped playback cannot advance a later run",()=>{
  let time=0;const callbacks=[],frames=[],cancelled=[];
  const controller=createPlaybackController({now:()=>time,schedule:callback=>{callbacks.push(callback);return callbacks.length;},cancel:handle=>cancelled.push(handle),onFrame:value=>frames.push(value)});
  controller.play();controller.stop();time=1000;callbacks[0]();
  assert.deepEqual(frames,[]);
  controller.play();time=2000;callbacks[0]();
  assert.deepEqual(frames,[]);
  callbacks[1]();assert.deepEqual(frames,[.125]);
  controller.dispose();time=3000;callbacks[1]();
  assert.deepEqual(frames,[.125]);assert.deepEqual(cancelled,[1,2]);
});
test("a reentrant restart does not schedule the superseded run",()=>{
  let controller,restarted=false;const callbacks=[];
  controller=createPlaybackController({now:()=>0,onChange:playing=>{if(playing&&!restarted){restarted=true;controller.stop();controller.play();}},schedule:callback=>{callbacks.push(callback);return callbacks.length;},cancel:()=>{}});
  assert.equal(controller.play(),"stopped");assert.equal(controller.playing,true);assert.equal(callbacks.length,1);
  controller.dispose();
});
test("dispose prevents restart from a stop notification",()=>{
  let controller,restart;const callbacks=[],cancelled=[];
  controller=createPlaybackController({now:()=>0,onChange:playing=>{if(!playing)restart=controller.play();},schedule:callback=>{callbacks.push(callback);return callbacks.length;},cancel:handle=>cancelled.push(handle)});
  controller.play();controller.dispose();
  assert.equal(restart,"disposed");assert.equal(controller.disposed,true);assert.equal(controller.playing,false);
  assert.equal(callbacks.length,1);assert.deepEqual(cancelled,[1]);
});
test("playback rejects invalid scheduler configuration",()=>{
  assert.throws(()=>createPlaybackController({durationMs:0}),/positive and finite/);
  assert.throws(()=>createPlaybackController({intervalMs:Infinity}),/positive and finite/);
  assert.throws(()=>createPlaybackController({schedule:1}),/dependencies must be functions/);
});
