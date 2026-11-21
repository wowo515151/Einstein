export function createPlaybackController({
  reducedMotion=()=>false,
  available=()=>true,
  onFrame=()=>{},
  onChange=()=>{},
  onError=error=>{throw error;},
  now=()=>performance.now(),
  schedule=(callback,delay)=>setInterval(callback,delay),
  cancel=handle=>clearInterval(handle),
  durationMs=8000,
  intervalMs=100,
}={}){
  if(typeof reducedMotion!=="function"||typeof available!=="function"||typeof onFrame!=="function"||typeof onChange!=="function"||typeof onError!=="function"||typeof now!=="function"||typeof schedule!=="function"||typeof cancel!=="function")throw new TypeError("playback dependencies must be functions");
  if(!Number.isFinite(durationMs)||durationMs<=0||!Number.isFinite(intervalMs)||intervalMs<=0)throw new RangeError("playback timing must be positive and finite");
  let timer,startedAt=0,position=0,startPosition=0,disposed=false,playing=false,generation=0;
  const stop=()=>{
    generation++;
    if(timer!==undefined){cancel(timer);timer=undefined;}
    if(playing){playing=false;onChange(false);}
    return "stopped";
  };
  const play=()=>{
    if(disposed)return "disposed";
    if(playing)return "already-playing";
    if(reducedMotion())return "reduced-motion";
    if(!available())return "unavailable";
    const run=++generation;
    startedAt=now();startPosition=position;playing=true;
    try{onChange(true);}catch(error){if(run===generation)playing=false;throw error;}
    if(run!==generation||!playing||disposed)return "stopped";
    let handle;
    try{handle=schedule(()=>{
      if(run!==generation||!playing)return;
      if(disposed)return stop();
      if(reducedMotion())return stop();
      if(!available())return stop();
      const elapsed=now()-startedAt;
      if(!Number.isFinite(elapsed)||elapsed<0)return stop();
      position=(startPosition+elapsed/durationMs)%1;
      try{onFrame(position);}catch(error){if(run===generation)stop();onError(error);}
    },intervalMs);}catch(error){if(run===generation&&playing){playing=false;onChange(false);}throw error;}
    if(run!==generation||!playing||disposed){cancel(handle);return "stopped";}
    timer=handle;
    return "started";
  };
  const toggle=()=>playing?stop():play();
  const seek=value=>{if(!Number.isFinite(value)||value<0||value>1)throw new RangeError("position must be between 0 and 1");position=value;if(playing){startedAt=now();startPosition=value;}return position;};
  const dispose=()=>{if(disposed)return;disposed=true;stop();};
  return Object.freeze({play,stop,toggle,seek,dispose,get playing(){return playing;},get disposed(){return disposed;}});
}
