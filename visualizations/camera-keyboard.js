/** Apply one keyboard orbit or zoom step to the viewer's current camera. */
export function moveCameraWithKey(viewer, key) {
  if (!viewer || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "_"].includes(key)) return false;
  const pose = viewer.getCameraPose();
  const [x,y,z] = pose.position.map((value,index) => value - pose.target[index]);
  let radius = Math.hypot(x,y,z);
  if (!(radius > 0) || !Number.isFinite(radius)) return false;
  let azimuth = Math.atan2(x,z), polar = Math.acos(Math.max(-1,Math.min(1,y/radius)));
  if (key === "ArrowLeft") azimuth -= Math.PI/18;
  if (key === "ArrowRight") azimuth += Math.PI/18;
  if (key === "ArrowUp") polar = Math.max(.05,polar-Math.PI/18);
  if (key === "ArrowDown") polar = Math.min(Math.PI-.05,polar+Math.PI/18);
  if (key === "+" || key === "=") radius = Math.max(2.5,radius/1.15);
  if (key === "-" || key === "_") radius = Math.min(15,radius*1.15);
  const position = [radius*Math.sin(polar)*Math.sin(azimuth),radius*Math.cos(polar),radius*Math.sin(polar)*Math.cos(azimuth)].map((value,index)=>value+pose.target[index]);
  viewer.setCameraPose({ ...pose, position });
  return true;
}
