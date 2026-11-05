"""Five-camera browser viewer for WayPOP images, GT, and predictions."""

from __future__ import annotations

import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from .data import CAMERAS, Sample, WaypopRelease, relative_path
from .labels import ground_truth_mask, load_ids, prediction_mask


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WayPOP viewer</title><style>
:root{font-family:system-ui,sans-serif;color:#e9eef4;background:#151b22}body{margin:0;padding:18px}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:18px}h1{font-size:20px;margin:0 22px 0 0}
select,button,input{font:inherit}select,button{background:#263241;color:#f1f5f9;border:1px solid #5a6a7e;border-radius:6px;padding:6px 10px}
button{cursor:pointer}input[type=range]{accent-color:#5bbfcf}#frame{min-width:220px;flex:1}#position{min-width:115px}
.row{margin:0 0 22px}.row h2{font-size:15px;font-weight:600;margin:0 0 8px;color:#aee8ee}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.camera{min-width:0}.camera h3{font-size:12px;margin:0 0 5px;text-align:center;color:#b8c6d5}
.stack{position:relative;background:#080c10;aspect-ratio:3/2}.stack img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.overlay{opacity:.5}#status{font-size:13px;color:#b8c6d5}#error{color:#ff9e9e}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body>
<header><h1>WayPOP viewer</h1><label>Sequence <select id="sequence"></select></label>
<button id="previous">Previous</button><button id="play">Play</button><button id="next">Next</button>
<label>FPS <select id="fps"><option value="0.5">0.5</option><option value="1" selected>1</option><option value="2">2</option><option value="5">5</option></select></label>
<input id="frame" type="range" min="0" value="0"><span id="position"></span>
<label>Overlay <input id="alpha" type="range" min="0" max="100" value="50"><span id="alphaValue">50%</span></label></header>
<div id="status"></div><div id="error"></div><main id="content"></main>
<script>
const cameras=['SIDE_LEFT','FRONT_LEFT','FRONT','FRONT_RIGHT','SIDE_RIGHT'];
const q=id=>document.getElementById(id);
let state=null,activeLoad=null,displayedIndex=null,displayedSequence=null;
let displayedStatus='',currentUrls=[],playing=false,playTimer=null,playSession=0;

function cancelLoad(){
  if(activeLoad){activeLoad.abort();activeLoad=null;}
}

function stopPlayback(){
  playing=false;
  playSession++;
  if(playTimer!==null){clearTimeout(playTimer);playTimer=null;}
  q('play').textContent='Play';
}

function applyAlpha(){
  const value=Number(q('alpha').value)/100;
  document.querySelectorAll('.overlay').forEach(node=>node.style.opacity=value);
  q('alphaValue').textContent=q('alpha').value+'%';
}

function makeRow(title,items,kind,urls,decodes){
  const row=document.createElement('section');row.className='row';
  const heading=document.createElement('h2');heading.textContent=title;row.append(heading);
  const grid=document.createElement('div');grid.className='grid';
  for(const camera of cameras){
    const sample=items[camera];
    const card=document.createElement('div');card.className='camera';
    const label=document.createElement('h3');label.textContent=camera.replace('_',' ');card.append(label);
    const stack=document.createElement('div');stack.className='stack';
    for(const [path,alt,overlay] of [[sample.image,camera+' RGB',false],[sample[kind],camera+' '+kind,true]]){
      if(!path)continue;
      const img=document.createElement('img');img.src=urls.get(path);img.alt=alt;
      if(overlay)img.className='overlay';
      stack.append(img);
      decodes.push(img.decode().catch(()=>{throw new Error('Could not decode '+path);}));
    }
    card.append(stack);grid.append(card);
  }
  row.append(grid);return row;
}

async function render(index){
  if(!state)return false;
  cancelLoad();
  const controller=new AbortController();activeLoad=controller;
  const sequence=q('sequence').value;
  const isCurrent=()=>activeLoad===controller&&!controller.signal.aborted;
  const pendingUrls=[];
  let committed=false;
  q('status').textContent='Loading frame '+(index+1)+' / '+(Number(q('frame').max)+1)+'…';
  q('error').textContent='';
  try{
    const frameUrl='/api/frame?sequence='+encodeURIComponent(sequence)+'&index='+index;
    const response=await fetch(frameUrl,{signal:controller.signal});
    if(!response.ok)throw new Error('Frame request failed: HTTP '+response.status);
    const payload=await response.json();
    if(!isCurrent())return false;
    const paths=new Set();
    for(const camera of cameras){
      const sample=payload.cameras[camera];
      paths.add(sample.image);
      if(payload.has_gt&&sample.gt)paths.add(sample.gt);
      if(payload.has_predictions&&sample.pred)paths.add(sample.pred);
    }
    const entries=[...paths];
    const downloads=await Promise.allSettled(entries.map(async path=>{
      const asset=await fetch(path,{signal:controller.signal});
      if(!asset.ok)throw new Error('Image request failed: HTTP '+asset.status+' ('+path+')');
      const url=URL.createObjectURL(await asset.blob());
      pendingUrls.push(url);
      return [path,url];
    }));
    const failure=downloads.find(result=>result.status==='rejected');
    if(failure)throw failure.reason;
    if(!isCurrent())return false;
    const urls=new Map(downloads.map(result=>result.value));
    const decodes=[],rows=[];
    if(payload.has_gt)rows.push(makeRow('Ground truth',payload.cameras,'gt',urls,decodes));
    if(payload.has_predictions)rows.push(makeRow('Prediction',payload.cameras,'pred',urls,decodes));
    const decoded=await Promise.allSettled(decodes);
    const decodeFailure=decoded.find(result=>result.status==='rejected');
    if(decodeFailure)throw decodeFailure.reason;
    if(!isCurrent())return false;
    q('content').replaceChildren(...rows);
    const oldUrls=currentUrls;currentUrls=pendingUrls;
    oldUrls.forEach(url=>URL.revokeObjectURL(url));
    displayedIndex=index;displayedSequence=sequence;
    q('frame').value=index;
    q('position').textContent=(index+1)+' / '+payload.count;
    displayedStatus='Split: '+payload.split+' · Timestamp: '+payload.timestamp_micros+' · Frame index: '+payload.frame_index;
    q('status').textContent=displayedStatus;
    applyAlpha();
    committed=true;
    return true;
  }catch(error){
    if(isCurrent()){
      q('status').textContent=displayedStatus;
      q('error').textContent=error.message||String(error);
    }
    return false;
  }finally{
    if(!committed)pendingUrls.forEach(url=>URL.revokeObjectURL(url));
    if(activeLoad===controller)activeLoad=null;
  }
}

function selectSequence(){
  stopPlayback();
  const found=state.sequences.find(item=>item.id===q('sequence').value);
  q('frame').max=found.frames-1;q('frame').value=0;
  render(0);
}

function step(direction){
  stopPlayback();
  const slider=q('frame');
  const index=Math.max(0,Math.min(Number(slider.max),Number(slider.value)+direction));
  slider.value=index;
  render(index);
}

function scheduleNext(delay,session){
  if(!playing||session!==playSession)return;
  if(displayedIndex>=Number(q('frame').max)){stopPlayback();return;}
  playTimer=setTimeout(async()=>{
    playTimer=null;
    const start=performance.now();
    const loaded=await render(displayedIndex+1);
    if(!playing||session!==playSession)return;
    if(!loaded){stopPlayback();return;}
    const interval=1000/Number(q('fps').value);
    scheduleNext(Math.max(0,interval-(performance.now()-start)),session);
  },delay);
}

async function startPlayback(){
  playing=true;q('play').textContent='Pause';
  const session=++playSession;
  const target=Number(q('frame').value);
  if(activeLoad||displayedSequence!==q('sequence').value||displayedIndex!==target){
    const loaded=await render(target);
    if(!playing||session!==playSession)return;
    if(!loaded){stopPlayback();return;}
  }
  scheduleNext(1000/Number(q('fps').value),session);
}

q('sequence').addEventListener('change',selectSequence);
q('frame').addEventListener('input',()=>{stopPlayback();render(Number(q('frame').value));});
q('alpha').addEventListener('input',applyAlpha);
q('fps').addEventListener('change',()=>{
  if(playing&&playTimer!==null){clearTimeout(playTimer);playTimer=null;scheduleNext(1000/Number(q('fps').value),playSession);}
});
q('previous').addEventListener('click',()=>step(-1));
q('next').addEventListener('click',()=>step(1));
q('play').addEventListener('click',()=>{
  if(playing){
    stopPlayback();cancelLoad();
    if(displayedSequence===q('sequence').value&&displayedIndex!==null)q('frame').value=displayedIndex;
    q('status').textContent=displayedStatus;
  }else startPlayback();
});
fetch('/api/sequences').then(response=>{
  if(!response.ok)throw new Error('Sequence request failed: HTTP '+response.status);
  return response.json();
}).then(payload=>{
  state=payload;
  for(const item of payload.sequences){
    const option=document.createElement('option');option.value=item.id;option.textContent=item.id;q('sequence').append(option);
  }
  selectSequence();
}).catch(error=>q('error').textContent=error.message||String(error));
</script></body></html>"""


class Viewer:
    def __init__(self, data_root: Path | str, split: str, prediction_root: Path | str | None = None):
        self.release = WaypopRelease(data_root)
        self.split = split
        self.samples = self.release.samples(split)
        self.sequences = self.release.sequence_ids(split)
        self.prediction_root = Path(prediction_root).expanduser().resolve() if prediction_root else None
        if split != "val" and self.prediction_root is None:
            raise ValueError("Test visualization requires --pred-root; test GT is withheld")
        self.predictions: dict[str, list[dict]] = {}
        if self.prediction_root is not None:
            manifest = self.prediction_root / split / "predictions.json"
            with manifest.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for annotation in payload["annotations"]:
                file_name = annotation["file_name"]
                relative_path(file_name)
                if file_name in self.predictions:
                    raise ValueError(f"Duplicate prediction annotation: {file_name}")
                self.predictions[file_name] = annotation["segments_info"]
        self.categories = self.release.category_map
        if 29 in self.categories:
            self.categories[255] = self.categories[29]

    def _safe_path(self, path: Path, root: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
            raise FileNotFoundError(path)
        return resolved

    def frame(self, sequence_id: str, index: int) -> dict:
        if sequence_id not in self.sequences:
            raise ValueError(f"Unknown sequence: {sequence_id}")
        frames = self.release.frames(self.split, sequence_id)
        if index < 0 or index >= len(frames):
            raise ValueError(f"Frame index must be in [0, {len(frames) - 1}]")
        frame = frames[index]
        first = frame["FRONT"]
        cameras = {}
        for name in CAMERAS:
            sample = frame[name]
            urls = {"image": f"/api/image?index={sample.index}"}
            if self.split == "val":
                urls["gt"] = f"/api/overlay?kind=gt&index={sample.index}"
            if sample.prediction_file_name in self.predictions:
                urls["pred"] = f"/api/overlay?kind=pred&index={sample.index}"
            cameras[name] = urls
        return {
            "split": self.split,
            "count": len(frames),
            "timestamp_micros": first.row["timestamp_micros"],
            "frame_index": first.row["frame_index"],
            "has_gt": self.split == "val",
            "has_predictions": self.prediction_root is not None,
            "cameras": cameras,
        }

    def image(self, index: int) -> bytes:
        sample = self.samples[index]
        path = self._safe_path(sample.path("image_path"), self.release.root / self.split)
        return path.read_bytes()

    def overlay(self, index: int, kind: str) -> bytes:
        sample = self.samples[index]
        if kind == "gt":
            if self.split != "val":
                raise ValueError("Test ground truth is withheld")
            path = self._safe_path(sample.path("panoptic_path"), self.release.root / self.split)
            ids = load_ids(path)
            mask = ground_truth_mask(
                ids, int(sample.row["panoptic_label_divisor"]), sample.sequence_id, self.categories
            )
        elif kind == "pred":
            if self.prediction_root is None or sample.prediction_file_name not in self.predictions:
                raise FileNotFoundError(f"No prediction for {sample.sample_id}")
            path = self._safe_path(
                self.prediction_root / self.split / "predictions" / sample.prediction_file_name,
                self.prediction_root / self.split,
            )
            ids = load_ids(path)
            mask = prediction_mask(
                ids, sample.sequence_id, self.predictions[sample.prediction_file_name], self.categories
            )
        else:
            raise ValueError(f"Unknown overlay kind: {kind}")
        if ids.shape != (sample.size[1], sample.size[0]):
            raise ValueError(f"Mask dimensions do not match {sample.sample_id}")
        buffer = io.BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        return buffer.getvalue()


class _ViewerServer(ThreadingHTTPServer):
    viewer: Viewer


class _Handler(BaseHTTPRequestHandler):
    server: _ViewerServer

    def _send(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        viewer = self.server.viewer
        try:
            if parsed.path == "/":
                self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/sequences":
                payload = {
                    "split": viewer.split,
                    "sequences": [
                        {"id": sequence, "frames": len(viewer.release.frames(viewer.split, sequence))}
                        for sequence in viewer.sequences
                    ],
                }
                self._send(json.dumps(payload).encode(), "application/json")
            elif parsed.path == "/api/frame":
                payload = viewer.frame(query["sequence"][0], int(query["index"][0]))
                self._send(json.dumps(payload).encode(), "application/json")
            elif parsed.path == "/api/image":
                self._send(viewer.image(int(query["index"][0])), "image/png")
            elif parsed.path == "/api/overlay":
                self._send(viewer.overlay(int(query["index"][0]), query["kind"][0]), "image/png")
            else:
                self.send_error(404)
        except (KeyError, ValueError, IndexError, FileNotFoundError) as error:
            self.send_error(400, str(error))


def serve(
    data_root: Path | str,
    split: str = "val",
    *,
    prediction_root: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = False,
) -> None:
    viewer = Viewer(data_root, split, prediction_root)
    server = _ViewerServer((host, port), _Handler)
    server.viewer = viewer
    url = f"http://{host}:{server.server_port}/"
    print(f"WayPOP viewer: {url} ({split}, {len(viewer.sequences)} sequences)")
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
