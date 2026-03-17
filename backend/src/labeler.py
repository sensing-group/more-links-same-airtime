#!/usr/bin/env python3
"""
Real-time web-based labeling interface for WiFi CSI data collection.

Serves a browser UI on port 8080 that lets users label occupancy
states in real time. Label events are written to JSONL files alongside CSI data
for later merge via merge_asof in the preprocessing pipeline.
"""

import asyncio
import json
import logging
import os
import weakref
from datetime import datetime
from pathlib import Path

import websockets

logger = logging.getLogger(__name__)

LABEL_PORT = int(os.environ.get("LABEL_PORT", "80"))
DATA_DIR = os.environ.get("CSI_DATA_DIR", "./csi_data")

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>CSI Labeler</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#1a1a2e;color:#e0e0e0;
  height:100vh;display:flex;flex-direction:column;overflow:hidden;touch-action:manipulation}
.status-bar{display:flex;align-items:center;gap:10px;padding:8px 12px;background:#16213e;
  font-size:13px;flex-shrink:0;border-bottom:1px solid #0f3460}
.dot{width:10px;height:10px;border-radius:50%;background:#4ecca3;flex-shrink:0}
.dot.disconnected{background:#666;animation:none}
.dot.connected{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.status-bar .label-now{font-weight:bold;color:#4ecca3;margin-left:auto}
.status-bar .elapsed{color:#aaa;font-size:12px}
.status-bar .count{color:#aaa;font-size:12px}
.tabs{display:flex;flex-shrink:0;background:#16213e}
.tabs button{flex:1;padding:10px 0;border:none;background:transparent;color:#888;font-size:14px;
  cursor:pointer;border-bottom:2px solid transparent}
.tabs button.active{color:#4ecca3;border-bottom-color:#4ecca3}
.tab-content{flex:1;display:flex;flex-direction:column;overflow:hidden}
.tab-panel{display:none;flex:1;flex-direction:column;padding:12px;overflow-y:auto}
.tab-panel.active{display:flex}

/* Toggle */
#toggle-btn{flex:1;border:none;border-radius:12px;font-size:32px;font-weight:bold;cursor:pointer;
  transition:background .2s,color .2s;min-height:64px}
#toggle-btn.occupied{background:#1b4332;color:#4ecca3}
#toggle-btn.empty{background:#2a2a3e;color:#666}

/* Hold */
#hold-btn{flex:1;border:none;border-radius:12px;font-size:28px;font-weight:bold;cursor:pointer;
  background:#2a2a3e;color:#666;min-height:64px;transition:background .15s,color .15s;
  user-select:none;-webkit-user-select:none}
#hold-btn.active{background:#1b4332;color:#4ecca3}

/* Timer */
.timer-controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.timer-controls button,.timer-controls input{height:44px;border-radius:8px;border:1px solid #333;
  background:#2a2a3e;color:#e0e0e0;padding:0 12px;font-size:14px;cursor:pointer;min-width:56px}
.timer-controls button.active{border-color:#4ecca3;color:#4ecca3}
.timer-controls input{width:80px}
.timer-labels{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.timer-labels input{height:36px;border-radius:6px;border:1px solid #333;background:#2a2a3e;
  color:#e0e0e0;padding:0 8px;font-size:13px;width:100px}
.timer-labels button{height:36px;border-radius:6px;border:1px solid #333;background:#2a2a3e;
  color:#e0e0e0;padding:0 12px;font-size:18px;cursor:pointer}
.timer-display{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px}
.timer-display .current-label{font-size:36px;font-weight:bold;color:#4ecca3}
.timer-display .next-label{font-size:16px;color:#888}
.timer-ring{position:relative;width:120px;height:120px}
.timer-ring svg{transform:rotate(-90deg)}
.timer-ring circle{fill:none;stroke-width:6}
.timer-ring .bg{stroke:#2a2a3e}
.timer-ring .fg{stroke:#4ecca3;stroke-linecap:round;transition:stroke-dashoffset .3s linear}
.timer-ring .time-text{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;font-size:20px;font-weight:bold}
#timer-start{width:100%;height:56px;border:none;border-radius:10px;background:#4ecca3;color:#1a1a2e;
  font-size:18px;font-weight:bold;cursor:pointer;margin-top:12px}
#timer-start.running{background:#e74c3c}

/* Manual */
.manual-presets{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.manual-presets button{height:56px;min-width:90px;border-radius:10px;border:1px solid #333;
  background:#2a2a3e;color:#e0e0e0;font-size:15px;cursor:pointer;padding:0 14px;flex:1}
.manual-presets button.active{background:#1b4332;color:#4ecca3;border-color:#4ecca3}
.manual-presets button.edit-btn{flex:0;min-width:44px;font-size:18px}
.manual-free{display:flex;gap:8px;margin-bottom:12px}
.manual-free input{flex:1;height:50px;border-radius:10px;border:1px solid #333;background:#2a2a3e;
  color:#e0e0e0;padding:0 14px;font-size:15px}
.manual-free button{height:50px;min-width:80px;border-radius:10px;border:none;background:#4ecca3;
  color:#1a1a2e;font-size:15px;font-weight:bold;cursor:pointer}
.edit-panel{display:none;margin-bottom:8px;gap:6px;flex-wrap:wrap}
.edit-panel.show{display:flex}
.edit-panel input{height:36px;border-radius:6px;border:1px solid #333;background:#2a2a3e;
  color:#e0e0e0;padding:0 8px;font-size:13px;flex:1;min-width:120px}
.edit-panel button{height:36px;border-radius:6px;border:1px solid #333;background:#2a2a3e;
  color:#e0e0e0;padding:0 12px;font-size:13px;cursor:pointer}
</style>
</head>
<body>

<div class="status-bar">
  <div class="dot disconnected" id="ws-dot"></div>
  <span id="status-label">--</span>
  <span class="elapsed" id="status-elapsed">0s ago</span>
  <span class="count" id="status-count">0 events</span>
  <span class="label-now" id="status-current">--</span>
</div>

<div class="tabs">
  <button class="active" data-tab="toggle">Toggle</button>
  <button data-tab="hold">Hold</button>
  <button data-tab="timer">Timer</button>
  <button data-tab="manual">Manual</button>
</div>

<div class="tab-content">

  <!-- Toggle Tab -->
  <div class="tab-panel active" id="tab-toggle">
    <button id="toggle-btn" class="empty">EMPTY</button>
  </div>

  <!-- Hold Tab -->
  <div class="tab-panel" id="tab-hold">
    <button id="hold-btn">HOLD = OCCUPIED</button>
  </div>

  <!-- Timer Tab -->
  <div class="tab-panel" id="tab-timer">
    <div class="timer-controls">
      <button data-interval="5" class="active">5s</button>
      <button data-interval="15">15s</button>
      <button data-interval="30">30s</button>
      <button data-interval="60">60s</button>
      <input type="number" id="custom-interval" placeholder="sec" min="1">
    </div>
    <div class="timer-labels" id="timer-label-list"></div>
    <div class="timer-display">
      <div class="current-label" id="timer-current">--</div>
      <div class="timer-ring">
        <svg viewBox="0 0 120 120" width="120" height="120">
          <circle class="bg" cx="60" cy="60" r="54"/>
          <circle class="fg" id="timer-arc" cx="60" cy="60" r="54"
            stroke-dasharray="339.292" stroke-dashoffset="0"/>
        </svg>
        <div class="time-text" id="timer-time">0</div>
      </div>
      <div class="next-label" id="timer-next">Next: --</div>
    </div>
    <button id="timer-start">START</button>
  </div>

  <!-- Manual Tab -->
  <div class="tab-panel" id="tab-manual">
    <div class="manual-presets" id="manual-presets"></div>
    <div class="edit-panel" id="edit-panel">
      <input type="text" id="edit-input" placeholder="New preset name">
      <button id="edit-add">Add</button>
      <button id="edit-done">Done</button>
    </div>
    <div class="manual-free">
      <input type="text" id="free-input" placeholder="Custom label...">
      <button id="free-send">Log</button>
    </div>
  </div>

</div>

<script>
(function(){
"use strict";

// --- State ---
let ws, currentLabel="--", eventCount=0, lastEventTime=null, reconnectTimer=null;
let toggleState="empty";
let timerRunning=false, timerInterval=5, timerLabels=["empty","occupied"], timerIdx=0, timerHandle=null, timerStart=null;
let manualPresets=JSON.parse(localStorage.getItem("csi_presets")||'null')||["empty","occupied","sedentary","moving","sleeping","unknown"];
let activeManual=null;

// --- WebSocket ---
function connect(){
  const proto=location.protocol==="https:"?"wss:":"ws:";
  ws=new WebSocket(proto+"//"+location.host+"/ws");
  ws.onopen=()=>{document.getElementById("ws-dot").className="dot connected"};
  ws.onclose=()=>{document.getElementById("ws-dot").className="dot disconnected";
    reconnectTimer=setTimeout(connect,2000)};
  ws.onerror=()=>{};
  ws.onmessage=(e)=>{try{const d=JSON.parse(e.data);if(d.ack)eventCount=d.count}catch(ex){}};
}
connect();

function send(label, mode, note){
  currentLabel=label;
  lastEventTime=Date.now();
  eventCount++;
  updateStatus();
  if(ws&&ws.readyState===1){
    ws.send(JSON.stringify({label,mode,note:note||""}));
  }
}

// --- Status bar ---
function updateStatus(){
  document.getElementById("status-current").textContent=currentLabel;
  document.getElementById("status-count").textContent=eventCount+" events";
}
setInterval(()=>{
  const el=document.getElementById("status-elapsed");
  if(lastEventTime){
    const s=Math.floor((Date.now()-lastEventTime)/1000);
    el.textContent=s<60?s+"s ago":Math.floor(s/60)+"m "+s%60+"s ago";
  }
},1000);

// --- Tabs ---
document.querySelector(".tabs").addEventListener("click",(e)=>{
  const btn=e.target.closest("button");if(!btn)return;
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.remove("active"));
  btn.classList.add("active");
  document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));
  document.getElementById("tab-"+btn.dataset.tab).classList.add("active");
});

// --- Toggle ---
const toggleBtn=document.getElementById("toggle-btn");
toggleBtn.addEventListener("click",()=>{
  toggleState=toggleState==="empty"?"occupied":"empty";
  toggleBtn.className=toggleState;
  toggleBtn.textContent=toggleState.toUpperCase();
  send(toggleState,"toggle");
});

// --- Hold ---
const holdBtn=document.getElementById("hold-btn");
function holdDown(e){e.preventDefault();holdBtn.classList.add("active");send("occupied","hold")}
function holdUp(e){e.preventDefault();holdBtn.classList.remove("active");send("empty","hold")}
holdBtn.addEventListener("mousedown",holdDown);
holdBtn.addEventListener("mouseup",holdUp);
holdBtn.addEventListener("mouseleave",holdUp);
holdBtn.addEventListener("touchstart",holdDown,{passive:false});
holdBtn.addEventListener("touchend",holdUp,{passive:false});
holdBtn.addEventListener("touchcancel",holdUp,{passive:false});

// --- Timer ---
const timerArc=document.getElementById("timer-arc");
const CIRCUM=2*Math.PI*54;
timerArc.setAttribute("stroke-dasharray",CIRCUM);

document.querySelector(".timer-controls").addEventListener("click",(e)=>{
  const btn=e.target.closest("button[data-interval]");if(!btn)return;
  document.querySelectorAll(".timer-controls button").forEach(b=>b.classList.remove("active"));
  btn.classList.add("active");
  timerInterval=parseInt(btn.dataset.interval);
  document.getElementById("custom-interval").value="";
});
document.getElementById("custom-interval").addEventListener("change",(e)=>{
  const v=parseInt(e.target.value);if(v>0){timerInterval=v;
    document.querySelectorAll(".timer-controls button").forEach(b=>b.classList.remove("active"));}
});

function renderTimerLabels(){
  const c=document.getElementById("timer-label-list");
  c.innerHTML="";
  timerLabels.forEach((l,i)=>{
    const inp=document.createElement("input");inp.value=l;
    inp.addEventListener("change",(e)=>{timerLabels[i]=e.target.value});
    c.appendChild(inp);
  });
  const addBtn=document.createElement("button");addBtn.textContent="+";
  addBtn.addEventListener("click",()=>{timerLabels.push("label");renderTimerLabels()});
  c.appendChild(addBtn);
  const rmBtn=document.createElement("button");rmBtn.textContent="-";
  rmBtn.addEventListener("click",()=>{if(timerLabels.length>1){timerLabels.pop();renderTimerLabels()}});
  c.appendChild(rmBtn);
}
renderTimerLabels();

const timerStartBtn=document.getElementById("timer-start");
timerStartBtn.addEventListener("click",()=>{
  if(timerRunning){stopTimer();return}
  timerRunning=true;timerIdx=0;timerStartBtn.textContent="STOP";timerStartBtn.classList.add("running");
  send(timerLabels[0],"timer");
  document.getElementById("timer-current").textContent=timerLabels[0];
  document.getElementById("timer-next").textContent="Next: "+timerLabels[(1)%timerLabels.length];
  timerStart=Date.now();
  timerTick();
});
function stopTimer(){
  timerRunning=false;timerStartBtn.textContent="START";timerStartBtn.classList.remove("running");
  if(timerHandle)cancelAnimationFrame(timerHandle);
  timerArc.style.strokeDashoffset=0;
  document.getElementById("timer-time").textContent="0";
}
function timerTick(){
  if(!timerRunning)return;
  const elapsed=(Date.now()-timerStart)/1000;
  const remaining=timerInterval-elapsed%timerInterval;
  const frac=1-(remaining/timerInterval);
  timerArc.style.strokeDashoffset=CIRCUM*(1-frac);
  document.getElementById("timer-time").textContent=Math.ceil(remaining);
  const cycle=Math.floor(elapsed/timerInterval);
  if(cycle!==timerIdx){
    timerIdx=cycle;
    const li=timerIdx%timerLabels.length;
    const label=timerLabels[li];
    send(label,"timer");
    document.getElementById("timer-current").textContent=label;
    document.getElementById("timer-next").textContent="Next: "+timerLabels[(li+1)%timerLabels.length];
  }
  timerHandle=requestAnimationFrame(timerTick);
}

// --- Manual ---
function renderPresets(){
  const c=document.getElementById("manual-presets");
  c.innerHTML="";
  manualPresets.forEach(p=>{
    const b=document.createElement("button");b.textContent=p;
    if(p===activeManual)b.classList.add("active");
    b.addEventListener("click",()=>{activeManual=p;send(p,"manual");renderPresets()});
    c.appendChild(b);
  });
  const eb=document.createElement("button");eb.classList.add("edit-btn");eb.innerHTML="&#9998;";
  eb.addEventListener("click",()=>{
    document.getElementById("edit-panel").classList.toggle("show");
  });
  c.appendChild(eb);
}
renderPresets();

document.getElementById("edit-add").addEventListener("click",()=>{
  const v=document.getElementById("edit-input").value.trim();
  if(v&&!manualPresets.includes(v)){manualPresets.push(v);localStorage.setItem("csi_presets",JSON.stringify(manualPresets));renderPresets()}
  document.getElementById("edit-input").value="";
});
document.getElementById("edit-done").addEventListener("click",()=>{
  document.getElementById("edit-panel").classList.remove("show");
});
document.getElementById("free-send").addEventListener("click",()=>{
  const v=document.getElementById("free-input").value.trim();
  if(v){send(v,"manual");document.getElementById("free-input").value=""}
});
document.getElementById("free-input").addEventListener("keydown",(e)=>{
  if(e.key==="Enter"){document.getElementById("free-send").click()}
});

})();
</script>
</body>
</html>"""


class LabelingServer:
    """Asyncio-compatible labeling server with WebSocket-based browser UI."""

    def __init__(self, port: int = LABEL_PORT, data_dir: str = DATA_DIR):
        self._port = port
        self._data_dir = Path(data_dir)
        self._server = None
        self._http_server = None
        self._clients: set = set()
        self._event_count = 0
        self._label_file = None
        self._label_file_date = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the HTTP and WebSocket servers."""
        # Start HTTP server for the HTML page
        http_server = await asyncio.start_server(
            self._handle_http, "0.0.0.0", self._port
        )
        self._http_server = http_server

        # Start WebSocket server on port+1
        ws_port = self._port + 1
        self._server = await websockets.serve(
            self._handle_ws, "0.0.0.0", ws_port
        )

        logger.info(f"Labeling UI available at http://0.0.0.0:{self._port}")

    async def stop(self) -> None:
        """Stop the servers and close the label file."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._http_server:
            self._http_server.close()
            await self._http_server.wait_closed()
        async with self._lock:
            if self._label_file:
                self._label_file.close()
                self._label_file = None
        logger.info("Labeling server stopped")

    async def _handle_http(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Serve the HTML page and upgrade info."""
        try:
            request = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        except Exception:
            writer.close()
            return

        request_str = request.decode("utf-8", errors="ignore")

        # Inject the actual WebSocket port into the HTML
        ws_port = self._port + 1
        page = HTML_PAGE.replace(
            'location.host+"/ws"',
            f'location.hostname+":{ws_port}/ws"'
        )
        body = page.encode("utf-8")

        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/html; charset=utf-8\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            + body
        )
        writer.write(response)
        await writer.drain()
        writer.close()

    async def _handle_ws(self, websocket):
        """Handle a WebSocket client connection."""
        self._clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._log_event(data)
                    self._event_count += 1
                    await websocket.send(json.dumps({
                        "ack": True, "count": self._event_count
                    }))
                except json.JSONDecodeError:
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)

    async def _log_event(self, data: dict) -> None:
        """Append a label event to the current JSONL file."""
        now = datetime.now()
        ts = now.timestamp()
        iso = now.isoformat(timespec="milliseconds")
        event = {
            "ts": round(ts, 3),
            "iso": iso,
            "label": data.get("label", "unknown"),
            "mode": data.get("mode", "manual"),
            "note": data.get("note", ""),
        }
        line = json.dumps(event, separators=(",", ":")) + "\n"

        async with self._lock:
            today = now.strftime("%Y-%m-%d")

            # Rotate file at midnight
            if self._label_file_date != today:
                if self._label_file:
                    self._label_file.close()
                date_dir = self._data_dir / today
                date_dir.mkdir(parents=True, exist_ok=True)
                path = date_dir / "labels.jsonl"
                self._label_file = open(path, "a")
                self._label_file_date = today
                logger.info(f"Label file: {path}")

            self._label_file.write(line)
            self._label_file.flush()