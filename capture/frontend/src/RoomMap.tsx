import { useCallback, useEffect, useRef, useState } from 'react';
import { useStore } from './store';
import type { LinkBaseline, RoomFeature } from './types';

const MARGIN = { top: 20, right: 20, bottom: 50, left: 50 };
const NODE_RADIUS = 14;
const GRID_STEP_M = 0.5;
const SCALE_PX_PER_M = 80;
const TICK_LEN = 5;
const S = SCALE_PX_PER_M; // shorthand for meter-to-px

const FONT = {
  tick: 12,
  axisLabel: 14,
  nodeLabel: 12,
  linkLabel: 10,
  legend: 12,
  scaleBar: 11,
};

function calColor(count: number, target: number): string {
  if (count >= target) return '#22c55e';
  if (count >= target * 0.5) return '#f59e0b';
  if (count > 0) return '#ef4444';
  return '#999';
}

function calPattern(count: number, target: number): string {
  if (count >= target) return 'none';
  if (count >= target * 0.5) return '6,3';
  if (count > 0) return '3,3';
  return '2,2';
}

// --- Feature defaults ---
const FEATURE_DEFAULTS: Record<string, Partial<RoomFeature>> = {
  door:   { width: 0.8, rotation: 0, flip: false },
  window: { width: 1.0, rotation: 0 },
  table:  { width: 1.2, height: 0.6, rotation: 0 },
  wall:   { width: 0.15, height: 1.5, rotation: 0 },
};

let featureCounter = 0;
function newFeatureId(type: string) {
  return `${type}-${++featureCounter}`;
}

// --- Feature SVG renderers (all in px space, anchor at 0,0) ---
function renderDoor(f: RoomFeature) {
  const w = f.width * S;
  const flip = f.flip ? -1 : 1;
  // Arc from (w,0) to (0, flip*w)
  return (
    <g>
      <line x1={0} y1={0} x2={w} y2={0} stroke="#333" strokeWidth={2} />
      <path
        d={`M ${w} 0 A ${w} ${w} 0 0 ${flip > 0 ? 1 : 0} 0 ${flip * w}`}
        fill="none" stroke="#333" strokeWidth={1.2} strokeDasharray="4,2" />
      <circle cx={0} cy={0} r={2.5} fill="#333" />
    </g>
  );
}

function renderWindow(f: RoomFeature) {
  const w = f.width * S;
  const t = 4; // wall thickness in px
  return (
    <g>
      {/* Wall thickness */}
      <rect x={0} y={-t} width={w} height={t * 2} fill="#fff" stroke="#333" strokeWidth={1.2} />
      {/* Glass pane (center line) */}
      <line x1={2} y1={0} x2={w - 2} y2={0} stroke="#3b9cdb" strokeWidth={2} />
    </g>
  );
}

function renderTable(f: RoomFeature) {
  const w = f.width * S;
  const h = (f.height ?? 0.6) * S;
  return (
    <g>
      <rect x={0} y={0} width={w} height={h}
            fill="#e8e8e8" stroke="#333" strokeWidth={1.5} rx={2} />
      <text x={w / 2} y={h / 2 + 4} textAnchor="middle"
            fontSize={10} fill="#555" fontFamily="serif"
            transform={`scale(1, -1) translate(0, ${-h})`}
            style={{ pointerEvents: 'none' }}>
        Table
      </text>
    </g>
  );
}

function renderWall(f: RoomFeature) {
  const w = f.width * S;
  const h = (f.height ?? 1.5) * S;
  return (
    <rect x={0} y={0} width={w} height={h} fill="#555" stroke="#333" strokeWidth={1} />
  );
}

const RENDERERS: Record<string, (f: RoomFeature) => JSX.Element> = {
  door: renderDoor,
  window: renderWindow,
  table: renderTable,
  wall: renderWall,
};

// Bounding box for hit testing / selection highlight (in px)
function featureBBox(f: RoomFeature): { x: number; y: number; w: number; h: number } {
  const w = f.width * S;
  switch (f.type) {
    case 'door': {
      const swing = f.flip ? -w : w;
      return { x: 0, y: Math.min(0, swing), w, h: Math.abs(swing) };
    }
    case 'window':
      return { x: 0, y: -4, w, h: 8 };
    case 'table':
      return { x: 0, y: 0, w, h: (f.height ?? 0.6) * S };
    case 'wall':
      return { x: 0, y: 0, w, h: (f.height ?? 1.5) * S };
    default:
      return { x: 0, y: 0, w, h: w };
  }
}

async function saveFeatures(features: RoomFeature[]) {
  try {
    await fetch('/api/features', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    });
  } catch { /* ignore */ }
}

export function RoomMap() {
  const { nodes, room, detection, stats, features: storeFeatures, localization, simulation } = useStore();
  const svgRef = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<string | null>(null); // node MAC
  const [selected, setSelected] = useState<Set<string> | null>(null); // node filter

  // Feature state
  const [localFeatures, setLocalFeatures] = useState<RoomFeature[]>([]);
  const [undoStack, setUndoStack] = useState<RoomFeature[][]>([]);
  const [placingType, setPlacingType] = useState<string | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [draggingFeature, setDraggingFeature] = useState<string | null>(null);
  const [pendingDrag, setPendingDrag] = useState<{ id: string; startX: number; startY: number } | null>(null);
  const [dragOffset, setDragOffset] = useState<{ dx: number; dy: number }>({ dx: 0, dy: 0 });
  const [resizingFeature, setResizingFeature] = useState<string | null>(null);
  const preDragSnapshot = useRef<RoomFeature[] | null>(null);
  const DRAG_THRESHOLD = 4; // pixels before drag starts

  // Sync from store when not actively editing
  useEffect(() => {
    if (!draggingFeature && !resizingFeature) {
      setLocalFeatures(storeFeatures);
    }
  }, [storeFeatures, draggingFeature, resizingFeature]);

  const getLabel = (node: { mac: string; label?: string | null }) =>
    node.label ?? node.mac.slice(-5);

  const toggleNode = (mac: string) => {
    setSelected(prev => {
      if (prev === null) return new Set([mac]);
      const next = new Set(prev);
      if (next.has(mac)) {
        next.delete(mac);
        return next.size === 0 ? null : next;
      }
      next.add(mac);
      return next;
    });
  };

  const isVisible = (mac: string) => selected === null || selected.has(mac);
  const linkVisible = (macA: string, macB: string) =>
    selected === null || selected.has(macA) || selected.has(macB);

  const plotW = room.width * S;
  const plotH = room.height * S;
  const svgW = MARGIN.left + plotW + MARGIN.right;
  const svgH = MARGIN.top + plotH + MARGIN.bottom;

  const toSvg = useCallback((x: number, y: number) => ({
    x: MARGIN.left + x * S,
    y: MARGIN.top + (room.height - y) * S,
  }), [room]);

  const fromSvg = useCallback((sx: number, sy: number) => ({
    x: Math.max(0, Math.min(room.width, (sx - MARGIN.left) / S)),
    y: Math.max(0, Math.min(room.height, room.height - (sy - MARGIN.top) / S)),
  }), [room]);

  const getSvgCoords = useCallback((e: React.MouseEvent) => {
    if (!svgRef.current) return { x: 0, y: 0 };
    const rect = svgRef.current.getBoundingClientRect();
    return fromSvg(e.clientX - rect.left, e.clientY - rect.top);
  }, [fromSvg]);

  // --- Node drag ---
  const handleNodeMouseDown = (mac: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(mac);
    setSelectedFeature(null);
  };

  // --- Feature click/drag ---
  const handleFeatureMouseDown = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (placingType) return;
    setSelectedFeature(id);
    preDragSnapshot.current = localFeatures;
    setPendingDrag({ id, startX: e.clientX, startY: e.clientY });
  };

  // --- Resize handle drag ---
  const handleResizeMouseDown = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    preDragSnapshot.current = localFeatures;
    setResizingFeature(id);
  };

  // --- Combined mouse move ---
  const handleMouseMove = useCallback(async (e: React.MouseEvent) => {
    if (!svgRef.current) return;
    const coords = getSvgCoords(e);

    // Promote pending drag to real drag after threshold
    if (pendingDrag && !draggingFeature) {
      const dx = e.clientX - pendingDrag.startX;
      const dy = e.clientY - pendingDrag.startY;
      if (Math.abs(dx) + Math.abs(dy) >= DRAG_THRESHOLD) {
        const feat = localFeatures.find(f => f.id === pendingDrag.id);
        if (feat) {
          setDragOffset({ dx: coords.x - feat.x, dy: coords.y - feat.y });
          setDraggingFeature(pendingDrag.id);
        }
        setPendingDrag(null);
      }
      return;
    }

    if (dragging) {
      const x = Math.round(coords.x * 100) / 100;
      const y = Math.round(coords.y * 100) / 100;
      try {
        await fetch(`/api/nodes/${encodeURIComponent(dragging)}/position`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ x, y }),
        });
      } catch { /* ignore */ }
      return;
    }

    if (draggingFeature) {
      setLocalFeatures(prev => prev.map(f =>
        f.id === draggingFeature
          ? { ...f, x: Math.round((coords.x - dragOffset.dx) * 100) / 100,
                     y: Math.round((coords.y - dragOffset.dy) * 100) / 100 }
          : f
      ));
      return;
    }

    if (resizingFeature) {
      setLocalFeatures(prev => prev.map(f => {
        if (f.id !== resizingFeature) return f;
        const dx = coords.x - f.x;
        const dy = coords.y - f.y;
        // Unrotate delta for rotated features
        const rad = -(f.rotation ?? 0) * Math.PI / 180;
        const udx = dx * Math.cos(rad) - dy * Math.sin(rad);
        const udy = dx * Math.sin(rad) + dy * Math.cos(rad);
        const newW = Math.max(0.1, Math.round(udx * 100) / 100);
        if (f.type === 'door' || f.type === 'window') {
          return { ...f, width: newW };
        }
        const newH = Math.max(0.1, Math.round(udy * 100) / 100);
        return { ...f, width: newW, height: newH };
      }));
      return;
    }
  }, [dragging, draggingFeature, resizingFeature, pendingDrag, getSvgCoords, dragOffset, localFeatures]);

  const handleMouseUp = useCallback(() => {
    if (draggingFeature || resizingFeature) {
      if (preDragSnapshot.current) {
        setUndoStack(prev => [...prev.slice(-19), preDragSnapshot.current!]);
        preDragSnapshot.current = null;
      }
      saveFeatures(localFeatures);
    }
    setDragging(null);
    setDraggingFeature(null);
    setPendingDrag(null);
    setResizingFeature(null);
  }, [draggingFeature, resizingFeature, localFeatures]);

  // --- Place new feature on SVG click ---
  const handleSvgClick = (e: React.MouseEvent) => {
    if (placingType === 'sim') {
      const coords = getSvgCoords(e);
      fetch('/api/simulation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: true, x: coords.x, y: coords.y }),
      }).catch(() => {});
      setPlacingType(null);
      return;
    }
    if (!placingType) {
      // Click on empty space deselects feature — only if the click target is the SVG root
      // or the room boundary rect (not feature hit-area rects)
      const target = e.target as SVGElement;
      const isBackground = target === svgRef.current
        || (target.tagName === 'rect' && !target.closest('[data-feature]'));
      if (isBackground) {
        setSelectedFeature(null);
      }
      return;
    }
    const coords = getSvgCoords(e);
    const defaults = FEATURE_DEFAULTS[placingType] ?? {};
    const newFeature: RoomFeature = {
      id: newFeatureId(placingType),
      type: placingType as RoomFeature['type'],
      x: Math.round(coords.x * 100) / 100,
      y: Math.round(coords.y * 100) / 100,
      width: defaults.width ?? 1,
      height: defaults.height,
      rotation: 0,
      flip: false,
    };
    applyChange([...localFeatures, newFeature]);
    setSelectedFeature(newFeature.id);
    setPlacingType(null);
  };

  // Push current state to undo stack, apply new state
  const applyChange = (updated: RoomFeature[]) => {
    setUndoStack(prev => [...prev.slice(-19), localFeatures]); // keep last 20
    setLocalFeatures(updated);
    saveFeatures(updated);
  };

  const undoFeatures = () => {
    if (undoStack.length === 0) return;
    const prev = undoStack[undoStack.length - 1];
    setUndoStack(s => s.slice(0, -1));
    setLocalFeatures(prev);
    saveFeatures(prev);
    setSelectedFeature(null);
  };

  // --- Rotate selected feature 90 deg ---
  const rotateSelected = () => {
    if (!selectedFeature) return;
    applyChange(localFeatures.map(f =>
      f.id === selectedFeature ? { ...f, rotation: ((f.rotation ?? 0) + 90) % 360 } : f
    ));
  };

  // --- Flip selected door ---
  const flipSelected = () => {
    if (!selectedFeature) return;
    applyChange(localFeatures.map(f =>
      f.id === selectedFeature && f.type === 'door' ? { ...f, flip: !f.flip } : f
    ));
  };

  // --- Delete selected feature ---
  const deleteSelected = () => {
    if (!selectedFeature) return;
    applyChange(localFeatures.filter(f => f.id !== selectedFeature));
    setSelectedFeature(null);
  };

  // Keyboard shortcuts — only when no input/select is focused
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        undoFeatures();
        return;
      }
      if (!selectedFeature) return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelected();
      } else if (e.key === 'r' || e.key === 'R') {
        rotateSelected();
      } else if (e.key === 'f' || e.key === 'F') {
        flipSelected();
      } else if (e.key === 'Escape') {
        setSelectedFeature(null);
        setPlacingType(null);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  });

  // --- Export ---
  const exportSvg = () => {
    if (!svgRef.current) return;
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svgRef.current);
    const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'room_map.svg';
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportPng = () => {
    if (!svgRef.current) return;
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svgRef.current);
    const canvas = document.createElement('canvas');
    const dpr = 4;
    canvas.width = svgW * dpr;
    canvas.height = svgH * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    const img = new Image();
    img.onload = () => {
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, svgW, svgH);
      ctx.drawImage(img, 0, 0, svgW, svgH);
      canvas.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'room_map.png';
        a.click();
        URL.revokeObjectURL(url);
      }, 'image/png');
    };
    img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgStr)));
  };

  // ========== SVG ELEMENTS ==========

  const activeSet = new Set(
    detection.active_link_details.map(l => `${l.tx}-${l.rx}`)
  );

  // --- Grid ---
  const gridLines: JSX.Element[] = [];
  for (let x = 0; x <= room.width; x += GRID_STEP_M) {
    const p = toSvg(x, 0);
    const p2 = toSvg(x, room.height);
    gridLines.push(<line key={`gx-${x}`} x1={p.x} y1={p.y} x2={p2.x} y2={p2.y} stroke="#e5e5e5" strokeWidth={0.5} />);
  }
  for (let y = 0; y <= room.height; y += GRID_STEP_M) {
    const p = toSvg(0, y);
    const p2 = toSvg(room.width, y);
    gridLines.push(<line key={`gy-${y}`} x1={p.x} y1={p.y} x2={p2.x} y2={p2.y} stroke="#e5e5e5" strokeWidth={0.5} />);
  }

  // --- Axis ticks and labels ---
  const axisTicks: JSX.Element[] = [];
  // X-axis ticks at bottom (y=0 in room coords = SVG bottom)
  for (let x = 0; x <= room.width; x += GRID_STEP_M) {
    const p = toSvg(x, 0);
    const showLabel = x === Math.round(x);
    axisTicks.push(
      <line key={`xt-${x}`} x1={p.x} y1={p.y} x2={p.x} y2={p.y + TICK_LEN} stroke="#333" strokeWidth={1} />
    );
    if (showLabel) {
      axisTicks.push(
        <text key={`xl-${x}`} x={p.x} y={p.y + TICK_LEN + FONT.tick + 2}
              textAnchor="middle" fontSize={FONT.tick} fill="#333" fontFamily="serif">
          {x.toFixed(0)}
        </text>
      );
    }
  }
  axisTicks.push(
    <text key="x-label" x={MARGIN.left + plotW / 2} y={MARGIN.top + plotH + 42}
          textAnchor="middle" fontSize={FONT.axisLabel} fill="#333" fontFamily="serif" fontStyle="italic">
      x (m)
    </text>
  );
  for (let y = 0; y <= room.height; y += GRID_STEP_M) {
    const p = toSvg(0, y);
    const showLabel = y === Math.round(y);
    axisTicks.push(
      <line key={`yt-${y}`} x1={p.x} y1={p.y} x2={p.x - TICK_LEN} y2={p.y} stroke="#333" strokeWidth={1} />
    );
    if (showLabel) {
      axisTicks.push(
        <text key={`yl-${y}`} x={p.x - TICK_LEN - 4} y={p.y + FONT.tick * 0.35}
              textAnchor="end" fontSize={FONT.tick} fill="#333" fontFamily="serif">
          {y.toFixed(0)}
        </text>
      );
    }
  }
  axisTicks.push(
    <text key="y-label" x={12} y={MARGIN.top + plotH / 2}
          textAnchor="middle" fontSize={FONT.axisLabel} fill="#333" fontFamily="serif" fontStyle="italic"
          transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}>
      y (m)
    </text>
  );

  const scaleBar = null;

  // --- Origin marker ---
  const origin = toSvg(0, 0);
  const originMarker = (
    <g key="origin">
      <circle cx={origin.x} cy={origin.y} r={3} fill="#333" />
      <text x={origin.x + 6} y={origin.y - 6} fontSize={FONT.tick} fill="#333" fontFamily="serif">(0,0)</text>
    </g>
  );

  // --- Features ---
  const featureElements = localFeatures.map(f => {
    const renderer = RENDERERS[f.type];
    if (!renderer) return null;
    const p = toSvg(f.x, f.y);
    const isSel = selectedFeature === f.id;
    const bbox = featureBBox(f);

    return (
      <g key={f.id} data-feature={f.id}
         transform={`translate(${p.x}, ${p.y}) scale(1, -1) rotate(${f.rotation ?? 0})`}
         style={{ cursor: draggingFeature === f.id ? 'grabbing' : 'grab' }}
         onMouseDown={handleFeatureMouseDown(f.id)}>
        {/* Invisible hit area */}
        <rect x={bbox.x - 4} y={bbox.y - 4} width={bbox.w + 8} height={bbox.h + 8}
              fill="transparent" />
        {renderer(f)}
        {/* Selection highlight */}
        {isSel && (
          <rect x={bbox.x - 3} y={bbox.y - 3} width={bbox.w + 6} height={bbox.h + 6}
                fill="none" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4,2" rx={2} />
        )}
        {/* Resize handle (bottom-right corner) */}
        {isSel && (
          <circle cx={bbox.x + bbox.w} cy={bbox.y + bbox.h} r={4}
                  fill="#3b82f6" stroke="#fff" strokeWidth={1}
                  style={{ cursor: 'nwse-resize' }}
                  onMouseDown={handleResizeMouseDown(f.id)} />
        )}
      </g>
    );
  });

  // --- Links ---
  const positionedNodes = nodes.filter(n => n.position);
  const baselines = stats.link_baselines;
  const target = stats.baseline_target;
  const isCalibrating = stats.calibrating;

  const links: JSX.Element[] = [];
  for (let i = 0; i < positionedNodes.length; i++) {
    for (let j = i + 1; j < positionedNodes.length; j++) {
      const a = positionedNodes[i];
      const b = positionedNodes[j];
      if (!a.position || !b.position) continue;
      if (!linkVisible(a.mac, b.mac)) continue;
      const p1 = toSvg(a.position[0], a.position[1]);
      const p2 = toSvg(b.position[0], b.position[1]);
      const isActive = activeSet.has(`${a.mac}-${b.mac}`) || activeSet.has(`${b.mac}-${a.mac}`);

      if (!isCalibrating) {
        links.push(
          <line key={`link-${a.mac}-${b.mac}`}
                x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
                stroke={isActive ? '#c00' : '#999'}
                strokeWidth={isActive ? 2.5 : 1}
                strokeDasharray={isActive ? 'none' : '6,4'}
                opacity={isActive ? 0.9 : 0.3} />
        );
        continue;
      }

      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;
      const off = 2.5;

      const abKey = `${a.mac}->${b.mac}`;
      const baKey = `${b.mac}->${a.mac}`;
      const abBl: LinkBaseline | undefined = baselines[abKey];
      const baBl: LinkBaseline | undefined = baselines[baKey];
      const abCount = abBl?.count ?? 0;
      const baCount = baBl?.count ?? 0;

      links.push(
        <line key={`cal-ab-${a.mac}-${b.mac}`}
              x1={p1.x + nx * off} y1={p1.y + ny * off}
              x2={p2.x + nx * off} y2={p2.y + ny * off}
              stroke={calColor(abCount, target)}
              strokeWidth={2} opacity={0.8}
              strokeDasharray={calPattern(abCount, target)}
              strokeLinecap="round" />
      );
      links.push(
        <line key={`cal-ba-${a.mac}-${b.mac}`}
              x1={p1.x - nx * off} y1={p1.y - ny * off}
              x2={p2.x - nx * off} y2={p2.y - ny * off}
              stroke={calColor(baCount, target)}
              strokeWidth={2} opacity={0.8}
              strokeDasharray={calPattern(baCount, target)}
              strokeLinecap="round" />
      );

      const mx = (p1.x + p2.x) / 2;
      const my = (p1.y + p2.y) / 2;
      const minCount = Math.min(abCount, baCount);
      if (minCount < target) {
        links.push(
          <text key={`cal-lbl-${a.mac}-${b.mac}`}
                x={mx} y={my - 6}
                textAnchor="middle" fontSize={FONT.linkLabel} fill="#555"
                fontFamily="serif" style={{ pointerEvents: 'none' }}>
            {minCount}
          </text>
        );
      }
    }
  }

  // --- Nodes ---
  const nodeElements = nodes.map(node => {
    const pos = node.position
      ? toSvg(node.position[0], node.position[1])
      : toSvg(room.width / 2, room.height / 2);
    const isOnline = node.status === 'online';
    const hasPosition = !!node.position;
    const vis = isVisible(node.mac);
    const label = getLabel(node);

    return (
      <g key={node.mac} style={{ cursor: 'grab' }} opacity={vis ? 1 : 0.15}
         onMouseDown={handleNodeMouseDown(node.mac)}>
        <circle cx={pos.x} cy={pos.y} r={NODE_RADIUS}
                fill={isOnline ? '#3b82f6' : '#bbb'}
                stroke={hasPosition ? '#1e3a5f' : '#b45309'}
                strokeWidth={2}
                strokeDasharray={hasPosition ? 'none' : '4,2'} />
        <text x={pos.x} y={pos.y + 1}
              textAnchor="middle" dominantBaseline="central"
              fontSize={FONT.nodeLabel} fill="#fff" fontWeight="bold"
              fontFamily="serif" style={{ pointerEvents: 'none' }}>
          {label}
        </text>
        <text x={pos.x} y={pos.y + NODE_RADIUS + 14}
              textAnchor="middle" fontSize={10} fill="#888"
              fontFamily="monospace" style={{ pointerEvents: 'none' }}>
          {node.mac.slice(-5)}
        </text>
      </g>
    );
  });

  // --- Legend entries ---
  type LegendEntry = { key: string; label: string } & (
    | { type: 'line'; color: string; dash: string; sw: number }
    | { type: 'circle'; fill: string }
    | { type: 'custom'; render: () => JSX.Element }
  );
  const legendEntries: LegendEntry[] = [];
  if (!isCalibrating) {
    legendEntries.push({ key: 'active', type: 'line', color: '#c00', dash: 'none', sw: 2.5, label: 'Active link' });
    legendEntries.push({ key: 'inactive', type: 'line', color: '#999', dash: '6,4', sw: 1, label: 'Inactive link' });
  } else {
    legendEntries.push({ key: 'cal-done', type: 'line', color: '#22c55e', dash: 'none', sw: 2, label: 'Calibrated' });
    legendEntries.push({ key: 'cal-partial', type: 'line', color: '#f59e0b', dash: '6,3', sw: 2, label: 'Partial' });
    legendEntries.push({ key: 'cal-low', type: 'line', color: '#ef4444', dash: '3,3', sw: 2, label: 'Low' });
    legendEntries.push({ key: 'cal-none', type: 'line', color: '#999', dash: '2,2', sw: 1, label: 'No data' });
  }
  legendEntries.push({ key: 'node-on', type: 'circle', fill: '#3b82f6', label: 'Online' });
  legendEntries.push({ key: 'node-off', type: 'circle', fill: '#bbb', label: 'Offline' });
  if (localization) {
    legendEntries.push({ key: 'loc-est', type: 'custom', label: 'Location est.',
      render: () => <svg width={16} height={16}><circle cx={8} cy={8} r={7} fill="rgba(239,68,68,0.15)" stroke="#ef4444" strokeWidth={1} strokeDasharray="2,1" /><circle cx={8} cy={8} r={2.5} fill="#ef4444" /></svg> });
  }
  if (simulation) {
    legendEntries.push({ key: 'sim-occ', type: 'custom', label: 'Sim occupant',
      render: () => <svg width={16} height={16}><line x1={3} y1={8} x2={13} y2={8} stroke="#22c55e" strokeWidth={1.5} /><line x1={8} y1={3} x2={8} y2={13} stroke="#22c55e" strokeWidth={1.5} /><circle cx={8} cy={8} r={5} fill="none" stroke="#22c55e" strokeWidth={1} /></svg> });
  }

  // Add feature type legends if any features exist
  const featureTypes = new Set(localFeatures.map(f => f.type));
  if (featureTypes.has('door')) {
    legendEntries.push({ key: 'feat-door', type: 'custom', label: 'Door',
      render: () => <svg width={20} height={16}><path d="M 2 14 L 2 2 A 12 12 0 0 1 14 14" fill="none" stroke="#333" strokeWidth={1.2} strokeDasharray="3,1.5" /><circle cx={2} cy={14} r={1.5} fill="#333" /></svg> });
  }
  if (featureTypes.has('window')) {
    legendEntries.push({ key: 'feat-window', type: 'custom', label: 'Window',
      render: () => <svg width={20} height={10}><rect x={0} y={1} width={20} height={8} fill="#fff" stroke="#333" strokeWidth={1} /><line x1={2} y1={5} x2={18} y2={5} stroke="#3b9cdb" strokeWidth={2} /></svg> });
  }
  if (featureTypes.has('table')) {
    legendEntries.push({ key: 'feat-table', type: 'custom', label: 'Table',
      render: () => <svg width={20} height={12}><rect x={0} y={0} width={20} height={12} fill="#e8e8e8" stroke="#333" strokeWidth={1} rx={1} /></svg> });
  }
  if (featureTypes.has('wall')) {
    legendEntries.push({ key: 'feat-wall', type: 'custom', label: 'Wall',
      render: () => <svg width={6} height={14}><rect x={0} y={0} width={6} height={14} fill="#555" stroke="#333" strokeWidth={0.8} /></svg> });
  }

  // --- Toolbar styles ---
  const btnStyle: React.CSSProperties = {
    padding: '3px 10px',
    fontSize: 11,
    border: '1px solid #ccc',
    borderRadius: 4,
    background: '#f9f9f9',
    cursor: 'pointer',
    color: '#333',
  };
  const toolBtnStyle = (type: string): React.CSSProperties => ({
    ...btnStyle,
    borderColor: placingType === type ? '#3b82f6' : '#ccc',
    background: placingType === type ? '#eff6ff' : '#f9f9f9',
    color: placingType === type ? '#3b82f6' : '#333',
  });
  const pillStyle = (mac: string): React.CSSProperties => {
    const active = selected !== null && selected.has(mac);
    return {
      padding: '2px 8px', fontSize: 11, fontFamily: 'monospace',
      border: '1px solid', borderColor: active ? '#3b82f6' : '#ddd',
      borderRadius: 12, background: active ? '#3b82f6' : '#fff',
      color: active ? '#fff' : '#666', cursor: 'pointer', userSelect: 'none',
    };
  };

  const selFeat = localFeatures.find(f => f.id === selectedFeature);

  return (
    <div>
      {/* Top bar: filter + export */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 4, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: '#999', marginRight: 4 }}>Filter:</span>
        {nodes.map(n => (
          <span key={n.mac} style={pillStyle(n.mac)} onClick={() => toggleNode(n.mac)}>
            {getLabel(n)} ({n.mac.slice(-5)})
          </span>
        ))}
        {selected !== null && (
          <span style={{ fontSize: 11, color: '#3b82f6', cursor: 'pointer', marginLeft: 4 }}
                onClick={() => setSelected(null)}>
            show all
          </span>
        )}
      </div>

      {/* Feature toolbar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 4, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, color: '#999', marginRight: 4 }}>Add:</span>
        <button style={toolBtnStyle('door')} onClick={() => setPlacingType(placingType === 'door' ? null : 'door')}>Door</button>
        <button style={toolBtnStyle('window')} onClick={() => setPlacingType(placingType === 'window' ? null : 'window')}>Window</button>
        <button style={toolBtnStyle('table')} onClick={() => setPlacingType(placingType === 'table' ? null : 'table')}>Table</button>
        <button style={toolBtnStyle('wall')} onClick={() => setPlacingType(placingType === 'wall' ? null : 'wall')}>Wall</button>
        {undoStack.length > 0 && (
          <button style={btnStyle} onClick={undoFeatures} title="Undo (Ctrl+Z)">Undo</button>
        )}
        {placingType && (
          <span style={{ fontSize: 11, color: '#f59e0b' }}>Click on map to place {placingType}</span>
        )}
        {selFeat && (
          <>
            <span style={{ flex: 1 }} />
            <span style={{ fontSize: 11, color: '#666' }}>{selFeat.type}</span>
            {(['x', 'y', 'width', ...(selFeat.type === 'table' || selFeat.type === 'wall' ? ['height'] : []), 'rotation'] as const).map(field => (
              <label key={field} style={{ display: 'inline-flex', alignItems: 'center', gap: 2, fontSize: 11, color: '#666' }}>
                {field}:
                <input
                  type="number"
                  step={field === 'rotation' ? 90 : 0.01}
                  value={(selFeat as any)[field] ?? ''}
                  style={{ width: field === 'rotation' ? 40 : 52, fontSize: 11, padding: '2px 4px', border: '1px solid #ccc', borderRadius: 3 }}
                  onChange={e => {
                    const val = parseFloat(e.target.value);
                    if (isNaN(val)) return;
                    applyChange(localFeatures.map(f =>
                      f.id === selectedFeature ? { ...f, [field]: val } : f
                    ));
                  }}
                />
              </label>
            ))}
            <button style={btnStyle} onClick={rotateSelected} title="Rotate 90° (R)">R</button>
            {selFeat.type === 'door' && (
              <button style={btnStyle} onClick={flipSelected} title="Flip swing direction (F)">F</button>
            )}
            <button style={{ ...btnStyle, color: '#c00', borderColor: '#c00' }} onClick={deleteSelected} title="Delete (Del)">Del</button>
          </>
        )}
      </div>

      {/* Simulation & localization toolbar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 4, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, color: '#999', marginRight: 4 }}>Loc:</span>
        <button style={toolBtnStyle('sim')}
                onClick={() => setPlacingType(placingType === 'sim' ? null : 'sim')}>
          Place occupant
        </button>
        {simulation && (
          <>
            <span style={{ fontSize: 11, color: '#22c55e' }}>
              sim ({simulation.x.toFixed(2)}, {simulation.y.toFixed(2)})
            </span>
            <button style={{ ...btnStyle, color: '#c00', borderColor: '#fca5a5' }}
                    onClick={() => {
                      fetch('/api/simulation', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ enabled: false }),
                      }).catch(() => {});
                    }}>
              Clear
            </button>
          </>
        )}
        {placingType === 'sim' && (
          <span style={{ fontSize: 11, color: '#f59e0b' }}>Click on map to place virtual occupant</span>
        )}
        {localization && (
          <span style={{ fontSize: 11, color: '#ef4444', marginLeft: 8 }}>
            est ({localization.x.toFixed(2)}, {localization.y.toFixed(2)})
            r={localization.radius.toFixed(2)}m [{localization.n_intersections} pts]
          </span>
        )}
      </div>

      {/* SVG Map */}
      <svg ref={svgRef} width={svgW} height={svgH}
           style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 4,
                    cursor: placingType ? 'crosshair' : undefined }}
           onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}
           onClick={handleSvgClick}>
        {gridLines}
        {axisTicks}
        {scaleBar}
        {originMarker}
        {featureElements}
        {links}
        {/* Localization estimate */}
        {localization && (() => {
          const p = toSvg(localization.x, localization.y);
          const rPx = Math.max(localization.radius * S, 10);
          return (
            <g>
              <circle cx={p.x} cy={p.y} r={rPx}
                      fill="rgba(239, 68, 68, 0.12)" stroke="#ef4444"
                      strokeWidth={1.5} strokeDasharray="4,2" />
              <circle cx={p.x} cy={p.y} r={4} fill="#ef4444" opacity={0.85} />
            </g>
          );
        })()}
        {/* Simulation marker */}
        {simulation && (() => {
          const p = toSvg(simulation.x, simulation.y);
          return (
            <g>
              <line x1={p.x - 10} y1={p.y} x2={p.x + 10} y2={p.y}
                    stroke="#22c55e" strokeWidth={2} />
              <line x1={p.x} y1={p.y - 10} x2={p.x} y2={p.y + 10}
                    stroke="#22c55e" strokeWidth={2} />
              <circle cx={p.x} cy={p.y} r={8} fill="none"
                      stroke="#22c55e" strokeWidth={1.5} />
            </g>
          );
        })()}
        {nodeElements}
        {/* Room boundary on top so it's not obscured */}
        <rect x={MARGIN.left} y={MARGIN.top}
              width={plotW} height={plotH}
              fill="none" stroke="#333" strokeWidth={1.5}
              style={{ pointerEvents: 'none' }} />
      </svg>

      {/* Legend bar */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center',
        marginTop: 6, padding: '6px 8px',
        border: '1px solid #e5e5e5', borderRadius: 4, background: '#fafafa',
        fontSize: FONT.legend, fontFamily: 'serif', color: '#333',
      }}>
        {legendEntries.map(e => (
          <span key={e.key} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            {e.type === 'line' ? (
              <svg width={24} height={12}>
                <line x1={0} y1={6} x2={24} y2={6}
                      stroke={e.color} strokeWidth={e.sw} strokeDasharray={e.dash} />
              </svg>
            ) : e.type === 'circle' ? (
              <svg width={14} height={14}>
                <circle cx={7} cy={7} r={6} fill={e.fill} stroke="#1e3a5f" strokeWidth={1.5} />
              </svg>
            ) : (
              e.render()
            )}
            {e.label}
          </span>
        ))}
      </div>

      {/* Export buttons */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 4, marginTop: 6 }}>
        <button style={btnStyle} onClick={exportSvg}>Export SVG</button>
        <button style={btnStyle} onClick={exportPng}>Export PNG</button>
      </div>
    </div>
  );
}
