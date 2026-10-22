import { useEffect, useRef } from 'react';
import { useStore } from './store';
import { LiveState } from './types';

const RECONNECT_DELAY_MS = 2000;

export function useWebSocket() {
  const { setConnected, updateFromWS } = useStore();
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let closed = false;

    function connect() {
      if (closed) return;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) setTimeout(connect, RECONNECT_DELAY_MS);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (ev) => {
        try {
          const data: LiveState = JSON.parse(ev.data);
          updateFromWS(data);
        } catch { /* ignore parse errors */ }
      };
    }

    connect();
    return () => { closed = true; wsRef.current?.close(); };
  }, [setConnected, updateFromWS]);
}
