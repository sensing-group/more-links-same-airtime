import { useState, useEffect, useCallback } from 'react';

type FieldDef = {
  key: string;
  label: string;
  placeholder?: string;
  type?: 'text' | 'number' | 'select';
  options?: string[];
  nested?: string; // dot path for nested objects like "dimensions_m.length"
  readOnly?: boolean;
};

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '8px 10px', border: '1px solid var(--border-strong)', borderRadius: 6,
  fontSize: 13, boxSizing: 'border-box', background: '#fff',
};
const btnStyle: React.CSSProperties = {
  padding: '6px 14px', border: '1px solid var(--border-strong)', borderRadius: 6,
  cursor: 'pointer', fontSize: 13, fontWeight: 500, background: '#fff',
};
const cardStyle: React.CSSProperties = {
  padding: 16, background: '#fff', borderRadius: 12,
  border: '1px solid var(--border)', marginBottom: 10,
  boxShadow: 'var(--shadow-sm)',
};

function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
  return path.split('.').reduce((o: unknown, k) => (o && typeof o === 'object' ? (o as Record<string, unknown>)[k] : undefined), obj);
}

function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const clone = JSON.parse(JSON.stringify(obj));
  const keys = path.split('.');
  let cur = clone;
  for (let i = 0; i < keys.length - 1; i++) {
    if (!cur[keys[i]] || typeof cur[keys[i]] !== 'object') cur[keys[i]] = {};
    cur = cur[keys[i]] as Record<string, unknown>;
  }
  cur[keys[keys.length - 1]] = value;
  return clone;
}

function ItemCard({ item, fields, idField, endpoint, reload }: {
  item: Record<string, unknown>; fields: FieldDef[]; idField: string;
  endpoint: string; reload: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState<Record<string, unknown>>(item);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => { setForm(item); }, [item]);

  const handleSave = async () => {
    setSaving(true);
    setError('');
    try {
      const r = await fetch(endpoint, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!r.ok) { const d = await r.json(); setError(d.detail || 'Failed'); setSaving(false); return; }
      setEditing(false);
      reload();
    } catch { setError('Connection error'); }
    setSaving(false);
  };

  const handleDelete = async () => {
    if (!confirm(`Delete ${item[idField]}?`)) return;
    await fetch(`${endpoint}/${item[idField]}`, { method: 'DELETE' });
    reload();
  };

  const fieldValue = (f: FieldDef): string => {
    const v = f.nested ? getNestedValue(form, f.nested) : form[f.key];
    return v == null ? '' : String(v);
  };

  const setFieldValue = (f: FieldDef, val: string) => {
    const path = f.nested || f.key;
    const parsed = f.type === 'number' ? (val === '' ? null : Number(val)) : val;
    setForm(prev => setNestedValue(prev as Record<string, unknown>, path, parsed));
  };

  const id = String(item[idField]);

  if (!editing) {
    return (
      <div style={cardStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <strong style={{ fontSize: 15 }}>{id}</strong>
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={() => setEditing(true)} style={{ ...btnStyle, background: 'var(--brand-soft)', color: '#1d4ed8', borderColor: 'var(--brand-soft)' }}>Edit</button>
            <button onClick={handleDelete} style={{ ...btnStyle, background: 'var(--danger-soft)', color: 'var(--danger)', borderColor: 'var(--danger-soft)' }}>Delete</button>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '4px 16px', fontSize: 13, color: '#555' }}>
          {fields.filter(f => f.key !== idField).map(f => {
            const v = f.nested ? getNestedValue(item, f.nested) : item[f.key];
            const display = v == null || v === '' ? '-' : String(v);
            return <div key={f.nested || f.key}><span style={{ color: '#999' }}>{f.label}:</span> {display}</div>;
          })}
        </div>
      </div>
    );
  }

  return (
    <div style={{ ...cardStyle, border: '2px solid var(--brand)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <strong style={{ fontSize: 15 }}>{id}</strong>
        <div style={{ display: 'flex', gap: 6 }}>
          <button onClick={handleSave} disabled={saving}
                  style={{ ...btnStyle, background: 'var(--success)', color: '#fff', borderColor: 'var(--success)' }}>
            {saving ? 'Saving…' : 'Save'}
          </button>
          <button onClick={() => { setForm(item); setEditing(false); setError(''); }}
                  style={{ ...btnStyle }}>Cancel</button>
        </div>
      </div>
      {error && <div style={{ color: '#ef4444', fontSize: 12, marginBottom: 8 }}>{error}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
        {fields.map(f => {
          const key = f.nested || f.key;
          const isId = f.key === idField;
          return (
            <div key={key}>
              <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 3, color: '#555' }}>
                {f.label}
              </label>
              {f.type === 'select' && f.options ? (
                <select value={fieldValue(f)} onChange={e => setFieldValue(f, e.target.value)}
                        disabled={isId} style={{ ...inputStyle, background: isId ? '#f3f4f6' : '#fff' }}>
                  <option value="">-</option>
                  {f.options.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              ) : (
                <input type={f.type === 'number' ? 'number' : 'text'}
                       value={fieldValue(f)}
                       onChange={e => setFieldValue(f, e.target.value)}
                       placeholder={f.placeholder}
                       disabled={isId}
                       style={{ ...inputStyle, background: isId ? '#f3f4f6' : '#fff' }} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Section({ title, endpoint, idField, fields, items, reload }: {
  title: string; endpoint: string; idField: string;
  fields: FieldDef[]; items: Record<string, unknown>[]; reload: () => void;
}) {
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState<Record<string, unknown>>({});
  const [error, setError] = useState('');

  const handleAdd = async () => {
    setError('');
    const id = String(form[idField] || '').trim();
    if (!id) { setError(`${idField} is required`); return; }
    const data = { ...form, [idField]: id };
    try {
      const r = await fetch(endpoint, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!r.ok) { const d = await r.json(); setError(d.detail || 'Failed'); return; }
      setForm({});
      setAdding(false);
      reload();
    } catch { setError('Connection error'); }
  };

  const fieldValue = (f: FieldDef): string => {
    const v = f.nested ? getNestedValue(form as Record<string, unknown>, f.nested) : form[f.key];
    return v == null ? '' : String(v);
  };

  const setFieldValue = (f: FieldDef, val: string) => {
    const path = f.nested || f.key;
    const parsed = f.type === 'number' ? (val === '' ? null : Number(val)) : val;
    setForm(prev => setNestedValue(prev as Record<string, unknown>, path, parsed));
  };

  return (
    <div style={{ marginBottom: 28 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 16 }}>{title} ({items.length})</h3>
        {!adding && (
          <button onClick={() => setAdding(true)}
                  style={{ ...btnStyle, background: 'var(--brand)', color: '#fff', borderColor: 'var(--brand)' }}>
            + New
          </button>
        )}
      </div>

      {adding && (
        <div style={{ ...cardStyle, border: '2px dashed var(--brand)', background: 'var(--brand-soft)', marginBottom: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10, marginBottom: 10 }}>
            {fields.map(f => {
              const key = f.nested || f.key;
              return (
                <div key={key}>
                  <label style={{ fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 3, color: '#555' }}>
                    {f.label} {f.key === idField && <span style={{ color: '#ef4444' }}>*</span>}
                  </label>
                  {f.type === 'select' && f.options ? (
                    <select value={fieldValue(f)} onChange={e => setFieldValue(f, e.target.value)} style={inputStyle}>
                      <option value="">-</option>
                      {f.options.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  ) : (
                    <input type={f.type === 'number' ? 'number' : 'text'}
                           value={fieldValue(f)}
                           onChange={e => setFieldValue(f, e.target.value)}
                           placeholder={f.placeholder}
                           style={inputStyle} />
                  )}
                </div>
              );
            })}
          </div>
          {error && <div style={{ color: 'var(--danger)', fontSize: 12, marginBottom: 6 }}>{error}</div>}
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={handleAdd} style={{ ...btnStyle, background: 'var(--success)', color: '#fff', borderColor: 'var(--success)' }}>Create</button>
            <button onClick={() => { setAdding(false); setForm({}); setError(''); }}
                    style={{ ...btnStyle }}>Cancel</button>
          </div>
        </div>
      )}

      {items.length === 0 && !adding && (
        <div style={{ fontSize: 13, color: '#999', padding: 12 }}>None yet. Click "+ New" to add one.</div>
      )}

      {items.map(item => (
        <ItemCard key={String(item[idField])} item={item} fields={fields}
                  idField={idField} endpoint={endpoint} reload={reload} />
      ))}
    </div>
  );
}

const SUBJECT_FIELDS: FieldDef[] = [
  { key: 'subject_id', label: 'Subject ID', placeholder: 'S01' },
  { key: 'height_cm', label: 'Height (cm)', type: 'number', placeholder: '170' },
  { key: 'weight_kg', label: 'Weight (kg)', type: 'number', placeholder: '70' },
  { key: 'age_range', label: 'Age Range', placeholder: 'e.g. 20-30' },
  { key: 'notes', label: 'Notes', placeholder: 'Optional notes' },
];

const ENV_FIELDS: FieldDef[] = [
  { key: 'environment_id', label: 'Environment ID', placeholder: 'lab-alpha' },
  { key: 'description', label: 'Description', placeholder: 'Room description' },
  { key: 'dimensions_m', label: 'Length (m)', type: 'number', nested: 'dimensions_m.length', placeholder: '0' },
  { key: '_dim_w', label: 'Width (m)', type: 'number', nested: 'dimensions_m.width', placeholder: '0' },
  { key: '_dim_h', label: 'Height (m)', type: 'number', nested: 'dimensions_m.height', placeholder: '0' },
  { key: 'furniture', label: 'Furniture', placeholder: 'Desks, chairs, etc.' },
  { key: 'wall_material', label: 'Wall Material', placeholder: 'e.g. drywall, concrete' },
  { key: 'floor_material', label: 'Floor Material', placeholder: 'e.g. tile, carpet' },
];

const TOPO_FIELDS: FieldDef[] = [
  { key: 'topology_id', label: 'Topology ID', placeholder: 'topo-10node' },
  { key: 'description', label: 'Description', placeholder: 'Node arrangement description' },
  { key: 'num_nodes', label: 'Num Nodes', type: 'number', placeholder: '10' },
  { key: 'height_profile', label: 'Height Profile', type: 'select', options: ['variable', 'fixed'] },
  { key: 'channel', label: 'WiFi Channel', type: 'number', placeholder: '6' },
  { key: 'bandwidth', label: 'Bandwidth', type: 'select', options: ['HT20', 'HT40'] },
  { key: 'tx_power_dbm', label: 'TX Power (dBm)', type: 'number', placeholder: '20' },
  { key: 'antenna_type', label: 'Antenna Type', type: 'select', options: ['onboard', 'external', 'patch'] },
];

export function SetupPanel() {
  const [environments, setEnvironments] = useState<Record<string, unknown>[]>([]);
  const [subjects, setSubjects] = useState<Record<string, unknown>[]>([]);
  const [topologies, setTopologies] = useState<Record<string, unknown>[]>([]);
  const [rfInfo, setRfInfo] = useState<{ channels: number[]; bandwidths: string[]; num_nodes: number } | null>(null);

  const loadEnv = useCallback(() => { fetch('/api/environments').then(r => r.json()).then(setEnvironments).catch(() => {}); }, []);
  const loadSub = useCallback(() => { fetch('/api/subjects').then(r => r.json()).then(setSubjects).catch(() => {}); }, []);
  const loadTopo = useCallback(() => { fetch('/api/topologies').then(r => r.json()).then(setTopologies).catch(() => {}); }, []);
  const loadRf = useCallback(() => { fetch('/api/rf-info').then(r => r.json()).then(setRfInfo).catch(() => {}); }, []);

  useEffect(() => { loadEnv(); loadSub(); loadTopo(); loadRf(); }, [loadEnv, loadSub, loadTopo, loadRf]);
  useEffect(() => { const id = setInterval(loadRf, 5000); return () => clearInterval(id); }, [loadRf]);

  return (
    <div style={{ maxWidth: 820, margin: '0 auto' }}>
      {rfInfo && rfInfo.num_nodes > 0 && (
        <div style={{
          padding: '10px 14px', background: 'var(--brand-soft)', border: '1px solid #bfdbfe',
          borderRadius: 10, marginBottom: 20, fontSize: 13, color: '#1e40af',
        }}>
          <strong>Live RF:</strong>{' '}
          {rfInfo.num_nodes} node{rfInfo.num_nodes !== 1 ? 's' : ''} online
          {rfInfo.channels.length > 0 && <> &middot; Channel {rfInfo.channels.join(', ')}</>}
          {rfInfo.bandwidths.length > 0 && <> &middot; {rfInfo.bandwidths.join(', ')}</>}
          <span style={{ color: '#6b7280', marginLeft: 8 }}>
            (auto-detected from packets)
          </span>
        </div>
      )}

      <Section title="Subjects" endpoint="/api/subjects" idField="subject_id"
               fields={SUBJECT_FIELDS} items={subjects} reload={loadSub} />

      <Section title="Environments" endpoint="/api/environments" idField="environment_id"
               fields={ENV_FIELDS} items={environments} reload={loadEnv} />

      <Section title="Topologies" endpoint="/api/topologies" idField="topology_id"
               fields={TOPO_FIELDS} items={topologies} reload={loadTopo} />
    </div>
  );
}
