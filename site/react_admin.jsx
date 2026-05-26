import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';

const TOKEN_KEY = 'scania_component_x_api_token_v1';
const USER_KEY = 'scania_component_x_api_user_v1';
const STATUS_OPTIONS = ['open', 'scheduled', 'resolved'];
const COUNTER_COLUMNS = [
  'counter_100_0',
  'counter_171_0',
  'counter_309_0',
  'counter_370_0',
  'counter_427_0',
  'counter_666_0',
  'counter_835_0',
  'counter_837_0'
];

function apiFetch(path, { token, method = 'GET', body } = {}) {
  const headers = { Accept: 'application/json' };
  if (body) headers['Content-Type'] = 'application/json';
  if (token) headers.Authorization = `Bearer ${token}`;
  return fetch(path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined
  }).then(async response => {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed with ${response.status}`);
    return payload;
  });
}

function blankVehicle() {
  return {
    vehicle_id: '',
    split: 'manual',
    risk_class: '0',
    timeline_kind: 'manual_entry',
    latest_time_step: '0'
  };
}

function blankTelemetry(vehicleDbId = '') {
  return {
    vehicle_db_id: vehicleDbId,
    time_step: '',
    source: 'manual',
    counter_100_0: '0',
    counter_171_0: '0',
    counter_309_0: '0',
    counter_370_0: '0',
    counter_427_0: '0',
    counter_666_0: '0',
    counter_835_0: '0',
    counter_837_0: '0'
  };
}

function blankLog() {
  return {
    vehicle_id: '228',
    risk_class: '4',
    status: 'open',
    note: ''
  };
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return '-';
  return `${Math.round(Number(value) * 1000) / 10}%`;
}

function displayModelName(name) {
  if (!name) return '-';
  if (name === 'nearest_centroid_component_x_v2') return 'Nearest-centroid live scorer';
  return String(name).replaceAll('_', ' ');
}

function LoginPanel({ login, setLogin, busy, onLogin }) {
  return (
    <form className="maintenance-login" onSubmit={onLogin}>
      <div className="login-help">
        <strong>Demo accounts</strong>
        <span>Admin: demoadmin / ComponentX453!</span>
        <span>Technician: demouser / ComponentX453!</span>
      </div>
      <label>
        <span>Username</span>
        <input
          value={login.username}
          maxLength={80}
          required
          onChange={event => setLogin({ ...login, username: event.target.value })}
        />
      </label>
      <label>
        <span>Password</span>
        <input
          value={login.password}
          type="password"
          maxLength={200}
          required
          onChange={event => setLogin({ ...login, password: event.target.value })}
        />
      </label>
      <button type="submit" disabled={busy}>Sign In</button>
    </form>
  );
}

function TelemetryDataTab({ token, busy, setBusy, setMessage }) {
  const [vehicleFilters, setVehicleFilters] = useState({ search: '', risk: 'all', split: 'all' });
  const [telemetryFilters, setTelemetryFilters] = useState({ vehicle_id: '228', risk: 'all' });
  const [vehicles, setVehicles] = useState([]);
  const [vehicleTotal, setVehicleTotal] = useState(0);
  const [telemetry, setTelemetry] = useState([]);
  const [telemetryTotal, setTelemetryTotal] = useState(0);
  const [vehicleForm, setVehicleForm] = useState(blankVehicle);
  const [telemetryForm, setTelemetryForm] = useState(blankTelemetry);

  async function loadVehicles() {
    const params = new URLSearchParams({ ...vehicleFilters, limit: '75' });
    const payload = await apiFetch(`/api/app-vehicles?${params}`, { token });
    setVehicles(payload.vehicles || []);
    setVehicleTotal(payload.total || 0);
    if (!telemetryForm.vehicle_db_id && payload.vehicles?.[0]) {
      setTelemetryForm(blankTelemetry(String(payload.vehicles[0].id)));
    }
  }

  async function loadTelemetry() {
    const params = new URLSearchParams({ ...telemetryFilters, limit: '75' });
    const payload = await apiFetch(`/api/telemetry-records?${params}`, { token });
    setTelemetry(payload.telemetry || []);
    setTelemetryTotal(payload.total || 0);
  }

  useEffect(() => {
    loadVehicles().catch(error => setMessage(error.message));
  }, [vehicleFilters.search, vehicleFilters.risk, vehicleFilters.split]);

  useEffect(() => {
    loadTelemetry().catch(error => setMessage(error.message));
  }, [telemetryFilters.vehicle_id, telemetryFilters.risk]);

  async function createVehicle(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await apiFetch('/api/app-vehicles', {
        method: 'POST',
        token,
        body: {
          vehicle_id: Number(vehicleForm.vehicle_id),
          split: vehicleForm.split,
          risk_class: Number(vehicleForm.risk_class),
          timeline_kind: vehicleForm.timeline_kind,
          latest_time_step: Number(vehicleForm.latest_time_step)
        }
      });
      setVehicleForm(blankVehicle());
      await loadVehicles();
      setMessage('Vehicle saved.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function updateVehicle(id, patch) {
    setBusy(true);
    try {
      await apiFetch(`/api/app-vehicles/${id}`, { method: 'PUT', token, body: patch });
      await loadVehicles();
      setMessage('Vehicle updated.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteVehicle(id) {
    setBusy(true);
    try {
      await apiFetch(`/api/app-vehicles/${id}`, { method: 'DELETE', token });
      await loadVehicles();
      setMessage('Vehicle archived.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function createTelemetry(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await apiFetch('/api/telemetry-records', {
        method: 'POST',
        token,
        body: Object.fromEntries(Object.entries(telemetryForm).map(([key, value]) => [
          key,
          key === 'source' ? value : Number(value)
        ]))
      });
      setTelemetryForm(blankTelemetry(telemetryForm.vehicle_db_id));
      await loadTelemetry();
      setMessage('Telemetry record saved.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function updateTelemetry(id, patch) {
    setBusy(true);
    try {
      await apiFetch(`/api/telemetry-records/${id}`, { method: 'PUT', token, body: patch });
      await loadTelemetry();
      setMessage('Telemetry record updated.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteTelemetry(id) {
    setBusy(true);
    try {
      await apiFetch(`/api/telemetry-records/${id}`, { method: 'DELETE', token });
      await loadTelemetry();
      setMessage('Telemetry record deleted.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="tab-grid">
      <section className="data-panel">
        <h3>Vehicles</h3>
        <div className="filter-row">
          <input placeholder="vehicle id" value={vehicleFilters.search} onChange={event => setVehicleFilters({ ...vehicleFilters, search: event.target.value })} />
          <select value={vehicleFilters.split} onChange={event => setVehicleFilters({ ...vehicleFilters, split: event.target.value })}>
            <option value="all">all splits</option>
            <option value="train">train</option>
            <option value="validation">validation</option>
            <option value="test">test</option>
            <option value="manual">manual</option>
          </select>
          <select value={vehicleFilters.risk} onChange={event => setVehicleFilters({ ...vehicleFilters, risk: event.target.value })}>
            <option value="all">all classes</option>
            {[0, 1, 2, 3, 4].map(value => <option key={value} value={value}>class {value}</option>)}
          </select>
        </div>
        <form className="compact-form" onSubmit={createVehicle}>
          <input placeholder="vehicle_id" value={vehicleForm.vehicle_id} onChange={event => setVehicleForm({ ...vehicleForm, vehicle_id: event.target.value })} required />
          <select value={vehicleForm.split} onChange={event => setVehicleForm({ ...vehicleForm, split: event.target.value })}>
            <option value="manual">manual</option>
            <option value="train">train</option>
            <option value="validation">validation</option>
            <option value="test">test</option>
          </select>
          <select value={vehicleForm.risk_class} onChange={event => setVehicleForm({ ...vehicleForm, risk_class: event.target.value })}>
            {[0, 1, 2, 3, 4].map(value => <option key={value} value={value}>{value}</option>)}
          </select>
          <input placeholder="timeline_kind" value={vehicleForm.timeline_kind} onChange={event => setVehicleForm({ ...vehicleForm, timeline_kind: event.target.value })} required />
          <input type="number" min="0" step="0.1" placeholder="latest t" value={vehicleForm.latest_time_step} onChange={event => setVehicleForm({ ...vehicleForm, latest_time_step: event.target.value })} required />
          <button type="submit" disabled={busy}>Add Vehicle</button>
        </form>
        <div className="table-caption">{vehicleTotal} vehicles match.</div>
        <div className="data-table vehicle-admin-table">
          <div className="data-heading">DB ID</div><div className="data-heading">Vehicle</div><div className="data-heading">Split</div><div className="data-heading">Class</div><div className="data-heading">Latest</div><div className="data-heading">Actions</div>
          {vehicles.map(vehicle => (
            <React.Fragment key={vehicle.id}>
              <div>{vehicle.id}</div>
              <div>{vehicle.vehicle_id}</div>
              <div>{vehicle.split}</div>
              <div>
                <select value={vehicle.risk_class ?? 0} onChange={event => updateVehicle(vehicle.id, { risk_class: Number(event.target.value) })}>
                  {[0, 1, 2, 3, 4].map(value => <option key={value} value={value}>{value}</option>)}
                </select>
              </div>
              <div>{vehicle.latest_time_step ?? '-'}</div>
              <div><button type="button" className="secondary-action" onClick={() => deleteVehicle(vehicle.id)} disabled={busy}>Delete</button></div>
            </React.Fragment>
          ))}
        </div>
      </section>

      <section className="data-panel">
        <h3>Telemetry Records</h3>
        <div className="filter-row">
          <input placeholder="vehicle id" value={telemetryFilters.vehicle_id} onChange={event => setTelemetryFilters({ ...telemetryFilters, vehicle_id: event.target.value })} />
          <select value={telemetryFilters.risk} onChange={event => setTelemetryFilters({ ...telemetryFilters, risk: event.target.value })}>
            <option value="all">all classes</option>
            {[0, 1, 2, 3, 4].map(value => <option key={value} value={value}>class {value}</option>)}
          </select>
        </div>
        <form className="compact-form telemetry-form" onSubmit={createTelemetry}>
          <input placeholder="vehicle_db_id" value={telemetryForm.vehicle_db_id} onChange={event => setTelemetryForm({ ...telemetryForm, vehicle_db_id: event.target.value })} required />
          <input type="number" min="0" step="0.1" placeholder="time_step" value={telemetryForm.time_step} onChange={event => setTelemetryForm({ ...telemetryForm, time_step: event.target.value })} required />
          {COUNTER_COLUMNS.slice(0, 4).map(column => (
            <input key={column} type="number" min="0" step="0.001" placeholder={column} value={telemetryForm[column]} onChange={event => setTelemetryForm({ ...telemetryForm, [column]: event.target.value })} />
          ))}
          <button type="submit" disabled={busy}>Add Telemetry</button>
        </form>
        <div className="table-caption">{telemetryTotal} telemetry records match.</div>
        <div className="data-table telemetry-table-admin">
          <div className="data-heading">Vehicle</div><div className="data-heading">Class</div><div className="data-heading">t</div><div className="data-heading">100_0</div><div className="data-heading">Actions</div>
          {telemetry.map(row => (
            <React.Fragment key={row.id}>
              <div>{row.vehicle_id}</div>
              <div>{row.risk_class}</div>
              <div>{row.time_step}</div>
              <div><input defaultValue={row.counter_100_0 ?? ''} onBlur={event => updateTelemetry(row.id, { counter_100_0: Number(event.target.value) })} /></div>
              <div><button type="button" className="secondary-action" onClick={() => deleteTelemetry(row.id)} disabled={busy}>Delete</button></div>
            </React.Fragment>
          ))}
        </div>
      </section>
    </div>
  );
}

function MaintenanceLogsTab({ token, busy, setBusy, setMessage }) {
  const [notes, setNotes] = useState([]);
  const [history, setHistory] = useState([]);
  const [form, setForm] = useState(blankLog);

  const sortedNotes = useMemo(() => [...notes].sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at))), [notes]);

  async function loadNotes() {
    const payload = await apiFetch('/api/notes', { token });
    setNotes(payload.notes || []);
  }

  async function loadHistory(noteId = '') {
    const suffix = noteId ? `?note_id=${noteId}` : '';
    const payload = await apiFetch(`/api/notes/history${suffix}`, { token });
    setHistory(payload.history || []);
  }

  useEffect(() => {
    loadNotes().catch(error => setMessage(error.message));
    loadHistory().catch(error => setMessage(error.message));
  }, []);

  async function createNote(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await apiFetch('/api/notes', {
        method: 'POST',
        token,
        body: {
          vehicle_id: Number(form.vehicle_id),
          risk_class: Number(form.risk_class),
          status: form.status,
          note: form.note
        }
      });
      setForm(blankLog());
      await loadNotes();
      await loadHistory();
      setMessage('Maintenance log saved.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function updateNote(id, patch) {
    setBusy(true);
    try {
      await apiFetch(`/api/notes/${id}`, { method: 'PUT', token, body: patch });
      await loadNotes();
      await loadHistory();
      setMessage('Maintenance log updated.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteNote(id) {
    setBusy(true);
    try {
      await apiFetch(`/api/notes/${id}`, { method: 'DELETE', token });
      await loadNotes();
      await loadHistory();
      setMessage('Maintenance log removed from active list. History was kept.');
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="tab-grid">
      <section className="data-panel">
        <h3>New Maintenance Log</h3>
        <form className="maintenance-form" onSubmit={createNote}>
          <label><span>vehicle_id</span><input type="number" min="1" required value={form.vehicle_id} onChange={event => setForm({ ...form, vehicle_id: event.target.value })} /></label>
          <label><span>risk_class</span><select value={form.risk_class} onChange={event => setForm({ ...form, risk_class: event.target.value })}>{[0, 1, 2, 3, 4].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
          <label><span>status</span><select value={form.status} onChange={event => setForm({ ...form, status: event.target.value })}>{STATUS_OPTIONS.map(value => <option key={value} value={value}>{value}</option>)}</select></label>
          <label className="wide-field"><span>maintenance_note</span><textarea value={form.note} required maxLength={500} rows={4} onChange={event => setForm({ ...form, note: event.target.value })} /></label>
          <button type="submit" disabled={busy}>Create Log</button>
        </form>
      </section>

      <section className="data-panel">
        <h3>Active Logs</h3>
        <div className="maintenance-list">
          {sortedNotes.length === 0 ? <div className="empty-state">No active maintenance logs.</div> : sortedNotes.map(note => (
            <article className="maintenance-note" key={note.id}>
              <div>
                <strong>Vehicle {note.vehicle_id}</strong>
                <span>Class {note.risk_class} | {note.created_by || 'demo'} | {new Date(note.updated_at).toLocaleString()}</span>
              </div>
              <textarea defaultValue={note.note} maxLength={500} rows={3} onBlur={event => {
                if (event.target.value.trim() !== note.note) updateNote(note.id, { note: event.target.value });
              }} />
              <div className="note-actions">
                <select value={note.status} onChange={event => updateNote(note.id, { status: event.target.value })}>
                  {STATUS_OPTIONS.map(value => <option key={value} value={value}>{value}</option>)}
                </select>
                <button type="button" className="secondary-action" onClick={() => loadHistory(note.id)}>History</button>
                <button type="button" className="secondary-action" onClick={() => deleteNote(note.id)} disabled={busy}>Delete</button>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="data-panel full-width-panel">
        <h3>Maintenance History</h3>
        <div className="data-table history-table">
          <div className="data-heading">Note</div><div className="data-heading">Action</div><div className="data-heading">Status</div><div className="data-heading">Actor</div><div className="data-heading">Time</div>
          {history.map(row => (
            <React.Fragment key={row.id}>
              <div>{row.note_id}</div>
              <div>{row.action}</div>
              <div>{row.previous_status || '-'} {'->'} {row.new_status || '-'}</div>
              <div>{row.actor}</div>
              <div>{new Date(row.created_at).toLocaleString()}</div>
            </React.Fragment>
          ))}
        </div>
      </section>
    </div>
  );
}

function MlModelTab({ token, busy, setBusy, setMessage }) {
  const [summary, setSummary] = useState(null);
  const [analysisLines, setAnalysisLines] = useState([]);

  async function loadSummary() {
    const payload = await apiFetch('/api/ml/summary', { token });
    setSummary(payload);
  }

  useEffect(() => {
    loadSummary().catch(error => setMessage(error.message));
  }, []);

  async function runAnalysis() {
    setBusy(true);
    try {
      const payload = await apiFetch('/api/ml/run-analysis', { method: 'POST', token });
      setSummary(payload.summary);
      setAnalysisLines(payload.output || []);
      setMessage(`ML analysis completed at ${new Date(payload.completedAt).toLocaleString()}.`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  if (!summary) return <div className="empty-state">Loading ML model summary.</div>;

  return (
    <div className="tab-grid">
      <section className="data-panel">
        <h3>ML Analysis</h3>
        <div className="ml-summary-grid">
          <div><span>Recommended model</span><strong>{summary.modelName}</strong></div>
          <div><span>Dashboard scorer</span><strong>{displayModelName(summary.liveModelName)}</strong></div>
          <div><span>Validation accuracy</span><strong>{formatPercent(summary.validation?.accuracy)}</strong></div>
          <div><span>Validation cost</span><strong>{summary.validation?.totalCost ?? '-'}</strong></div>
          <div><span>Test accuracy</span><strong>{formatPercent(summary.test?.accuracy)}</strong></div>
          <div><span>Test cost</span><strong>{summary.test?.totalCost ?? '-'}</strong></div>
        </div>
        {summary.modelSelection?.recommendationRule ? <p className="table-caption">{summary.modelSelection.recommendationRule}</p> : null}
        <button type="button" onClick={runAnalysis} disabled={busy}>Run ML Analysis</button>
        <div className="analysis-complete-line">Analysis completed: {summary.generatedAt ? new Date(summary.generatedAt).toLocaleString() : '-'}</div>
        {analysisLines.length ? <pre className="analysis-output">{analysisLines.join('\n')}</pre> : null}
      </section>

      <section className="data-panel">
        <h3>Model Comparison</h3>
        <div className="data-table model-admin-table">
          <div className="data-heading">Model</div><div className="data-heading">Val Acc</div><div className="data-heading">Val Cost</div><div className="data-heading">Test Acc</div><div className="data-heading">Test Cost</div>
          {(summary.comparison || []).map(row => (
            <React.Fragment key={row.name}>
              <div className={row.name === summary.modelName ? 'recommended-cell' : ''}>{row.name}</div>
              <div className={row.name === summary.modelName ? 'recommended-cell' : ''}>{formatPercent(row.validation?.accuracy)}</div>
              <div className={row.name === summary.modelName ? 'recommended-cell' : ''}>{row.validation?.totalCost ?? '-'}</div>
              <div>{formatPercent(row.test?.accuracy)}</div>
              <div>{row.test?.totalCost ?? '-'}</div>
            </React.Fragment>
          ))}
        </div>
      </section>
    </div>
  );
}

function FullStackApp() {
  const [token, setToken] = useState(() => window.localStorage.getItem(TOKEN_KEY) || '');
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(window.localStorage.getItem(USER_KEY) || 'null');
    } catch {
      return null;
    }
  });
  const [login, setLogin] = useState({ username: 'demoadmin', password: 'ComponentX453!' });
  const [activeTab, setActiveTab] = useState('telemetry');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const isAdmin = user?.role === 'admin';

  useEffect(() => {
    if (user && !isAdmin && activeTab !== 'logs') setActiveTab('logs');
  }, [user?.role, activeTab, isAdmin]);

  function rememberSession(payload) {
    window.localStorage.setItem(TOKEN_KEY, payload.token);
    window.localStorage.setItem(USER_KEY, JSON.stringify(payload.user));
    setToken(payload.token);
    setUser(payload.user);
    setActiveTab(payload.user.role === 'admin' ? 'telemetry' : 'logs');
  }

  function clearSession() {
    window.localStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(USER_KEY);
    setToken('');
    setUser(null);
  }

  async function handleLogin(event) {
    event.preventDefault();
    setBusy(true);
    setMessage('');
    try {
      const payload = await apiFetch('/api/auth/login', { method: 'POST', body: login });
      rememberSession(payload);
      setMessage(`Signed in as ${payload.user.username}.`);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleLogout() {
    setBusy(true);
    try {
      if (token) await apiFetch('/api/auth/logout', { method: 'POST', token });
    } catch {
      // Local session cleanup still happens below.
    } finally {
      clearSession();
      setMessage('Signed out.');
      setBusy(false);
    }
  }

  return (
    <div className="maintenance-app">
      <div className="maintenance-heading">
        <div>
          <p className="eyebrow">Fleet Maintenance</p>
          <h2>Operations Console</h2>
          {user ? <p className="role-line">Signed in as {user.username} ({user.role})</p> : null}
        </div>
        {user ? <button type="button" className="secondary-action" onClick={handleLogout} disabled={busy}>Sign Out</button> : null}
      </div>

      {!user ? (
        <LoginPanel login={login} setLogin={setLogin} busy={busy} onLogin={handleLogin} />
      ) : (
        <>
          <div className="app-tabs">
            {isAdmin ? <button type="button" className={activeTab === 'telemetry' ? 'active' : ''} onClick={() => setActiveTab('telemetry')}>Telemetry Data</button> : null}
            <button type="button" className={activeTab === 'logs' ? 'active' : ''} onClick={() => setActiveTab('logs')}>Maintenance Logs</button>
            {isAdmin ? <button type="button" className={activeTab === 'ml' ? 'active' : ''} onClick={() => setActiveTab('ml')}>ML Model</button> : null}
          </div>
          <div className="tab-body">
            {activeTab === 'telemetry' && isAdmin ? <TelemetryDataTab token={token} busy={busy} setBusy={setBusy} setMessage={setMessage} /> : null}
            {activeTab === 'logs' ? <MaintenanceLogsTab token={token} busy={busy} setBusy={setBusy} setMessage={setMessage} /> : null}
            {activeTab === 'ml' && isAdmin ? <MlModelTab token={token} busy={busy} setBusy={setBusy} setMessage={setMessage} /> : null}
          </div>
        </>
      )}

      <div className="maintenance-status" role="status">{message}</div>
    </div>
  );
}

const root = document.getElementById('maintenance-notes-root');
if (root) {
  createRoot(root).render(<FullStackApp />);
}
