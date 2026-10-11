import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceLine } from 'recharts'
import { GlobalImportance, Physiology } from './Panels'
import { api, Demo, Feature, Metrics, Result, TargetName } from './api'

const Heart3D = lazy(() => import('./Heart3D'))
const COLORS: Record<string, string> = { LOW: '#22c55e', MODERATE: '#f59e0b', HIGH: '#ef4444' }
const KEY = ['Age', 'Sex', 'BMI', 'BP', 'PR', 'HTN', 'DM', 'Current Smoker', 'FH', 'Typical Chest Pain', 'Atypical', 'Region RWMA',
  'EF-TTE', 'FBS', 'LDL', 'HDL', 'TG', 'ESR', 'HB', 'Tinversion', 'ST Elevation', 'ST Depression', 'LVH', 'Q Wave']
const pct = (p: number) => `${(p * 100).toFixed(0)}%`

const LABELS: Record<string, string[]> = { Sex: ['Female', 'Male'], VHD: ['None', 'Mild', 'Moderate', 'Severe'] }
const bounds = (f: Feature) => { const sp = f.max - f.min; return [f.min >= 0 ? Math.max(f.min - 0.5 * sp, 0) : f.min - 0.5 * sp, f.max + 0.5 * sp] }
const isBad = (f: Feature, v: string) => f.type === 'numeric' && v !== '' && (Number(v) < bounds(f)[0] || Number(v) > bounds(f)[1])

function Field({ f, v, set }: { f: Feature; v: string; set: (s: string) => void }) {
  const label = <label className="block text-[11px] text-slate-400 mb-0.5">{f.name}</label>
  if (f.type === 'binary' || (f.type === 'ordinal' && f.max - f.min <= 4))
    return <div>{label}<select value={v} onChange={e => set(e.target.value)}><option value="">— (impute)</option>
      {Array.from({ length: f.max - f.min + 1 }, (_, i) => f.min + i).map(o => <option key={o} value={o}>{LABELS[f.name]?.[o] ?? (f.type === 'binary' ? (o ? 'Yes' : 'No') : o)}</option>)}</select></div>
  const bad = isBad(f, v); const [lo, hi] = bounds(f)
  return <div>{label}<input type="number" step="any" placeholder={`median ${f.median}`} value={v} onChange={e => set(e.target.value)} aria-invalid={bad}
    className={bad ? '!border-red-500' : ''} title={`plausible range ${lo.toPrecision(3)} to ${hi.toPrecision(3)}`} />{bad && <span className="text-[10px] text-red-400">outside {lo.toPrecision(3)}–{hi.toPrecision(3)}</span>}</div>
}

export default function App() {
  const [schema, setSchema] = useState<Feature[]>([]); const [demos, setDemos] = useState<Record<string, Demo>>({})
  const [metrics, setMetrics] = useState<Metrics | null>(null); const [vals, setVals] = useState<Record<string, string>>({})
  const [res, setRes] = useState<Result | null>(null); const [sel, setSel] = useState<TargetName>('CAD')
  const [analyzed, setAnalyzed] = useState(''); const [busy, setBusy] = useState(false); const [err, setErr] = useState(''); const [more, setMore] = useState(false)

  useEffect(() => { Promise.all([api.schema(), api.demos(), api.metrics()]).then(([s, d, m]) => { setSchema(s.features); setDemos(d); setMetrics(m) })
    .catch(e => setErr(`Cannot reach the API (${e.message}). Start it with: uvicorn backend.app.main:app --port 8000`)) }, [])

  const hasData = Object.values(vals).some(x => x !== '')
  const byName = useMemo(() => Object.fromEntries(schema.map(f => [f.name, f])), [schema])
  const invalid = schema.some(f => isBad(f, vals[f.name] ?? ''))
  const dirty = !!res && JSON.stringify(vals) !== analyzed
  const key = KEY.filter(k => byName[k]).map(k => byName[k]); const rest = schema.filter(f => !KEY.includes(f.name))

  const analyze = async (v = vals) => {
    setBusy(true); setErr('')
    try { const feats: Record<string, number> = {}; Object.entries(v).forEach(([k, s]) => { if (s !== '') feats[k] = Number(s) })
      setRes(await api.predict(feats)); setAnalyzed(JSON.stringify(v)) } catch (e: any) { setErr(e.message) } finally { setBusy(false) }
  }
  const loadDemo = (k: string) => { const v = Object.fromEntries(Object.entries(demos[k].features).map(([a, b]) => [a, String(b)])); setVals(v); setRes(null); setSel('CAD'); analyze(v) }

  const pred = res ? (sel === 'CAD' ? res.cad : res.vessels[sel]) : null
  const ex = res?.explanations[sel]
  const chart = ex?.top_features.map(f => ({ name: `${f.feature} = ${Number.isInteger(f.value) ? f.value : f.value.toFixed(1)}${f.imputed ? ' (imputed)' : ''}`, v: f.shap_value })) ?? []

  return (
    <div className="min-h-screen p-4 max-w-[1500px] mx-auto">
      <header className="flex flex-wrap items-end justify-between gap-2 mb-3">
        <div><h1 className="text-2xl font-semibold text-white">CardioVision <span className="text-cyan-400">AI</span></h1>
          <p className="text-sm text-slate-400">See the Risk. Understand the Heart.</p></div>
        <div className="flex gap-2">{Object.keys(demos).map(k => <button key={k} onClick={() => loadDemo(k)} className="px-3 py-1.5 text-sm rounded-lg bg-slate-800 border border-slate-600 hover:border-cyan-500">
          Load Demo Patient · {k.startsWith('high') ? 'high-risk' : 'low-risk'}</button>)}</div>
      </header>
      <div className="mb-3 text-xs text-amber-200 bg-amber-950/40 border border-amber-700/50 rounded-lg px-3 py-2">
        <b>Clinical decision-support prototype.</b> Predictions are for educational/research purposes only and are NOT a diagnosis or a substitute for professional medical evaluation or diagnostic imaging.
        {' '}The 3D visualization represents predicted vessel-level risk and does not depict an actual anatomical lesion.</div>
      {err && <div className="mb-3 text-sm text-red-300 bg-red-950/40 border border-red-800 rounded-lg px-3 py-2">{err}</div>}

      <div className="grid lg:grid-cols-[300px_1fr_320px] gap-3">
        <section className="glass p-3 fade-in lg:max-h-[640px] overflow-y-auto">
          <h2 className="font-medium mb-2">Patient clinical data</h2>
          <div className="grid grid-cols-2 gap-2">{key.map(f => <Field key={f.name} f={f} v={vals[f.name] ?? ''} set={s => setVals({ ...vals, [f.name]: s })} />)}</div>
          <button onClick={() => setMore(!more)} className="text-xs text-cyan-400 mt-3">{more ? 'Hide' : 'Show'} {rest.length} more features</button>
          {more && <div className="grid grid-cols-2 gap-2 mt-2">{rest.map(f => <Field key={f.name} f={f} v={vals[f.name] ?? ''} set={s => setVals({ ...vals, [f.name]: s })} />)}</div>}
          <p className="text-[11px] text-slate-500 mt-2">Empty fields are imputed with training medians.</p>
          <button disabled={busy || !schema.length || !hasData || invalid} onClick={() => analyze()} className="w-full mt-3 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 font-medium text-white">{busy ? 'Analyzing…' : 'Analyze'}</button>
          <button onClick={() => { setVals({}); setRes(null); setErr(''); setSel('CAD') }} className="w-full mt-1.5 py-1 text-xs rounded-lg border border-slate-700 text-slate-400 hover:border-slate-500">Clear form</button>
          {dirty && <p className="text-[11px] text-amber-300 mt-1">Inputs changed since the last analysis. Click Analyze to update.</p>}
        </section>

        <section className="glass p-3 flex flex-col items-center">
          <div className="flex gap-1 mb-1">{(['CAD', 'LAD', 'LCX', 'RCA'] as TargetName[]).map(t => <button key={t} onClick={() => setSel(t)}
            className={`px-3 py-1 text-xs rounded-md border ${sel === t ? 'border-cyan-400 text-cyan-300' : 'border-slate-700 text-slate-400'}`}>{t}</button>)}</div>
          <div className="w-full h-[460px]"><Suspense fallback={<p className="text-sm text-slate-500 p-4">Loading 3D heart…</p>}><Heart3D r={res} sel={sel} onSel={setSel} /></Suspense></div>
        </section>

        <section className="glass p-3">
          <h2 className="font-medium mb-2">Risk results</h2>
          {!res ? <p className="text-sm text-slate-500">Load a demo patient or enter data, then click Analyze.</p> : <>
            <div className="rounded-lg p-3 mb-2 border" style={{ borderColor: COLORS[res.cad.risk_category] }}>
              <div className="text-xs text-slate-400">Overall CAD probability</div>
              <div className="text-3xl font-semibold" style={{ color: COLORS[res.cad.risk_category] }}>{pct(res.cad.probability)}</div>
              <div className="text-xs">{res.cad.risk_category} · model AUC {res.cad.cv_roc_auc.toFixed(2)} (CV)</div>
              <div className="text-[11px] text-slate-400">Screening flag (threshold {res.cad.decision_threshold}): {res.cad.predicted_positive ? 'positive' : 'negative'}</div></div>
            {(['LAD', 'LCX', 'RCA'] as const).map(t => { const p = res.vessels[t]; return (
              <button key={t} onClick={() => setSel(t)} className={`w-full text-left rounded-lg p-2 mb-1.5 border ${sel === t ? 'border-cyan-400' : 'border-slate-700'}`}>
                <div className="flex justify-between text-sm"><span>Predicted {t} stenosis probability</span><b style={{ color: COLORS[p.risk_category] }}>{pct(p.probability)}</b></div>
                <div className="h-1.5 bg-slate-800 rounded mt-1"><div className="h-1.5 rounded" style={{ width: pct(p.probability), background: COLORS[p.risk_category] }} /></div>
                <div className="text-[11px] text-slate-500 mt-0.5">{p.risk_category} · model AUC {p.cv_roc_auc.toFixed(2)}{p.cv_roc_auc < 0.8 ? ' · lower model confidence' : ''}<br />Screening flag (threshold {p.decision_threshold}): {p.predicted_positive ? 'positive' : 'negative'}</div></button>) })}
            <p className="text-[11px] text-slate-500">{res.input_summary.n_provided}/{res.input_summary.n_features_total} features provided, {res.input_summary.n_imputed} imputed.</p></>}
        </section>
      </div>

      <div className="grid lg:grid-cols-[1fr_380px] gap-3 mt-3">
        <section className="glass p-3">
          <h2 className="font-medium">Why this prediction? <span className="text-cyan-400">{sel}</span></h2>
          {!ex ? <p className="text-sm text-slate-500 mt-2">Run an analysis to see patient-level SHAP contributions.</p> : <>
            <p className="text-xs text-slate-400 mb-1"><b className="text-red-400">↑ Red</b> bars push risk up, <b className="text-green-400">↓ green</b> bars push it down. Units: {ex.unit}. Values are real SHAP values from the {pred?.model} model.</p>
            <div style={{ height: 320 }}><ResponsiveContainer><BarChart data={chart} layout="vertical" margin={{ left: 150, right: 20 }}>
              <XAxis type="number" stroke="#64748b" fontSize={11} /><YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={11} width={150} interval={0} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} formatter={(v: number) => v.toFixed(4)} /><ReferenceLine x={0} stroke="#64748b" />
              <Bar dataKey="v" radius={3}>{chart.map((c, i) => <Cell key={i} fill={c.v > 0 ? '#ef4444' : '#22c55e'} />)}</Bar></BarChart></ResponsiveContainer></div></>}
        </section>
        <section className="glass p-3">
          <h2 className="font-medium mb-2">Model performance</h2>
          {metrics && <table className="w-full text-xs"><thead className="text-slate-400"><tr><th className="text-left">Target</th><th>CV AUC</th><th>Test AUC</th><th>Test recall</th></tr></thead><tbody>
            {Object.entries(metrics.metrics.targets).map(([t, m]: [string, any]) => <tr key={t} className="border-t border-slate-800 text-center"><td className="text-left py-1">{t} <span className="text-slate-500">{m.best_model}</span></td>
              <td>{m.model_comparison_nested_cv[m.best_model].cv_roc_auc_mean.toFixed(2)}</td><td>{m.test_at_threshold_0_5.roc_auc.toFixed(2)}</td><td>{m.test_at_threshold_0_5.recall.toFixed(2)}</td></tr>)}</tbody></table>}
          <p className="text-[11px] text-slate-500 mt-2">Nested stratified CV on training data; test = 61 held-out patients (small, noisy). LCX/RCA are weaker models.</p>
          {metrics && <><h3 className="text-xs text-slate-400 mt-3 mb-1">Global top features ({sel})</h3>
            <ol className="text-xs list-decimal ml-4">{metrics.global_importance[sel].features.slice(0, 6).map(f => <li key={f.feature}>{f.feature}</li>)}</ol></>}
        </section>
      </div>
      <div className="grid lg:grid-cols-2 gap-3 mt-3">
        {metrics && <GlobalImportance m={metrics} sel={sel} />}
        <Physiology schema={schema} vals={vals} />
      </div>
      <footer className="text-[11px] text-slate-600 text-center mt-4">CardioVision AI · research prototype · not for clinical use</footer>
    </div>)
}
