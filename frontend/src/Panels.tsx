import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Feature, Metrics, TargetName } from './api'

const PHYSIO = ['Age', 'BMI', 'BP', 'PR', 'FBS', 'LDL', 'HDL', 'TG', 'EF-TTE', 'ESR', 'HB', 'Cr']
const FINDINGS = ['HTN', 'DM', 'Current Smoker', 'FH', 'Typical Chest Pain', 'Atypical', 'Tinversion', 'ST Elevation', 'ST Depression', 'LVH', 'Q Wave', 'Poor R Progression', 'Region RWMA']

export function GlobalImportance({ m, sel }: { m: Metrics; sel: TargetName }) {
  const g = m.global_importance[sel]; const data = g.features.slice(0, 12).map(f => ({ name: f.feature, v: f.mean_abs_shap }))
  return (
    <section className="glass p-3 fade-in">
      <h2 className="font-medium">Global feature importance <span className="text-cyan-400">{sel}</span></h2>
      <p className="text-xs text-slate-400 mb-1">Mean |SHAP| over the 242 training patients ({g.unit}). Larger = the model relies on it more overall.</p>
      <div style={{ height: 300 }}><ResponsiveContainer><BarChart data={data} layout="vertical" margin={{ left: 100, right: 16 }}>
        <XAxis type="number" stroke="#64748b" fontSize={11} /><YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={11} width={100} interval={0} />
        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} formatter={(v: number) => v.toFixed(4)} />
        <Bar dataKey="v" radius={3}>{data.map((_, i) => <Cell key={i} fill="#22d3ee" fillOpacity={1 - i * 0.05} />)}</Bar></BarChart></ResponsiveContainer></div>
    </section>)
}

export function Physiology({ schema, vals }: { schema: Feature[]; vals: Record<string, string> }) {
  const by = Object.fromEntries(schema.map(f => [f.name, f]))
  const nums = PHYSIO.filter(n => by[n] && vals[n] !== undefined && vals[n] !== '')
  const flags = FINDINGS.filter(n => by[n] && Number(vals[n]) > 0)
  return (
    <section className="glass p-3 fade-in">
      <h2 className="font-medium">Patient physiology</h2>
      <p className="text-xs text-slate-400 mb-2">Position of each value within the 303-patient dataset cohort (shaded = middle 50%, tick = median). Not clinical reference ranges.</p>
      {nums.length === 0 && flags.length === 0 && <p className="text-sm text-slate-500">Enter data or load a demo patient.</p>}
      <div className="space-y-2">{nums.map(n => { const f = by[n]; const v = Number(vals[n]); const sp = f.max - f.min || 1; const pc = (x: number) => Math.min(100, Math.max(0, ((x - f.min) / sp) * 100))
        return (<div key={n}><div className="flex justify-between text-[11px]"><span className="text-slate-300">{n}</span><span className="text-slate-400">{v} <span className="text-slate-600">(cohort median {f.median})</span></span></div>
          <div className="relative h-2 bg-slate-800 rounded" role="img" aria-label={`${n} ${v}`}><div className="absolute h-2 bg-slate-600/70 rounded" style={{ left: `${pc(f.p25)}%`, width: `${pc(f.p75) - pc(f.p25)}%` }} />
            <div className="absolute h-2 w-px bg-slate-300" style={{ left: `${pc(f.median)}%` }} /><div className="absolute -top-0.5 h-3 w-1.5 rounded bg-cyan-400" style={{ left: `calc(${pc(v)}% - 3px)` }} /></div></div>) })}</div>
      {flags.length > 0 && <div className="mt-3"><div className="text-[11px] text-slate-400 mb-1">Positive findings entered</div>
        <div className="flex flex-wrap gap-1">{flags.map(n => <span key={n} className="text-[11px] px-2 py-0.5 rounded-full bg-slate-800 border border-slate-600">{n}{by[n].type === 'ordinal' ? ` ${vals[n]}` : ''}</span>)}</div></div>}
    </section>)
}
