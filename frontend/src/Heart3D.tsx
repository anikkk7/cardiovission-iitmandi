import { useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html, ContactShadows } from '@react-three/drei'
import * as THREE from 'three'
import { ParametricGeometry } from 'three/examples/jsm/geometries/ParametricGeometry.js'
import type { Result, TargetName } from './api'

const COLORS: Record<string, string> = { LOW: '#22c55e', MODERATE: '#f59e0b', HIGH: '#ef4444' }
type V = 'LAD' | 'LCX' | 'RCA'
const Y0 = -1.15, H = 2.05

// Ventricular surface: theta around the long axis (anterior = +z), t from apex (0) to base (1).
const prof = (t: number) => 0.98 * Math.pow(Math.max(Math.sin(Math.PI * Math.pow(t, 1.3)), 0), 0.6) * (1 - 0.12 * t)
export function surf(th: number, t: number, off = 0, out = new THREE.Vector3()) {
  const r = prof(t) * (1 + 0.025 * Math.sin(3 * th + 5 * t) + 0.015 * Math.sin(7 * th - 3 * t)) + (t > 0.01 && t < 0.99 ? off : 0)
  return out.set(r * Math.cos(th), Y0 + H * t, r * 0.82 * Math.sin(th))
}
// Coronary paths in (theta, t) space, so they lie ON the surface. Anterior view: patient's right = viewer's left.
const PATHS: Record<V, [number, number][]> = {
  LAD: [[1.3, 0.86], [1.34, 0.7], [1.4, 0.5], [1.45, 0.3], [1.5, 0.14], [1.62, 0.05]],            // anterior interventricular groove
  LCX: [[1.2, 0.88], [0.85, 0.87], [0.5, 0.84], [0.15, 0.79], [-0.2, 0.74], [-0.55, 0.68]],       // left AV groove (wraps posteriorly)
  RCA: [[1.9, 0.9], [2.3, 0.88], [2.75, 0.85], [3.2, 0.8], [3.7, 0.74], [4.2, 0.66]],             // right AV groove (wraps posteriorly)
}
const ROT_Y: Record<TargetName, number> = { CAD: 0, LAD: 0, RCA: 0.55, LCX: -0.55 }
const curve = (pts: [number, number, number][]) => new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(...p)))

function Tube({ pts, r, color, rough = 0.5 }: { pts: [number, number, number][]; r: number; color: string; rough?: number }) {
  const g = useMemo(() => new THREE.TubeGeometry(curve(pts), 64, r, 20, false), [pts, r])
  return <mesh geometry={g}><meshPhysicalMaterial color={color} roughness={rough} clearcoat={0.4} clearcoatRoughness={0.4} /></mesh>
}
const Blob = ({ p, s, color }: { p: [number, number, number]; s: [number, number, number]; color: string }) =>
  <mesh position={p} scale={s}><sphereGeometry args={[1, 40, 32]} /><meshPhysicalMaterial color={color} roughness={0.55} clearcoat={0.4} clearcoatRoughness={0.45} /></mesh>

function Vessel({ name, color, selected, hasResult, prob, onSel }: { name: V; color: string; selected: boolean; hasResult: boolean; prob?: number; onSel: () => void }) {
  const c = useMemo(() => curve(PATHS[name].map(([th, t]) => surf(th, t, 0.03).toArray() as [number, number, number])), [name])
  const geo = useMemo(() => new THREE.TubeGeometry(c, 120, selected ? 0.062 : 0.042, 16, false), [c, selected])
  const hit = useMemo(() => new THREE.TubeGeometry(c, 40, 0.12, 8, false), [c])
  const mat = useRef<THREE.MeshStandardMaterial>(null); const [hover, setHover] = useState(false)
  useFrame(({ clock }) => { if (mat.current) mat.current.emissiveIntensity = selected ? 1.1 + 0.35 * Math.sin(clock.elapsedTime * 4) : hover ? 0.8 : hasResult ? 0.3 : 0.05 })
  const lp = surf(PATHS[name][2][0], PATHS[name][2][1], 0.2)
  return (
    <group>
      <mesh geometry={geo}><meshStandardMaterial ref={mat} color={color} emissive={color} roughness={0.35} /></mesh>
      <mesh geometry={hit} onClick={e => { e.stopPropagation(); onSel() }} onPointerOver={e => { e.stopPropagation(); setHover(true); document.body.style.cursor = 'pointer' }}
        onPointerOut={() => { setHover(false); document.body.style.cursor = 'auto' }}><meshBasicMaterial transparent opacity={0} depthWrite={false} /></mesh>
      <Html position={lp} center style={{ pointerEvents: 'none' }}>
        <div className={`text-[11px] font-semibold whitespace-nowrap px-1.5 py-0.5 rounded border bg-slate-950/80 ${selected ? 'text-white' : 'text-slate-300'}`} style={{ borderColor: hasResult ? color : '#475569' }}>
          {name}{prob !== undefined ? ` ${(prob * 100).toFixed(0)}%` : ''}</div></Html>
    </group>)
}

function Scene({ r, sel, onSel }: { r: Result | null; sel: TargetName; onSel: (t: TargetName) => void }) {
  const body = useMemo(() => new ParametricGeometry((u, v, t) => { surf(-u * Math.PI * 2, v, 0, t) }, 140, 90), [])
  const beat = useRef<THREE.Group>(null); const turn = useRef<THREE.Group>(null)
  useFrame(({ clock }, dt) => {
    const ph = clock.elapsedTime % 1.1, b = 0.6 * Math.exp(-(((ph - 0.1) / 0.05) ** 2)) + 0.4 * Math.exp(-(((ph - 0.32) / 0.06) ** 2))
    beat.current?.scale.setScalar(1 + 0.016 * b)                       // restrained lub-dub
    if (turn.current) turn.current.rotation.y += (ROT_Y[sel] - turn.current.rotation.y) * Math.min(1, dt * 3) // bring selected vessel toward camera
  })
  const tint = r ? COLORS[r.cad.risk_category] : '#000000'
  return (
    <group ref={turn}><group ref={beat} rotation={[0, 0, 0.42]} position={[-0.1, -0.15, 0]}>
      <mesh geometry={body} onClick={() => onSel('CAD')}>
        <meshPhysicalMaterial color="#9a2f3d" roughness={0.5} clearcoat={0.55} clearcoatRoughness={0.4} sheen={0.4} sheenColor={new THREE.Color('#ff8a8a')}
          emissive={tint} emissiveIntensity={r && sel === 'CAD' ? 0.12 : 0.02} side={THREE.DoubleSide} /></mesh>
      <Blob p={[-0.72, 0.62, 0.08]} s={[0.4, 0.52, 0.4]} color="#7e2634" />{/* right atrium */}
      <Blob p={[0.62, 0.86, 0.3]} s={[0.22, 0.28, 0.18]} color="#7e2634" />{/* left atrial appendage */}
      <Blob p={[0.05, 0.95, -0.35]} s={[0.7, 0.36, 0.36]} color="#7a2633" />{/* left atrium (posterior) */}
      <Tube pts={[[-0.1, 0.8, 0.02], [-0.14, 1.35, 0.0], [0.0, 1.85, -0.05], [0.4, 2.1, -0.18], [0.85, 1.85, -0.28], [0.95, 1.2, -0.3], [0.95, 0.5, -0.32]]} r={0.24} color="#b4655e" />{/* aorta */}
      <Tube pts={[[0.22, 0.85, 0.38], [0.3, 1.3, 0.28], [0.26, 1.65, 0.1]]} r={0.21} color="#6b86b3" />{/* pulmonary trunk */}
      <Tube pts={[[0.26, 1.65, 0.1], [0.6, 1.8, -0.1], [1.0, 1.85, -0.25]]} r={0.13} color="#6b86b3" />
      <Tube pts={[[0.26, 1.65, 0.1], [-0.2, 1.75, -0.3], [-0.8, 1.7, -0.4]]} r={0.13} color="#6b86b3" />
      <Tube pts={[[-0.68, 1.0, 0.0], [-0.7, 1.5, -0.08], [-0.68, 2.0, -0.12]]} r={0.17} color="#5a78a8" />{/* SVC */}
      {(['LAD', 'LCX', 'RCA'] as V[]).map(v => (
        <Vessel key={v} name={v} selected={sel === v} hasResult={!!r} prob={r?.vessels[v].probability} color={r ? COLORS[r.vessels[v].risk_category] : '#94a3b8'} onSel={() => onSel(v)} />))}
    </group></group>)
}

export default function Heart3D({ r, sel, onSel }: { r: Result | null; sel: TargetName; onSel: (t: TargetName) => void }) {
  const ctl = useRef<any>(null); const [auto, setAuto] = useState(false)
  return (
    <div className="relative w-full h-full min-h-[440px]">
      <Canvas camera={{ position: [0.2, 0.3, 5.4], fov: 40 }} dpr={[1, 2]}>
        <hemisphereLight args={['#ffe8e8', '#1e293b', 0.9]} /><directionalLight position={[3, 4, 5]} intensity={2.0} /><directionalLight position={[-4, 1, -3]} intensity={0.9} color="#7dd3fc" />
        <Scene r={r} sel={sel} onSel={onSel} />
        <ContactShadows position={[0, -1.7, 0]} opacity={0.35} scale={7} blur={2.6} far={3} />
        <OrbitControls ref={ctl} makeDefault enablePan={false} enableDamping autoRotate={auto} autoRotateSpeed={1.2} minDistance={2.5} maxDistance={9} target={[0, 0.35, 0]} />
      </Canvas>
      <div className="absolute top-2 right-2 flex gap-1">
        <button onClick={() => setAuto(!auto)} className={`text-[11px] px-2 py-1 rounded bg-slate-800/80 border ${auto ? 'border-cyan-400' : 'border-slate-600'}`}>Auto-rotate</button>
        <button onClick={() => ctl.current?.reset()} className="text-[11px] px-2 py-1 rounded bg-slate-800/80 border border-slate-600 hover:border-cyan-500">Reset view</button></div>
      <div className="absolute bottom-2 left-2 flex gap-2 text-[10px] text-slate-400">
        {Object.entries(COLORS).map(([k, c]) => <span key={k} className="flex items-center gap-1"><i className="inline-block w-2 h-2 rounded-full" style={{ background: c }} />{k.toLowerCase()}</span>)}</div>
      <p className="absolute bottom-2 right-2 text-[10px] text-slate-500 max-w-[58%] text-right">Anterior view (patient's right = viewer's left). Drag to rotate · scroll to zoom · click a vessel. Stylised heart showing predicted vessel-level risk, not an anatomical lesion.</p>
    </div>)
}
