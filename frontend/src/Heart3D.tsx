import { useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import type { Result, TargetName } from './api'

const COLORS: Record<string, string> = { LOW: '#22c55e', MODERATE: '#f59e0b', HIGH: '#ef4444' }
const Z = 0.63 // just in front of the heart's front surface
type V = 'LAD' | 'LCX' | 'RCA'
// Stylised, schematic coronary paths (NOT anatomical lesion locations).
const PATHS: Record<V, [number, number][]> = {
  LAD: [[-0.05, 0.62], [-0.12, 0.25], [-0.12, -0.15], [-0.07, -0.5], [0.0, -0.8]],
  LCX: [[-0.2, 0.6], [-0.5, 0.5], [-0.75, 0.2], [-0.75, -0.1], [-0.6, -0.4]],
  RCA: [[0.2, 0.6], [0.5, 0.5], [0.75, 0.2], [0.75, -0.1], [0.55, -0.45]],
}

function heartGeometry() {
  const s = new THREE.Shape()
  s.moveTo(0, -1.2)
  s.bezierCurveTo(-0.6, -0.7, -1.3, -0.2, -1.3, 0.4)
  s.bezierCurveTo(-1.3, 1.0, -0.5, 1.3, 0, 0.6)
  s.bezierCurveTo(0.5, 1.3, 1.3, 1.0, 1.3, 0.4)
  s.bezierCurveTo(1.3, -0.2, 0.6, -0.7, 0, -1.2)
  const g = new THREE.ExtrudeGeometry(s, { depth: 0.2, bevelEnabled: true, bevelThickness: 0.5, bevelSize: 0.4, bevelOffset: -0.4, bevelSegments: 18, curveSegments: 48 })
  g.center(); g.computeVertexNormals()
  return g
}

function Vessel({ name, color, selected, hasResult, prob, onSel }: { name: V; color: string; selected: boolean; hasResult: boolean; prob?: number; onSel: () => void }) {
  const curve = useMemo(() => new THREE.CatmullRomCurve3(PATHS[name].map(([x, y]) => new THREE.Vector3(x, y, Z))), [name])
  const geo = useMemo(() => new THREE.TubeGeometry(curve, 48, selected ? 0.075 : 0.05, 14, false), [curve, selected])
  const mid = curve.getPoint(0.5)
  const [hover, setHover] = useState(false)
  return (
    <group>
      <mesh geometry={geo} onClick={e => { e.stopPropagation(); onSel() }}
        onPointerOver={e => { e.stopPropagation(); setHover(true); document.body.style.cursor = 'pointer' }}
        onPointerOut={() => { setHover(false); document.body.style.cursor = 'auto' }}>
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={selected ? 1.4 : hover ? 0.7 : hasResult ? 0.25 : 0.05} roughness={0.35} />
      </mesh>
      <Html position={[mid.x + (name === 'LCX' ? -0.28 : name === 'RCA' ? 0.28 : 0.2), mid.y, Z + 0.05]} center style={{ pointerEvents: 'none' }}>
        <div className={`text-[11px] font-semibold whitespace-nowrap px-1.5 py-0.5 rounded bg-slate-950/70 ${selected ? 'text-white' : 'text-slate-300'}`}>
          {name}{prob !== undefined ? ` ${(prob * 100).toFixed(0)}%` : ''}</div>
      </Html>
    </group>)
}

function Scene({ r, sel, onSel }: { r: Result | null; sel: TargetName; onSel: (t: TargetName) => void }) {
  const body = useMemo(heartGeometry, [])
  const grp = useRef<THREE.Group>(null)
  useFrame(({ clock }) => { if (grp.current) grp.current.scale.setScalar(1 + 0.012 * Math.sin(clock.elapsedTime * 2.2)) }) // subtle pulse
  const cadColor = r ? COLORS[r.cad.risk_category] : '#475569'
  return (
    <group ref={grp}>
      <mesh geometry={body} onClick={() => onSel('CAD')}>
        <meshStandardMaterial color="#5b2a3a" roughness={0.55} metalness={0.05} emissive={cadColor} emissiveIntensity={sel === 'CAD' && r ? 0.18 : 0.04} />
      </mesh>
      <mesh position={[0, 1.0, -0.05]} rotation={[0, 0, 0.15]}><cylinderGeometry args={[0.18, 0.22, 0.6, 24]} /><meshStandardMaterial color="#7a3a4c" roughness={0.5} /></mesh>
      {(['LAD', 'LCX', 'RCA'] as V[]).map(v => (
        <Vessel key={v} name={v} selected={sel === v} hasResult={!!r} prob={r?.vessels[v].probability}
          color={r ? COLORS[r.vessels[v].risk_category] : '#64748b'} onSel={() => onSel(v)} />))}
    </group>)
}

export default function Heart3D({ r, sel, onSel }: { r: Result | null; sel: TargetName; onSel: (t: TargetName) => void }) {
  const ctl = useRef<any>(null)
  return (
    <div className="relative w-full h-full min-h-[420px]">
      <Canvas camera={{ position: [0, 0.2, 4.2], fov: 42 }} dpr={[1, 2]}>
        <ambientLight intensity={0.7} /><directionalLight position={[3, 4, 5]} intensity={1.6} /><pointLight position={[-4, -2, 3]} intensity={0.8} color="#60a5fa" />
        <Scene r={r} sel={sel} onSel={onSel} />
        <OrbitControls ref={ctl} enablePan={false} enableDamping minDistance={2.2} maxDistance={7} />
      </Canvas>
      <button onClick={() => ctl.current?.reset()} className="absolute top-2 right-2 text-[11px] px-2 py-1 rounded bg-slate-800/80 border border-slate-600 hover:border-cyan-500">Reset view</button>
      <div className="absolute bottom-2 left-2 flex gap-2 text-[10px] text-slate-400">
        {Object.entries(COLORS).map(([k, c]) => <span key={k} className="flex items-center gap-1"><i className="inline-block w-2 h-2 rounded-full" style={{ background: c }} />{k.toLowerCase()}</span>)}</div>
      <p className="absolute bottom-2 right-2 text-[10px] text-slate-500 max-w-[55%] text-right">Drag to rotate · scroll to zoom · click a vessel. Shows predicted vessel-level risk, not an anatomical lesion.</p>
    </div>)
}
