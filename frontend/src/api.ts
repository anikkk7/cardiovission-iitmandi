const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'
export type TargetName = 'CAD' | 'LAD' | 'LCX' | 'RCA'
export interface Pred { probability: number; risk_category: string; predicted_positive: boolean; decision_threshold: number; model: string; cv_roc_auc: number }
export interface Contribution { feature: string; value: number; imputed: boolean; shap_value: number; direction: string }
export interface Expl { unit: string; base_value: number; top_features: Contribution[] }
export interface Result { cad: Pred; vessels: Record<string, Pred>; explanations: Record<TargetName, Expl>;
  input_summary: { n_features_total: number; n_provided: number; n_imputed: number; imputed_features: string[] }; disclaimer: string; visualization_note: string }
export interface Feature { name: string; type: 'binary' | 'numeric' | 'ordinal'; min: number; max: number; median: number; p25: number; p75: number }
export interface Demo { dataset_row: number; features: Record<string, number>; true_labels_for_reference_only: Record<string, number> }
export interface Metrics { metrics: { targets: Record<string, any> }; global_importance: Record<string, { unit: string; features: { feature: string; mean_abs_shap: number }[] }>;
  risk_bands: { low_below: number; high_from: number } }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, init)
  if (!r.ok) { const b = await r.json().catch(() => ({})); throw new Error(typeof b.detail === 'string' ? b.detail : Array.isArray(b.detail) ? b.detail.map((d: any) => String(d.msg ?? d).replace(/^Value error, /, '')).join('; ') : r.statusText) }
  return r.json()
}
export const api = {
  schema: () => req<{ features: Feature[] }>('/feature-schema'),
  demos: () => req<Record<string, Demo>>('/demo-patients'),
  metrics: () => req<Metrics>('/model-metrics'),
  predict: (features: Record<string, number>) => req<Result>('/predict?top_k=10', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features }) }),
}
