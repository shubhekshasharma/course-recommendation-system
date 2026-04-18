'use client'

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'

const WORKLOAD_TO_NUM: Record<string, number> = { High: 2, Standard: 1, Low: 0 }
const NUM_TO_WORKLOAD: Record<number, string> = { 2: 'High', 1: 'Standard', 0: 'Low' }

export interface SimilarCourse {
  key: string
  title: string
  similarity: number
  predicted_credit_level: string
  'minimum credits': number
}

interface Props {
  similarCourses: SimilarCourse[]
  recommendedKeys: string[]
  preferredLevel: string
}

interface ChartPoint {
  x: number
  y: number
  z: number
  key: string
  title: string
  credits: number
}

function toChartPoint(c: SimilarCourse): ChartPoint {
  return {
    x: Math.round(c.similarity * 100),
    y: WORKLOAD_TO_NUM[c.predicted_credit_level] ?? 1,
    z: Math.max(c['minimum credits'] || 1, 1) * 80,
    key: c.key,
    title: c.title,
    credits: c['minimum credits'],
  }
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: { payload: ChartPoint }[] }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-slate-900 border border-white/10 rounded-xl px-3 py-2.5 text-xs shadow-xl">
      <p className="text-white font-semibold mb-1">{d.key}</p>
      <p className="text-slate-300 mb-2 max-w-[180px] leading-relaxed">{d.title}</p>
      <p className="text-purple-400">Match: {d.x}%</p>
      <p className="text-slate-400">Workload: {NUM_TO_WORKLOAD[d.y]}</p>
      <p className="text-slate-400">Credits: {d.credits}</p>
    </div>
  )
}

export function BubbleChart({ similarCourses, recommendedKeys, preferredLevel }: Props) {
  const recommendedSet = new Set(recommendedKeys.map((k) => k.trim()))

  const recommended = similarCourses
    .filter((c) => recommendedSet.has(String(c.key).trim()))
    .map(toChartPoint)

  const similar = similarCourses
    .filter((c) => !recommendedSet.has(String(c.key).trim()))
    .map(toChartPoint)

  const preferredY = WORKLOAD_TO_NUM[preferredLevel] ?? 1

  return (
    <div className="bg-white/[0.03] border border-white/[0.08] rounded-2xl p-6">
      <ResponsiveContainer width="100%" height={380}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 40, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />

          <XAxis
            type="number"
            dataKey="x"
            name="Interest Match"
            domain={[0, 100]}
            label={{
              value: 'Interest Match (%)',
              position: 'bottom',
              offset: 20,
              fill: '#64748b',
              fontSize: 12,
            }}
            tick={{ fill: '#64748b', fontSize: 11 }}
            stroke="rgba(255,255,255,0.08)"
          />

          <YAxis
            type="number"
            dataKey="y"
            name="Workload"
            domain={[-0.5, 2.5]}
            ticks={[0, 1, 2]}
            tickFormatter={(v) => NUM_TO_WORKLOAD[v] ?? ''}
            label={{
              value: 'Predicted Workload',
              angle: -90,
              position: 'insideLeft',
              offset: 10,
              fill: '#64748b',
              fontSize: 12,
            }}
            tick={{ fill: '#64748b', fontSize: 11 }}
            stroke="rgba(255,255,255,0.08)"
          />

          <ZAxis type="number" dataKey="z" range={[60, 600]} />

          <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.08)' }} />

          <Legend
            verticalAlign="top"
            wrapperStyle={{ paddingBottom: '12px', fontSize: '12px', color: '#94a3b8' }}
          />

          {/* Highlight band for preferred workload level */}
          <ReferenceLine
            y={preferredY}
            stroke="rgba(139,92,246,0.25)"
            strokeWidth={36}
          />

          <Scatter name="Other Similar Courses" data={similar} fill="#60a5fa" fillOpacity={0.65} />
          <Scatter name="Recommended Courses" data={recommended} fill="#10b981" fillOpacity={0.9} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
