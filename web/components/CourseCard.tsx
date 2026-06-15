export interface Recommendation {
  key: string
  title: string
  description: string
  minimum_credits: number
  similarity: number
  reasoning: string
}

interface Props {
  course: Recommendation
  index: number
}

export function CourseCard({ course, index }: Props) {
  const matchPct = Math.round(course.similarity * 100)

  return (
    <div
      className="bg-white/[0.03] border border-white/[0.08] rounded-2xl p-6 hover:border-purple-500/30 hover:-translate-y-0.5 transition-all duration-200"
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <div className="flex items-start justify-between gap-4 mb-5">
        <div className="min-w-0">
          <span className="text-purple-400 font-mono text-xs font-semibold tracking-wider">
            {course.key}
          </span>
          <h3 className="text-white text-lg font-semibold mt-1 leading-snug">
            {course.title}
          </h3>
        </div>
        <div className="flex flex-col gap-1.5 shrink-0 items-end">
          <span className="bg-slate-800 text-slate-300 text-xs px-3 py-1 rounded-full whitespace-nowrap">
            {course.minimum_credits} credits
          </span>
          <span className="bg-purple-500/10 text-purple-300 text-xs px-3 py-1 rounded-full border border-purple-500/20 whitespace-nowrap">
            {matchPct}% match
          </span>
        </div>
      </div>

      <div className="mb-4">
        <p className="text-slate-500 text-xs uppercase tracking-wider font-medium mb-2">
          Description
        </p>
        <p className="text-slate-300 text-sm leading-relaxed">{course.description}</p>
      </div>

      <div className="pt-4 border-t border-white/[0.06]">
        <p className="text-slate-500 text-xs uppercase tracking-wider font-medium mb-2">
          Why this course?
        </p>
        <p className="text-slate-300 text-sm leading-relaxed">{course.reasoning}</p>
      </div>
    </div>
  )
}
