'use client'

import { useState } from 'react'
import { CourseCard, type Recommendation } from '@/components/CourseCard'
import { BubbleChart, type SimilarCourse } from '@/components/BubbleChart'

interface ApiResponse {
  recommendations: Recommendation[]
  similar_courses: SimilarCourse[]
  preferred_credit_level: string
  error?: string
}

function LoadingDots() {
  return (
    <div className="text-center py-16">
      <div className="inline-flex gap-1.5 mb-4">
        {[0, 150, 300].map((delay) => (
          <div
            key={delay}
            className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
      <p className="text-slate-400 text-sm">Analyzing your interests and generating recommendations...</p>
    </div>
  )
}

export default function Home() {
  const [userInput, setUserInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ApiResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!userInput.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: userInput }),
      })

      const data: ApiResponse = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Something went wrong. Please try again.')
      }

      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit()
    }
  }

  return (
    <main className="min-h-screen bg-[#070712] text-white">
      <div className="max-w-3xl mx-auto px-6 py-16">

        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-violet-400 to-blue-400 bg-clip-text text-transparent mb-3">
            Course Recommendation System
          </h1>
          <p className="text-slate-400 text-base leading-relaxed max-w-xl mx-auto">
            Describe your interests and preferred workload to discover university courses tailored for you.
          </p>
        </div>

        {/* Input Card */}
        <div className="bg-white/[0.03] border border-white/[0.08] rounded-2xl p-6 mb-5">
          <label className="block text-sm font-medium text-slate-300 mb-3">
            What are you interested in?
          </label>
          <textarea
            className="w-full bg-white/[0.04] border border-white/[0.1] rounded-xl px-4 py-3 text-white placeholder-slate-600 resize-none focus:outline-none focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/20 transition-all text-sm leading-relaxed"
            rows={4}
            placeholder="Example: I enjoy biology and data analysis but want a lighter workload this semester..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <div className="flex items-center justify-between mt-3">
            <span className="text-slate-600 text-xs">⌘ + Enter to submit</span>
            <button
              onClick={handleSubmit}
              disabled={loading || !userInput.trim()}
              className="px-6 py-2.5 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed rounded-xl font-semibold text-white text-sm transition-all duration-200 shadow-lg shadow-purple-900/20"
            >
              {loading ? 'Finding courses...' : 'Get Recommendations'}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 text-red-400 text-sm mb-6">
            {error}
          </div>
        )}

        {/* Loading */}
        {loading && <LoadingDots />}

        {/* Results */}
        {result && !loading && (
          <div className="space-y-10">

            {/* Recommendations */}
            <section>
              <div className="mb-5">
                <h2 className="text-xl font-semibold text-white">Recommended Courses</h2>
                <p className="text-slate-400 text-sm mt-1">
                  Best matches for a{' '}
                  <span className="text-purple-400 font-medium">{result.preferred_credit_level}</span>{' '}
                  workload based on your interests.
                </p>
              </div>

              {result.recommendations.length === 0 ? (
                <div className="bg-white/[0.03] border border-white/[0.08] rounded-2xl px-6 py-8 text-center text-slate-400 text-sm">
                  No recommendations found. Try being more specific about your interests or workload preference.
                </div>
              ) : (
                <div className="space-y-4">
                  {result.recommendations.map((rec, i) => (
                    <CourseCard key={rec.key} course={rec} index={i} />
                  ))}
                </div>
              )}
            </section>

            {/* Chart */}
            {result.similar_courses.length > 0 && (
              <section>
                <div className="mb-5">
                  <h2 className="text-xl font-semibold text-white">Course Landscape</h2>
                  <p className="text-slate-400 text-sm mt-1">
                    All similar courses plotted by interest match and predicted workload. The{' '}
                    <span className="text-purple-400 font-medium">{result.preferred_credit_level}</span>{' '}
                    band highlights your preferred level.
                  </p>
                </div>
                <BubbleChart
                  similarCourses={result.similar_courses}
                  recommendedKeys={result.recommendations.map((r) => r.key)}
                  preferredLevel={result.preferred_credit_level}
                />
              </section>
            )}

          </div>
        )}
      </div>
    </main>
  )
}
