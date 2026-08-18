import { FiActivity, FiCheckCircle, FiDatabase } from 'react-icons/fi'
import type { SystemStatus } from '../types/rag'

interface HeaderProps {
  status: SystemStatus
}

export function Header({ status }: HeaderProps) {
  return (
    <header className="flex items-center justify-between border-b border-white/10 py-5">
      <div className="flex items-center gap-3">
        <span className="grid size-10 place-items-center rounded-xl border border-cyan-300/25 bg-cyan-300/10 text-cyan-200">
          <FiDatabase aria-hidden="true" />
        </span>
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-cyan-300">HH Goa 2026</p>
          <p className="text-sm font-semibold text-white">Voice RAG</p>
        </div>
      </div>
      <div className="flex items-center gap-2 text-xs" aria-live="polite">
        {status === 'ready' && (
          <>
            <FiCheckCircle className="size-4 text-emerald-400" aria-hidden="true" />
            <span className="font-medium text-emerald-300">System ready</span>
          </>
        )}
        {status === 'warming' && (
          <>
            <FiActivity className="size-4 animate-pulse text-cyan-300" aria-hidden="true" />
            <span className="text-cyan-200">Warming up…</span>
          </>
        )}
        {status === 'preparing' && (
          <>
            <FiActivity className="size-4 text-slate-400" aria-hidden="true" />
            <span className="text-slate-400">Preparing…</span>
          </>
        )}
        {status === 'degraded' && (
          <>
            <FiActivity className="size-4 text-amber-400" aria-hidden="true" />
            <span className="text-amber-200">Backend connected</span>
          </>
        )}
        {status === 'unavailable' && (
          <>
            <FiActivity className="size-4 text-rose-400" aria-hidden="true" />
            <span className="text-rose-300">Backend unavailable</span>
          </>
        )}
      </div>
    </header>
  )
}

