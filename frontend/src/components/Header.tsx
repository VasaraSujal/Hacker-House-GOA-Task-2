import { FiActivity, FiDatabase } from 'react-icons/fi'

interface HeaderProps {
  connected: boolean | null
}

export function Header({ connected }: HeaderProps) {
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
      <div className="flex items-center gap-2 text-xs text-slate-400" aria-live="polite">
        <FiActivity className={connected ? 'text-emerald-400' : connected === false ? 'text-rose-400' : ''} />
        <span>{connected ? 'Backend connected' : connected === false ? 'Backend unavailable' : 'Checking backend'}</span>
      </div>
    </header>
  )
}
