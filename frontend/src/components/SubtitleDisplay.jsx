export default function SubtitleDisplay({ text, lastLabel, confidence, accepted }) {
  return (
    <div className="card" style={{ minHeight: 130 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:12 }}>
        <span style={{ fontWeight:700, color:'#fff', fontSize:15, fontFamily: 'var(--font-display)' }}>📝 Translation Teleprompter</span>
        {lastLabel && (
          <div style={{ display:'flex', gap:8, alignItems:'center' }}>
            <span style={{
              background: accepted ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
              color:      accepted ? 'var(--neon-emerald)' : 'var(--neon-rose)',
              border: `1px solid ${accepted ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
              padding:'3px 12px', borderRadius:999, fontSize:13, fontWeight:700,
              boxShadow: accepted ? '0 0 10px rgba(16, 185, 129, 0.2)' : 'none'
            }}>
              {lastLabel}
            </span>
            <span style={{ fontSize:12, color:'var(--text-muted)' }}>
              {(confidence * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {/* Subtitle text area */}
      <div style={{
        minHeight: 76,
        background: '#070A13',
        border: '1px solid var(--border-glass)',
        borderRadius: 8,
        padding: '16px 20px',
        fontSize: 22,
        fontWeight: 600,
        color: text ? 'var(--text-main)' : 'var(--text-muted)',
        letterSpacing: 2,
        wordBreak: 'break-all',
        lineHeight: 1.5,
        fontFamily: 'var(--font-body)',
        display: 'flex',
        alignItems: 'center'
      }}>
        {text || 'Start signing or press start...'}
        {text && <span style={{ animation:'blink 1.2s infinite', marginLeft:4, color: 'var(--neon-cyan)', fontWeight: 800 }}>|</span>}
      </div>

      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
      `}</style>
    </div>
  )
}
