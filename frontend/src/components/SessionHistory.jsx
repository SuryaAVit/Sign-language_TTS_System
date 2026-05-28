export default function SessionHistory({ history }) {
  if (!history.length) return (
    <div className="card" style={{ minHeight:120 }}>
      <span style={{ fontWeight:700, color:'#fff', fontSize:15, fontFamily: 'var(--font-display)' }}>📋 Translation Timeline</span>
      <p style={{ color:'var(--text-muted)', fontSize:13, marginTop:20, textAlign:'center' }}>
        Timeline is empty. Begin signing to log gestures.
      </p>
    </div>
  )

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:14 }}>
        <span style={{ fontWeight:700, color:'#fff', fontSize:15, fontFamily: 'var(--font-display)' }}>📋 Translation Timeline</span>
        <span className="badge blue">{history.length} signs</span>
      </div>
      
      <div style={{ maxHeight:240, overflowY:'auto', display:'flex', flexDirection:'column', gap:8, paddingRight: 4 }}>
        {[...history].reverse().map((item, i) => (
          <div key={i} style={{
            display:'flex', justifyContent:'space-between', alignItems:'center',
            padding:'8px 14px',
            background: i % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'rgba(255,255,255,0.01)',
            borderRadius:8,
            border:'1px solid var(--border-glass)'
          }}>
            <span style={{ fontWeight:800, fontSize:18, color:'var(--neon-cyan)', minWidth:32 }}>
              {item.label}
            </span>
            <span style={{
              fontSize:12, fontWeight:700,
              color: item.confidence >= 0.85 ? '#34D399' : '#FBBF24'
            }}>
              {(item.confidence * 100).toFixed(0)}% accuracy
            </span>
            <span style={{ fontSize:11, color:'var(--text-muted)' }}>{item.time}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
