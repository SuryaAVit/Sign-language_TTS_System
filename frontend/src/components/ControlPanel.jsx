export default function ControlPanel({
  active, onStart, onStop, onSpace, onClear, onSpeak,
  onExportPDF, onExportWord, loading, text
}) {
  return (
    <div className="card" style={{ display:'flex', flexDirection:'column', gap:14 }}>
      <span style={{ fontWeight:700, color:'#fff', fontSize:15, fontFamily: 'var(--font-display)', marginBottom: 2 }}>🎛️ Dashboard Controls</span>

      {/* Start / Stop */}
      <div style={{ display:'flex', gap:10 }}>
        <button
          className="success"
          style={{ flex:1, fontSize:14, padding:'12px 0' }}
          onClick={onStart}
          disabled={active || loading}
        >
          ▶ Start Stream
        </button>
        <button
          className="danger"
          style={{ flex:1, fontSize:14, padding:'12px 0' }}
          onClick={onStop}
          disabled={!active || loading}
        >
          ■ Stop Stream
        </button>
      </div>

      {/* Text controls */}
      <div style={{ display:'flex', gap:10 }}>
        <button className="secondary" style={{ flex:1 }} onClick={onSpace} disabled={!active}>
          ␣ Spacebar
        </button>
        <button className="secondary" style={{ flex:1 }} onClick={onClear}>
          ✕ Clear Text
        </button>
      </div>

      {/* Speak */}
      <button
        className="primary"
        style={{ width:'100%', fontSize:14, padding:'12px 0' }}
        onClick={onSpeak}
        disabled={!text.trim() || loading}
      >
        🔊 Speak Transcript
      </button>

      {/* Export */}
      <div style={{ borderTop:'1px solid var(--border-glass)', paddingTop:14 }}>
        <p style={{ fontSize:11, color:'var(--text-muted)', marginBottom:10, fontWeight:700, letterSpacing: '0.5px' }}>
          EXPORT UTILITIES
        </p>
        <div style={{ display:'flex', gap:10 }}>
          <button
            className="secondary"
            style={{ flex:1, padding: '10px 0' }}
            onClick={onExportPDF}
            disabled={!text.trim() || loading}
          >
            📄 PDF Report
          </button>
          <button
            className="secondary"
            style={{ flex:1, padding: '10px 0' }}
            onClick={onExportWord}
            disabled={!text.trim() || loading}
          >
            📝 Word Doc
          </button>
        </div>
      </div>
    </div>
  )
}
