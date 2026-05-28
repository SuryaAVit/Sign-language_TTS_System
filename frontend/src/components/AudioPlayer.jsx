import { useRef, useEffect } from 'react'

export default function AudioPlayer({ src }) {
  const audioRef = useRef(null)

  useEffect(() => {
    if (src && audioRef.current) {
      audioRef.current.load()
      audioRef.current.play().catch(e => console.warn('[Audio]', e))
    }
  }, [src])

  if (!src) return null

  return (
    <div className="card" style={{ display:'flex', flexDirection:'column', gap:10 }}>
      <span style={{ fontWeight:700, color:'#fff', fontSize:15, fontFamily: 'var(--font-display)' }}>🔊 Voice Synthesizer Output</span>
      <audio
        ref={audioRef}
        controls
        style={{ width:'100%', height:40, borderRadius:8, background: '#1F2937' }}
      >
        <source src={src} type="audio/mpeg" />
      </audio>
    </div>
  )
}
