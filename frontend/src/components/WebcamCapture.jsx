import { useRef, useEffect, useCallback } from 'react'

const FRAME_INTERVAL_MS = 500  // 1 frame per 500ms — prevents Flask overload

export default function WebcamCapture({ active, onFrame, showAnnotated, annotatedFrame }) {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef  = useRef(null)
  const sendingRef = useRef(false)  // prevent overlapping requests

  useEffect(() => {
    if (!active) { stopStream(); return }
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then(stream => {
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play().catch(() => {})
        }
      })
      .catch(err => console.error('[Webcam]', err))
    return () => stopStream()
  }, [active])

  useEffect(() => {
    if (!active) { clearInterval(timerRef.current); return }
    timerRef.current = setInterval(captureFrame, FRAME_INTERVAL_MS)
    return () => clearInterval(timerRef.current)
  }, [active, onFrame])

  const stopStream = () => {
    clearInterval(timerRef.current)
    sendingRef.current = false
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) videoRef.current.srcObject = null
  }

  const captureFrame = useCallback(() => {
    if (sendingRef.current) return  // skip if previous frame still processing
    const video  = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    const b64 = canvas.toDataURL('image/jpeg', 0.6)

    sendingRef.current = true
    Promise.resolve(onFrame(b64)).finally(() => { sendingRef.current = false })
  }, [onFrame])

  return (
    <div style={{ position:'relative', width:'100%', borderRadius:10, overflow:'hidden', background:'#0D1B2A', minHeight:260 }}>

      {/* Always-visible video element — camera works regardless of backend */}
      <video
        ref={videoRef}
        autoPlay playsInline muted
        style={{
          width:'100%',
          display: active ? 'block' : 'none',
          transform:'scaleX(-1)',
          borderRadius:10,
        }}
      />

      {showAnnotated && annotatedFrame && (
        <img
          src={`data:image/jpeg;base64,${annotatedFrame}`}
          alt="annotated"
          style={{ position:'absolute', top:0, left:0, width:'100%', opacity:0.5, borderRadius:10 }}
        />
      )}

      {!active && (
        <div style={{ display:'flex', flexDirection:'column', alignItems:'center',
          justifyContent:'center', height:280, color:'#64748B', gap:12 }}>
          <span style={{ fontSize:52 }}>📷</span>
          <p style={{ fontSize:14 }}>Camera is off — click <strong>Start</strong> to begin</p>
        </div>
      )}

      {active && (
        <div style={{ position:'absolute', top:12, right:12,
          background:'rgba(220,38,38,0.85)', color:'#fff',
          borderRadius:999, padding:'4px 12px', fontSize:12, fontWeight:700,
          display:'flex', alignItems:'center', gap:6 }}>
          <span style={{ width:8, height:8, borderRadius:'50%', background:'#fff',
            animation:'pulse 1s infinite' }} />
          LIVE
        </div>
      )}

      <canvas ref={canvasRef} style={{ display:'none' }} />
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
    </div>
  )
}