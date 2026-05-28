import { useState, useCallback, useEffect, useRef } from 'react'
import SubtitleDisplay from './components/SubtitleDisplay.jsx'
import ControlPanel    from './components/ControlPanel.jsx'
import AudioPlayer     from './components/AudioPlayer.jsx'
import SessionHistory  from './components/SessionHistory.jsx'
import DatasetProcessor from './components/DatasetProcessor.jsx'
import {
  startSession, endSession, recognizeFromLandmarks,
  updateTranscript, generateTTS, exportPDF, exportWord, getHistory
} from './api.js'

export default function App() {
  const videoRef       = useRef(null)
  const canvasRef      = useRef(null)
  const handsRef       = useRef(null)
  const cameraRef      = useRef(null)

  // Navigation tab: 'live' | 'processor' | 'history' | 'collector'
  const [activeTab,     setActiveTab]   = useState('live')
  const [sessionId,     setSessionId]   = useState(null)
  const [active,        setActive]      = useState(false)
  const [text,          setText]        = useState('')
  const [lastLabel,     setLastLabel]   = useState(null)
  const [confidence,    setConfidence]  = useState(0)
  const [history,       setHistory]     = useState([])
  const [audioSrc,      setAudioSrc]    = useState(null)
  const [loading,       setLoading]     = useState(false)
  const [status,        setStatus]      = useState('idle')
  const [toast,         setToast]       = useState(null)
  const [stats,         setStats]       = useState({ gestures: 0, words: 0 })
  const [backendOk,     setBackendOk]   = useState(null)
  const [backendModel,  setBackendModel] = useState('demo')
  const [handVisible,   setHandVisible]   = useState(false)
  
  // Custom webcam collection states
  const [trainingMode,  setTrainingMode]  = useState(false)
  const [trainingIdx,   setTrainingIdx]   = useState(0)
  const [trainingSamps, setTrainingSamps] = useState(0)
  const [trainingWait,  setTrainingWait]  = useState(false)
  const [trainingDone,  setTrainingDone]  = useState(false)

  const activeRef    = useRef(false)
  const sessionIdRef = useRef(null)
  const sendingRef   = useRef(false)

  const stableLabelRef  = useRef(null)
  const stableCountRef  = useRef(0)
  const lastAcceptRef   = useRef(0)
  const STABLE_FRAMES   = 5
  const COOLDOWN_MS     = 1200
  const transcriptTimer  = useRef(null)
  
  const trainingModeRef  = useRef(false)
  const trainingIdxRef   = useRef(0)
  const trainingSampRef  = useRef(0)
  const collectingRef    = useRef(false)
  
  const TRAIN_LETTERS    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
  const SAMPLES_PER_LTR  = 40

  // ── Health check helper ──────────────────────────────────────────────────
  const checkHealth = useCallback(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => {
        setBackendOk(true)
        setBackendModel(d.model)
        showToast(`Backend online · Model: ${d.model.toUpperCase()}`, 'green')
      })
      .catch(() => {
        setBackendOk(false)
        showToast('Backend offline — please start Flask!', 'red')
      })
  }, [])

  useEffect(() => {
    checkHealth()
  }, [checkHealth])

  // ── Auto-save transcript ─────────────────────────────────────────────────
  useEffect(() => {
    if (!sessionIdRef.current) return
    clearTimeout(transcriptTimer.current)
    transcriptTimer.current = setTimeout(() => {
      updateTranscript(sessionIdRef.current, text)
    }, 1500)
    return () => clearTimeout(transcriptTimer.current)
  }, [text])

  // ── Init MediaPipe Hands once ─────────────────────────────────────────────
  useEffect(() => {
    if (!window.Hands) {
      console.error('MediaPipe Hands not loaded — check index.html CDN scripts')
      return
    }
    const hands = new window.Hands({ locateFile: f =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${f}` })

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence:  0.7,
    })

    hands.onResults(onResults)
    handsRef.current = hands

    return () => { handsRef.current = null }
  }, [])

  // ── MediaPipe results callback ────────────────────────────────────────────
  const onResults = useCallback(async (results) => {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video) return

    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.save()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height)

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      setHandVisible(true)
      const landmarks = results.multiHandLandmarks[0]

      // Draw skeleton skeleton on canvas
      if (window.drawConnectors && window.HAND_CONNECTIONS) {
        window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS,
          { color: '#06B6D4', lineWidth: 3 })
        window.drawLandmarks(ctx, landmarks,
          { color: '#F43F5E', lineWidth: 1, radius: 4 })
      }

      // ── Custom webcam collection (runs instead of regular recognition) ──
      if (trainingModeRef.current && collectingRef.current) {
        const flat = landmarks.flatMap(lm => [lm.x, lm.y, lm.z])
        const wx = flat[0], wy = flat[1], wz = flat[2]
        const norm = []
        for (let i = 0; i < flat.length; i += 3)
          norm.push(flat[i]-wx, flat[i+1]-wy, flat[i+2]-wz)
          
        fetch('/api/training/collect', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ label: TRAIN_LETTERS[trainingIdxRef.current], landmarks: norm })
        }).catch(() => {})
        
        trainingSampRef.current++
        setTrainingSamps(trainingSampRef.current)
        if (trainingSampRef.current >= SAMPLES_PER_LTR) {
          collectingRef.current = false
          const ni = trainingIdxRef.current + 1
          if (ni >= TRAIN_LETTERS.length) {
            trainingModeRef.current = false
            setTrainingMode(false)
            setTrainingDone(true)
          } else {
            trainingIdxRef.current = ni
            trainingSampRef.current = 0
            setTrainingIdx(ni)
            setTrainingSamps(0)
            setTrainingWait(true)
            setTimeout(() => { collectingRef.current = true; setTrainingWait(false) }, 2500)
          }
        }
      }

      // ── Regular recognition ────────────────────────────────────────────────
      if (!trainingModeRef.current && activeRef.current && sessionIdRef.current && !sendingRef.current) {
        sendingRef.current = true
        try {
          const flat = landmarks.flatMap(lm => [lm.x, lm.y, lm.z])
          const wx = flat[0], wy = flat[1], wz = flat[2]
          const normalized = []
          for (let i = 0; i < flat.length; i += 3) {
            normalized.push(flat[i]-wx, flat[i+1]-wy, flat[i+2]-wz)
          }

          const result = await recognizeFromLandmarks(normalized, sessionIdRef.current)

          if (result && result.label) {
            const now = Date.now()
            setConfidence(result.confidence)
            setLastLabel(result.label)

            if (result.accepted) {
              // stable-frame check
              if (result.label === stableLabelRef.current) {
                stableCountRef.current++
              } else {
                stableLabelRef.current = result.label
                stableCountRef.current = 1
              }

              if (stableCountRef.current >= STABLE_FRAMES &&
                  now - lastAcceptRef.current >= COOLDOWN_MS) {
                lastAcceptRef.current  = now
                stableCountRef.current = 0

                setText(prev => {
                  let next = prev
                  if      (result.label === 'SPACE')  next = prev + ' '
                  else if (result.label === 'DELETE') next = prev.slice(0, -1)
                  else                                next = prev + result.label
                  setStats(s => ({ gestures: s.gestures+1, words: next.trim().split(/\s+/).filter(Boolean).length }))
                  return next
                })
                setHistory(prev => [...prev, {
                  label: result.label, confidence: result.confidence,
                  time: new Date().toLocaleTimeString()
                }])
              }
            }
          }
        } catch { /* ignore */ }
        finally { sendingRef.current = false }
      }
    } else {
      setHandVisible(false)
    }
    ctx.restore()
  }, [])

  // ── start/stop camera ────────────────────────────────────────────────────
  const startCamera = () => {
    if (!handsRef.current || !videoRef.current) return
    const camera = new window.Camera(videoRef.current, {
      onFrame: async () => {
        if (handsRef.current) {
          await handsRef.current.send({ image: videoRef.current })
        }
      },
      width: 640, height: 480
    })
    camera.start()
    cameraRef.current = camera
  }

  const stopCamera = () => {
    if (cameraRef.current) { cameraRef.current.stop(); cameraRef.current = null }
  }

  // ── session control ───────────────────────────────────────────────────────
  const handleStart = async () => {
    try {
      setLoading(true)
      const data = await startSession()
      setSessionId(data.session_id)
      sessionIdRef.current = data.session_id
      activeRef.current    = true
      setActive(true)
      setStatus('running')
      setText(''); setHistory([]); setStats({ gestures:0, words:0 }); setAudioSrc(null)
      startCamera()
      showToast('Session started — hold your hand in front of camera!', 'green')
    } catch { showToast('Could not start session. Is Flask offline?', 'red') }
    finally   { setLoading(false) }
  }

  const handleStop = async () => {
    activeRef.current = false
    setActive(false)
    setStatus('stopped')
    stopCamera()
    if (sessionIdRef.current) {
      await endSession(sessionIdRef.current)
      const h = await getHistory(sessionIdRef.current)
      setHistory(h.history || [])
    }
    showToast('Session stopped.', 'blue')
  }

  // ── custom collector training data collection ────────────────────────────
  const startTraining = () => {
    if (!cameraRef.current) startCamera()
    trainingModeRef.current = true
    trainingIdxRef.current  = 0
    trainingSampRef.current = 0
    collectingRef.current   = false
    setTrainingMode(true); setTrainingIdx(0); setTrainingSamps(0)
    setTrainingWait(true);  setTrainingDone(false)
    setTimeout(() => { collectingRef.current = true; setTrainingWait(false) }, 2500)
  }

  // ── helpers ───────────────────────────────────────────────────────────────
  const showToast = (msg, color='blue') => {
    setToast({ msg, color }); setTimeout(() => setToast(null), 3500)
  }
  const handleSpace  = () => setText(p => p + ' ')
  const handleClear  = () => { setText(''); setLastLabel(null); setStats(s=>({...s,words:0})) }

  const handleSpeak = async () => {
    if (!text.trim()) return
    try { setLoading(true); setAudioSrc(await generateTTS(text)) }
    catch { showToast('TTS Audio generation failed.', 'red') }
    finally { setLoading(false) }
  }

  const handleExportPDF = async () => {
    try { setLoading(true); await exportPDF(text, sessionIdRef.current); showToast('PDF downloaded successfully!','green') }
    catch { showToast('PDF export failed.','red') } finally { setLoading(false) }
  }

  const handleExportWord = async () => {
    try { setLoading(true); await exportWord(text, sessionIdRef.current); showToast('Word document downloaded!','green') }
    catch { showToast('Word export failed.','red') } finally { setLoading(false) }
  }

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ minHeight:'100vh', display:'flex', flexDirection:'column' }}>

      {/* HEADER */}
      <header style={{
        background: 'rgba(15, 23, 42, 0.6)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid var(--border-glass)',
        padding:'0 28px', height:70, display:'flex', alignItems:'center',
        justifyContent:'space-between', boxShadow:'0 4px 20px rgba(0,0,0,0.15)',
        position: 'sticky', top: 0, zIndex: 100
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:14 }}>
          <span style={{ fontSize:32, filter: 'drop-shadow(0 0 10px var(--neon-cyan))' }}>🤟</span>
          <div>
            <h1 style={{ color:'#fff', fontWeight:900, fontSize:19, letterSpacing: '0.5px', fontFamily: 'var(--font-display)' }}>
              Sign Language translation system
            </h1>
            <div style={{ color:'var(--text-muted)', fontSize:11, fontWeight: 500 }}>
              Surya A · MCA Final Year · PEC, Vaniyambadi
            </div>
          </div>
        </div>
        
        <div style={{ display:'flex', alignItems:'center', gap:12 }}>
          {/* Connection status */}
          <div style={{ display:'flex', alignItems:'center', gap:8, background:'rgba(255,255,255,0.03)',
            padding:'6px 14px', borderRadius:99, fontSize:12, color:'var(--text-main)', border: '1px solid var(--border-glass)' }}>
            <span style={{ width:8, height:8, borderRadius:'50%',
              background: backendOk===null?'#F59E0B':backendOk?'#10B981':'#EF4444',
              boxShadow: backendOk ? '0 0 8px #10B981' : 'none' }} />
            {backendOk===null?'Connecting…':backendOk?`Model: ${backendModel.toUpperCase()}`:'Offline'}
          </div>
          
          {/* Hand detection status */}
          {active && activeTab === 'live' && (
            <div className={`badge ${handVisible?'green':'gray'}`} style={{ padding: '6px 14px', borderRadius: 99 }}>
              <span style={{ width:6, height:6, borderRadius:'50%',
                background: handVisible?'#10B981':'#9CA3AF',
                marginRight: 6, display: 'inline-block',
                animation: handVisible?'pulse-emerald 1.2s infinite':'' }} />
              {handVisible ? '✋ Hand Detected' : 'Scanning...'}
            </div>
          )}
          
          {/* Live Status Badge */}
          {activeTab === 'live' && (
            <div style={{ padding:'6px 16px', borderRadius:99, fontSize:12, fontWeight:700,
              background: status==='running'?'rgba(16,185,129,0.15)':status==='stopped'?'rgba(244,63,94,0.15)':'rgba(156,163,175,0.1)',
              color: status==='running'?'#34D399':status==='stopped'?'#FB7185':'#9CA3AF',
              border: `1px solid ${status==='running'?'rgba(16,185,129,0.3)':status==='stopped'?'rgba(244,63,94,0.3)':'rgba(156,163,175,0.2)'}` }}>
              {status==='running'?'● LIVE':status==='stopped'?'■ STOPPED':'○ IDLE'}
            </div>
          )}
        </div>
      </header>

      {/* TOAST NOTIFICATION */}
      {toast && (
        <div style={{ position:'fixed', top:84, right:28, zIndex:1000,
          background: toast.color==='green'?'rgba(16,185,129,0.15)':'rgba(244,63,94,0.15)',
          color:       toast.color==='green'?'#34D399':'#FB7185',
          border:`1.5px solid ${toast.color==='green'?'rgba(16,185,129,0.4)':'rgba(244,63,94,0.4)'}`,
          backdropFilter: 'blur(10px)',
          padding:'12px 24px', borderRadius:10, fontSize:13, fontWeight:600,
          boxShadow:'0 8px 32px rgba(0,0,0,0.3)', transition: 'all 0.3s ease' }}>
          {toast.msg}
        </div>
      )}

      {/* SUB HEADER TABS */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.3)',
        borderBottom: '1px solid var(--border-glass)',
        padding: '0 28px',
        height: 52,
        display: 'flex',
        alignItems: 'center',
        gap: 12
      }}>
        {[
          { id: 'live', name: '🎥 Live Recognizer' },
          { id: 'processor', name: '🔀 Dataset Processor' },
          { id: 'history', name: '📂 Session Logs & Exports' },
          { id: 'collector', name: '🧠 Custom Collector' }
        ].map(t => (
          <button key={t.id} 
            className="secondary"
            style={{
              background: activeTab === t.id ? 'rgba(6, 182, 212, 0.15)' : 'transparent',
              color: activeTab === t.id ? 'var(--neon-cyan)' : 'var(--text-muted)',
              borderColor: activeTab === t.id ? 'rgba(6, 182, 212, 0.3)' : 'transparent',
              padding: '6px 18px',
              fontSize: 13,
              borderRadius: 8
            }}
            onClick={() => {
              if (active && activeTab === 'live' && t.id !== 'live') {
                handleStop();
              }
              setActiveTab(t.id);
            }}
          >
            {t.name}
          </button>
        ))}
      </div>

      {/* STATS BAR (Only shown for live and history logs) */}
      {(activeTab === 'live' || activeTab === 'history') && (
        <div style={{ background:'rgba(17, 24, 39, 0.4)', borderBottom:'1px solid var(--border-glass)',
          padding:'10px 28px', display:'flex', gap:40 }}>
          {[
            ['Gestures', stats.gestures],
            ['Words', stats.words],
            ['Characters', text.length],
            ['Session ID', sessionId ? `#${sessionId}` : '—']
          ].map(([l, v]) => (
            <div key={l} style={{ display:'flex', gap:10, alignItems:'center' }}>
              <span style={{ fontSize:11, color:'var(--text-muted)', fontWeight:600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>{l}:</span>
              <span style={{ fontSize:15, fontWeight:800, color:'var(--neon-cyan)' }}>{v}</span>
            </div>
          ))}
        </div>
      )}

      {/* MAIN CONTAINER */}
      <main style={{ flex:1, padding:'28px', maxWidth:1240, margin:'0 auto', width:'100%' }}>
        
        {/* TABS CONTAINER */}
        {activeTab === 'live' && (
          <div style={{ display:'grid', gridTemplateColumns:'1fr 340px', gap:24 }}>
            {/* LEFT COLUMN */}
            <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
              
              {/* Webcam Live Capture Bezel */}
              <div className="card" style={{ padding:0, overflow:'hidden' }}>
                <div style={{ padding:'16px 20px', borderBottom:'1px solid var(--border-glass)',
                  display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                  <span style={{ fontWeight:800, color:'#fff', fontSize:15, fontFamily: 'var(--font-display)' }}>📷 Live Recognition Stream</span>
                  {active && confidence > 0 && (
                    <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                      <span style={{ fontSize:12, color:'var(--text-muted)', fontWeight: 500 }}>Confidence:</span>
                      <div style={{ width:100, height:8, background:'#1F2937', borderRadius:99, overflow:'hidden' }}>
                        <div style={{ height:'100%', borderRadius:99, transition:'width 0.3s ease',
                          width:`${confidence*100}%`,
                          background: confidence>=0.8?'var(--neon-emerald)':'var(--neon-amber)' }} />
                      </div>
                      <span style={{ fontSize:12, fontWeight:800,
                        color: confidence>=0.8?'#34D399':'#FBBF24' }}>
                        {(confidence*100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                </div>
                
                <div style={{ padding:16, position:'relative', background:'#070A13', minHeight:340, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  {/* Hidden source video */}
                  <video ref={videoRef} autoPlay playsInline muted style={{ display:'none' }} />

                  {/* Visual skeletal output */}
                  <canvas ref={canvasRef}
                    style={{ width:'100%', maxWidth: 640, borderRadius:12, display: active?'block':'none',
                      transform:'scaleX(-1)', border: '1px solid rgba(255,255,255,0.05)', boxShadow: '0 0 20px rgba(0,0,0,0.8)' }} />

                  {/* Camera Offline Placeholder */}
                  {!active && (
                    <div style={{ display:'flex', flexDirection:'column', alignItems:'center',
                      justifyContent:'center', height:300, color:'var(--text-muted)', gap:16 }}>
                      <span style={{ fontSize:58, filter: 'drop-shadow(0 0 10px rgba(255,255,255,0.1))' }}>📷</span>
                      <div style={{ textAlign: 'center' }}>
                        <p style={{ fontSize:15, fontWeight:600, color: '#fff', marginBottom: 4 }}>Webcam is offline</p>
                        <p style={{ fontSize:12, color:'var(--text-muted)' }}>
                          Click <strong style={{ color: 'var(--neon-emerald)' }}>Start</strong> to start sign classification
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <SubtitleDisplay text={text} lastLabel={lastLabel} confidence={confidence} accepted={confidence>=0.8} />
              <AudioPlayer src={audioSrc} />
            </div>

            {/* RIGHT COLUMN */}
            <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
              <ControlPanel active={active} onStart={handleStart} onStop={handleStop}
                onSpace={handleSpace} onClear={handleClear} onSpeak={handleSpeak}
                onExportPDF={handleExportPDF} onExportWord={handleExportWord}
                loading={loading} text={text} />

              {/* Dynamic ASL Reference card */}
              <div className="card">
                <span style={{ fontWeight:700, color:'var(--neon-cyan)', fontSize:15, display:'block', marginBottom:12, fontFamily: 'var(--font-display)' }}>
                  🔤 Alphanumeric Reference
                </span>
                <div style={{ display:'flex', flexWrap:'wrap', gap:6 }}>
                  {Array.from('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ').map(l => (
                    <div key={l} style={{ width:32, height:32, borderRadius:6,
                      background: lastLabel===l?'var(--neon-cyan)':'rgba(255,255,255,0.03)',
                      color:      lastLabel===l?'#1E293B':'var(--text-main)',
                      border:     `1px solid ${lastLabel===l?'var(--neon-cyan)':'rgba(255,255,255,0.06)'}`,
                      boxShadow:  lastLabel===l?'0 0 10px rgba(6, 182, 212, 0.4)':'none',
                      display:'flex', alignItems:'center', justifyContent:'center',
                      fontSize:13, fontWeight:700, transition:'all 0.15s',
                      transform: lastLabel===l?'scale(1.15)':'scale(1)' }}>
                      {l}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* DATASET PROCESSOR TAB */}
        {activeTab === 'processor' && (
          <DatasetProcessor showToast={showToast} onModelTrained={checkHealth} />
        )}

        {/* HISTORY LOGS TAB */}
        {activeTab === 'history' && (
          <div style={{ display:'grid', gridTemplateColumns:'1fr 340px', gap:24 }}>
            <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
              <div className="card">
                <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--neon-cyan)', marginBottom: 12 }}>
                  📝 Document Transcript
                </h3>
                <SubtitleDisplay text={text} lastLabel={null} confidence={0} accepted={true} />
              </div>
              <AudioPlayer src={audioSrc} />
            </div>
            
            <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
              <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
                  📂 Export Session Records
                </h3>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5, marginBottom: 4 }}>
                  Save your translation transcripts as a PDF or Microsoft Word document.
                </p>
                <button className="primary" onClick={handleSpeak} disabled={!text.trim() || loading} style={{ width: '100%' }}>
                  🔊 Speak Transcript
                </button>
                <div style={{ display: 'flex', gap: 10 }}>
                  <button className="secondary" onClick={handleExportPDF} disabled={!text.trim() || loading} style={{ flex: 1 }}>
                    📄 PDF Document
                  </button>
                  <button className="secondary" onClick={handleExportWord} disabled={!text.trim() || loading} style={{ flex: 1 }}>
                    📝 Word Document
                  </button>
                </div>
              </div>

              <SessionHistory history={history} />
            </div>
          </div>
        )}

        {/* CUSTOM WEBCAM COLLECTOR TAB */}
        {activeTab === 'collector' && (
          <div style={{ display:'grid', gridTemplateColumns:'1fr 340px', gap:24 }}>
            {/* Camera feed */}
            <div className="card" style={{ padding:0, overflow:'hidden' }}>
              <div style={{ padding:'16px 20px', borderBottom:'1px solid var(--border-glass)' }}>
                <span style={{ fontWeight:800, color:'#fff', fontSize:15, fontFamily: 'var(--font-display)' }}>📷 Live Webcam Collector</span>
              </div>
              <div style={{ padding:16, position:'relative', background:'#070A13', minHeight:340, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <video ref={videoRef} autoPlay playsInline muted style={{ display:'none' }} />
                <canvas ref={canvasRef} style={{ width:'100%', maxWidth:640, borderRadius:12, display: active?'block':'none', transform:'scaleX(-1)' }} />
                
                {!active && (
                  <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:300, color:'var(--text-muted)', gap:16 }}>
                    <span style={{ fontSize:58 }}>📷</span>
                    <p style={{ fontSize:15, fontWeight:600, color: '#fff' }}>Webcam is offline</p>
                  </div>
                )}
              </div>
            </div>

            {/* Collector info and options */}
            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <h3 style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 15, color: 'var(--neon-rose)' }}>
                🧠 Custom Gesture Collector
              </h3>
              
              {trainingDone ? (
                <div>
                  <p style={{ color:'#34D399', fontWeight:700, marginBottom:10 }}>✅ Custom Data Collected!</p>
                  <p style={{ fontSize:12, color:'var(--text-muted)', lineHeight: 1.5, marginBottom:12 }}>
                    Your recorded hand structures are stored. Click training tab or button below to compile and save your classifier.
                  </p>
                  <button className="success" onClick={handleTrainModel} style={{ width: '100%', marginBottom: 10 }}>
                    ⚡ Train Classifier Now
                  </button>
                  <button className="secondary" style={{ width:'100%' }} onClick={() => setTrainingDone(false)}>
                    Record More Data
                  </button>
                </div>
              ) : trainingMode ? (
                <div style={{ textAlign:'center', padding: '10px 0' }}>
                  <div style={{ fontSize:84, color:'var(--neon-cyan)', fontWeight:900, fontFamily: 'var(--font-display)' }}>
                    {TRAIN_LETTERS[trainingIdx]}
                  </div>
                  <p style={{ fontSize:13, color:'#fff', fontWeight: 600, margin:'8px 0' }}>
                    {trainingWait ? '⏳ Hold on, next letter...' : '✋ Form this sign now!'}
                  </p>
                  <div style={{ margin:'14px 0', height:8, background:'#1F2937', borderRadius:99, overflow: 'hidden' }}>
                    <div style={{ height:'100%', background:'var(--neon-cyan)', width:`${(trainingSamps/SAMPLES_PER_LTR)*100}%`, transition:'width 0.1s' }}/>
                  </div>
                  <p style={{ fontSize:11, color:'var(--text-muted)' }}>
                    {trainingSamps}/{SAMPLES_PER_LTR} samples · Class {trainingIdx+1}/26 (A-Z)
                  </p>
                  <button className="danger" style={{ width:'100%', marginTop:14 }}
                    onClick={() => { trainingModeRef.current=false; collectingRef.current=false; setTrainingMode(false); stopCamera(); setActive(false) }}>
                    Stop Recording
                  </button>
                </div>
              ) : (
                <div>
                  <p style={{ fontSize:12, color:'var(--text-muted)', lineHeight: 1.5, marginBottom:16 }}>
                    This tool allows you to record custom gestural landmark sets through your webcam. It loops through letters A-Z, capturing 40 frames each.
                  </p>
                  <button className="primary" style={{ width:'100%' }} onClick={startTraining}>
                    🎯 Start Webcam Recording
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* FOOTER */}
      <footer style={{
        background: 'rgba(15, 23, 42, 0.8)',
        borderTop: '1px solid var(--border-glass)',
        textAlign:'center', padding:'16px 0', fontSize:12, color: 'var(--text-muted)'
      }}>
        Sign Language translation system · Surya A · Final Year MCA Project · PEC, Vaniyambadi · 2026
      </footer>

      <style>{`
        @keyframes pulse-emerald {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        @media(max-width:860px){
          main > div { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  )
}