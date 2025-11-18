import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Webcam from 'react-webcam'
import axios from 'axios'
import { 
  Play, Pause, Square, TrendingUp, AlertTriangle,
  CheckCircle, Activity, Zap, ArrowLeft 
} from 'lucide-react'
import { Link } from 'react-router-dom'

const WorkoutSession = () => {
  const webcamRef = useRef(null)
  const [isActive, setIsActive] = useState(false)
  const [exercise, setExercise] = useState('DETECTING...')
  const [reps, setReps] = useState(0)
  const [stage, setStage] = useState('ready')
  const [feedback, setFeedback] = useState('Start exercising')
  const [confidence, setConfidence] = useState(0)
  const [formQuality, setFormQuality] = useState('GOOD')
  const [sessionStats, setSessionStats] = useState({
    totalReps: 0,
    goodFormReps: 0,
    duration: 0
  })
  const [angles, setAngles] = useState({ elbow: 0, back: 0 })

  const intervalRef = useRef(null)
  const isProcessingRef = useRef(false)

  useEffect(() => {
    if (isActive) {
      startWorkout()
    } else {
      stopWorkout()
    }
    
    return () => stopWorkout()
  }, [isActive])

  const startWorkout = () => {
    // Send video frames to backend every 150ms for better responsiveness
    intervalRef.current = setInterval(async () => {
      // Skip if previous request is still processing
      if (isProcessingRef.current) return
      
      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot()
        if (imageSrc) {
          await sendFrame(imageSrc)
        }
      }
    }, 150)
  }

  const stopWorkout = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }

  const sendFrame = async (imageData) => {
    // Prevent overlapping requests
    if (isProcessingRef.current) return
    
    isProcessingRef.current = true
    
    try {
      const response = await axios.post('/api/workout/analyze', {
        frame: imageData
      }, {
        timeout: 3000 // 3 second timeout for faster response
      })

      const data = response.data
      
      // Update state with backend response
      if (data.exercise) {
        setExercise(data.exercise.toUpperCase())
        setConfidence(data.confidence || 0)
      }
      
      if (data.reps !== undefined) setReps(data.reps)
      if (data.stage) setStage(data.stage)
      if (data.feedback) setFeedback(data.feedback)
      if (data.formQuality) setFormQuality(data.formQuality)
      if (data.angles) setAngles(data.angles)
      
      // Update session stats
      setSessionStats(prev => ({
        ...prev,
        totalReps: data.reps || prev.totalReps,
        goodFormReps: data.formQuality === 'GOOD' ? data.reps : prev.goodFormReps
      }))

    } catch (error) {
      console.error('Error analyzing frame:', error)
    } finally {
      isProcessingRef.current = false
    }
  }

  const getFeedbackColor = () => {
    if (feedback.includes('GOOD')) return 'from-accent-500 to-accent-600'
    if (feedback.includes('TUCK') || feedback.includes('STRAIGHTEN')) return 'from-primary-500 to-primary-600'
    if (feedback.includes('COUNTED')) return 'from-warning-500 to-warning-600'
    return 'from-gray-600 to-gray-700'
  }

  const getFormIcon = () => {
    if (formQuality === 'GOOD') return <CheckCircle className="w-6 h-6 text-accent-400" />
    return <AlertTriangle className="w-6 h-6 text-primary-400" />
  }

  return (
    <div className="min-h-screen bg-dark-900 p-4">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex items-center justify-between">
          <Link to="/user/dashboard" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Dashboard</span>
          </Link>
          
          <h1 className="text-3xl font-display font-bold">
            <span className="gradient-text">WORKOUT</span> SESSION
          </h1>
          
          <div className="w-32"></div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid lg:grid-cols-3 gap-6">
        {/* Main Video Feed */}
        <div className="lg:col-span-2">
          <div className="card p-0 overflow-hidden">
            {/* Video Container */}
            <div className="relative aspect-video bg-dark-900">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                className="w-full h-full object-cover"
                videoConstraints={{
                  facingMode: 'user',
                  width: 1280,
                  height: 720,
                }}
              />
              
              {/* Overlay UI */}
              <div className="absolute inset-0 pointer-events-none">
                {/* Top Stats Bar */}
                <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-dark-900/90 to-transparent p-6">
                  <div className="grid grid-cols-3 gap-4">
                    {/* Exercise Detection */}
                    <div className="bg-dark-800/80 backdrop-blur-sm rounded-lg p-4 border-2 border-primary-500/30">
                      <p className="text-xs text-gray-400 mb-1">EXERCISE</p>
                      <p className="text-2xl font-black gradient-text">{exercise}</p>
                      {confidence > 0 && (
                        <p className="text-xs text-accent-400 mt-1">{Math.round(confidence)}% confident</p>
                      )}
                    </div>
                    
                    {/* Rep Counter */}
                    <div className="bg-dark-800/80 backdrop-blur-sm rounded-lg p-4 border-2 border-accent-500/30">
                      <p className="text-xs text-gray-400 mb-1">REPS</p>
                      <p className="text-4xl font-black text-accent-400">{reps}</p>
                    </div>
                    
                    {/* Stage */}
                    <div className="bg-dark-800/80 backdrop-blur-sm rounded-lg p-4 border-2 border-warning-500/30">
                      <p className="text-xs text-gray-400 mb-1">STAGE</p>
                      <p className="text-2xl font-black text-warning-400">{stage.toUpperCase()}</p>
                    </div>
                  </div>
                </div>

                {/* Bottom Feedback Bar */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-dark-900/90 to-transparent p-6">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={feedback}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      className={`bg-gradient-to-r ${getFeedbackColor()} rounded-xl p-6 shadow-2xl`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          {getFormIcon()}
                          <p className="text-2xl font-bold text-white">{feedback}</p>
                        </div>
                        
                        {exercise === 'PUSH-UP' && (
                          <div className="flex gap-6 text-sm">
                            <div>
                              <span className="text-gray-300">Elbow:</span>
                              <span className="ml-2 font-bold text-white">{Math.round(angles.elbow)}°</span>
                            </div>
                            <div>
                              <span className="text-gray-300">Back:</span>
                              <span className="ml-2 font-bold text-white">{Math.round(angles.back)}°</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  </AnimatePresence>
                </div>

                {/* Recording Indicator */}
                {isActive && (
                  <div className="absolute top-6 right-6 flex items-center gap-2 bg-primary-500 px-4 py-2 rounded-full">
                    <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                    <span className="text-sm font-bold text-white">LIVE</span>
                  </div>
                )}
              </div>
            </div>

            {/* Controls */}
            <div className="p-6 bg-dark-800 border-t border-dark-700">
              <div className="flex items-center justify-center gap-4">
                {!isActive ? (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setIsActive(true)}
                    className="btn-primary px-12 py-4 flex items-center gap-3"
                  >
                    <Play className="w-6 h-6" />
                    <span className="text-lg font-bold">Start Workout</span>
                  </motion.button>
                ) : (
                  <>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setIsActive(false)}
                      className="btn-secondary px-8 py-3 flex items-center gap-2"
                    >
                      <Pause className="w-5 h-5" />
                      Pause
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        setIsActive(false)
                        setReps(0)
                        setSessionStats({ totalReps: 0, goodFormReps: 0, duration: 0 })
                      }}
                      className="btn-outline px-8 py-3 flex items-center gap-2"
                    >
                      <Square className="w-5 h-5" />
                      End Session
                    </motion.button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar Stats */}
        <div className="space-y-6">
          {/* Form Quality */}
          <div className="card">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary-500" />
              Form Quality
            </h3>
            <div className={`text-center p-6 rounded-lg ${formQuality === 'GOOD' ? 'bg-accent-500/20 border-2 border-accent-500/50' : 'bg-primary-500/20 border-2 border-primary-500/50'}`}>
              <div className="text-5xl font-black mb-2">
                {formQuality === 'GOOD' ? '✓' : '✗'}
              </div>
              <div className={`text-2xl font-bold ${formQuality === 'GOOD' ? 'text-accent-400' : 'text-primary-400'}`}>
                {formQuality}
              </div>
            </div>
          </div>

          {/* Session Stats */}
          <div className="card">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-accent-500" />
              Session Stats
            </h3>
            <div className="space-y-4">
              <div className="stat-box">
                <p className="text-sm text-gray-400 mb-1">Total Reps</p>
                <p className="text-3xl font-black text-white">{sessionStats.totalReps}</p>
              </div>
              <div className="stat-box">
                <p className="text-sm text-gray-400 mb-1">Good Form Reps</p>
                <p className="text-3xl font-black text-accent-400">{sessionStats.goodFormReps}</p>
              </div>
              <div className="stat-box">
                <p className="text-sm text-gray-400 mb-1">Form Accuracy</p>
                <p className="text-3xl font-black text-warning-400">
                  {sessionStats.totalReps > 0 
                    ? Math.round((sessionStats.goodFormReps / sessionStats.totalReps) * 100)
                    : 0}%
                </p>
              </div>
            </div>
          </div>

          {/* Quick Tips */}
          <div className="card bg-gradient-to-br from-primary-500/10 to-accent-500/10 border border-primary-500/20">
            <h3 className="text-lg font-bold mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5 text-warning-500" />
              Quick Tips
            </h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>• Keep your body in frame</li>
              <li>• Maintain good lighting</li>
              <li>• Follow form feedback</li>
              <li>• Focus on quality over quantity</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WorkoutSession
