import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { User, Lock, UserCheck, AlertCircle, Loader2, Dumbbell } from 'lucide-react'
import { useAuthStore } from '../store/authStore'

const FaceLogin = () => {
  const navigate = useNavigate()
  const { login } = useAuthStore()
  
  const [status, setStatus] = useState('ready') // ready, loading, success, error
  const [message, setMessage] = useState('Enter your credentials to continue')
  const [userName, setUserName] = useState('')
  const [userRole, setUserRole] = useState('user') // 'user' or 'admin'
  const [isLoading, setIsLoading] = useState(false)

  const handleLogin = async (e) => {
    e.preventDefault()
    
    if (!userName.trim()) {
      setStatus('error')
      setMessage('Please enter your name')
      setTimeout(() => setStatus('ready'), 2000)
      return
    }

    setIsLoading(true)
    setStatus('loading')
    setMessage('Logging in...')

    // Simulate login (replace with actual API call if needed)
    setTimeout(() => {
      const userData = {
        name: userName,
        role: userRole,
        email: `${userName.toLowerCase().replace(/\s+/g, '')}@alphareps.com`,
        joinDate: new Date().toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
      }

      setStatus('success')
      setMessage(`Welcome, ${userName}!`)
      
      setTimeout(() => {
        login(userData, null)
        navigate(userRole === 'admin' ? '/admin' : '/user/dashboard')
      }, 1000)
    }, 1500)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 flex items-center justify-center p-4">
      {/* Background Animation */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute w-96 h-96 bg-primary-500/10 rounded-full blur-3xl top-10 -left-20 animate-pulse-slow"></div>
        <div className="absolute w-96 h-96 bg-accent-500/10 rounded-full blur-3xl bottom-10 -right-20 animate-pulse-slow delay-75"></div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-md"
      >
        {/* Header */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-primary-500 to-accent-500 rounded-full mb-6"
          >
            <Dumbbell className="w-10 h-10 text-white" />
          </motion.div>
          
          <h1 className="text-5xl font-display font-black mb-4">
            <span className="gradient-text">ALPHA</span>
            <span className="text-white">REPS</span>
          </h1>
          <p className="text-lg text-gray-400">
            AI-Powered Personal Gym Trainer
          </p>
        </div>

        {/* Login Card */}
        <div className="card">
          <h2 className="text-2xl font-display font-bold mb-6 text-center">
            Login to Continue
          </h2>

          {/* Status Message */}
          <AnimatePresence mode="wait">
            <motion.div
              key={status}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className={`flex items-center gap-3 p-4 rounded-lg mb-6 ${
                status === 'success' ? 'bg-accent-500/20 border border-accent-500/30' :
                status === 'error' ? 'bg-primary-500/20 border border-primary-500/30' :
                status === 'loading' ? 'bg-warning-500/20 border border-warning-500/30' :
                'bg-dark-700 border border-dark-600'
              }`}
            >
              {status === 'success' && <UserCheck className="w-5 h-5 text-accent-400" />}
              {status === 'error' && <AlertCircle className="w-5 h-5 text-primary-400" />}
              {status === 'loading' && <Loader2 className="w-5 h-5 text-warning-400 animate-spin" />}
              {status === 'ready' && <User className="w-5 h-5 text-gray-400" />}
              <p className="text-sm font-medium">{message}</p>
            </motion.div>
          </AnimatePresence>

          {/* Login Form */}
          <form onSubmit={handleLogin} className="space-y-4">
            {/* Name Input */}
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">
                Your Name
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={userName}
                  onChange={(e) => setUserName(e.target.value)}
                  placeholder="Enter your name"
                  className="input-field w-full pl-11"
                  disabled={isLoading}
                  autoFocus
                />
              </div>
            </div>

            {/* Role Selection */}
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">
                Login As
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setUserRole('user')}
                  disabled={isLoading}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    userRole === 'user'
                      ? 'border-primary-500 bg-primary-500/20'
                      : 'border-dark-600 bg-dark-700 hover:border-dark-500'
                  }`}
                >
                  <User className="w-6 h-6 mx-auto mb-2 text-primary-400" />
                  <p className="text-sm font-semibold">User</p>
                </button>
                
                <button
                  type="button"
                  onClick={() => setUserRole('admin')}
                  disabled={isLoading}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    userRole === 'admin'
                      ? 'border-accent-500 bg-accent-500/20'
                      : 'border-dark-600 bg-dark-700 hover:border-dark-500'
                  }`}
                >
                  <Lock className="w-6 h-6 mx-auto mb-2 text-accent-400" />
                  <p className="text-sm font-semibold">Admin</p>
                </button>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading || !userName.trim()}
              className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Logging in...
                </span>
              ) : (
                'Login'
              )}
            </button>
          </form>

          {/* Quick Login Suggestions */}
          <div className="mt-6 pt-6 border-t border-dark-700">
            <p className="text-xs text-gray-500 text-center mb-3">Quick Login:</p>
            <div className="flex gap-2 justify-center">
              <button
                onClick={() => setUserName('John Doe')}
                disabled={isLoading}
                className="text-xs px-3 py-1 bg-dark-700 hover:bg-dark-600 rounded-full transition-colors"
              >
                John Doe
              </button>
              <button
                onClick={() => setUserName('Sarah Smith')}
                disabled={isLoading}
                className="text-xs px-3 py-1 bg-dark-700 hover:bg-dark-600 rounded-full transition-colors"
              >
                Sarah Smith
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-gray-500 mt-6">
          Powered by AI • Real-Time Feedback • 95%+ Accuracy
        </p>
      </motion.div>
    </div>
  )
}

export default FaceLogin
