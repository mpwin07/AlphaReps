import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Dumbbell, Zap, Brain, TrendingUp, Shield, Users } from 'lucide-react'

const Landing = () => {
  const features = [
    {
      icon: Brain,
      title: 'AI Exercise Detection',
      description: 'Automatically identifies your exercise in real-time',
      color: 'from-primary-500 to-primary-600'
    },
    {
      icon: Zap,
      title: 'Real-Time Feedback',
      description: 'Instant posture correction and form guidance',
      color: 'from-accent-500 to-accent-600'
    },
    {
      icon: TrendingUp,
      title: 'Accurate Rep Counting',
      description: 'Smart AI counts only perfect form reps',
      color: 'from-warning-500 to-warning-600'
    },
    {
      icon: Shield,
      title: 'Face Recognition',
      description: 'Secure login with advanced face detection',
      color: 'from-purple-500 to-purple-600'
    },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0">
        <div className="absolute w-96 h-96 bg-primary-500/20 rounded-full blur-3xl top-0 left-0 animate-pulse-slow"></div>
        <div className="absolute w-96 h-96 bg-accent-500/20 rounded-full blur-3xl bottom-0 right-0 animate-pulse-slow"></div>
      </div>

      {/* Hero Section */}
      <div className="relative z-10 min-h-screen flex items-center justify-center px-4">
        <div className="max-w-6xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Logo/Badge */}
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
              className="inline-block mb-6"
            >
              <div className="bg-gradient-to-r from-primary-500 to-accent-500 p-1 rounded-full">
                <div className="bg-dark-900 rounded-full px-6 py-2">
                  <span className="text-sm font-bold gradient-text">AI-POWERED TRAINING</span>
                </div>
              </div>
            </motion.div>

            {/* Main Heading */}
            <h1 className="text-7xl md:text-9xl font-display font-black mb-6">
              <span className="gradient-text">ALPHA</span>
              <span className="text-white text-shadow">REPS</span>
            </h1>

            <p className="text-2xl md:text-3xl text-gray-300 mb-4 font-semibold">
              Your AI Personal Trainer
            </p>
            <p className="text-lg text-gray-400 mb-12 max-w-2xl mx-auto">
              Real-time exercise detection, posture correction, and rep counting powered by advanced AI
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
              <Link to="/login">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="btn-primary px-12 py-4 text-lg"
                >
                  Get Started
                </motion.button>
              </Link>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-outline px-12 py-4 text-lg"
              >
                Watch Demo
              </motion.button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-8 max-w-3xl mx-auto mb-20">
              {[
                { value: '95%+', label: 'Accuracy' },
                { value: '5', label: 'Exercises' },
                { value: 'Real-Time', label: 'Feedback' }
              ].map((stat, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + i * 0.1 }}
                  className="stat-box"
                >
                  <div className="text-3xl md:text-4xl font-black gradient-text mb-2">
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-400">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mt-20">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + index * 0.1 }}
                className="card-hover group"
              >
                <div className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <feature.icon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                <p className="text-gray-400 text-sm">{feature.description}</p>
              </motion.div>
            ))}
          </div>

          {/* Supported Exercises */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="mt-20"
          >
            <h3 className="text-2xl font-bold mb-6">Supported Exercises</h3>
            <div className="flex flex-wrap justify-center gap-3">
              {['Push-ups', 'Squats', 'Bicep Curls', 'Hammer Curls', 'Shoulder Press'].map((exercise, i) => (
                <span
                  key={i}
                  className="badge-success text-base px-6 py-3"
                >
                  {exercise}
                </span>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

export default Landing
