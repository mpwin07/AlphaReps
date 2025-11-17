import { motion, useAnimation } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Dumbbell, Zap, Brain, TrendingUp, Shield, Users, Activity, Target, Award, Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import Navbar from '../components/Navbar'

const Landing = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const controls = useAnimation()

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 20,
        y: (e.clientY / window.innerHeight - 0.5) * 20
      })
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  useEffect(() => {
    controls.start({
      rotate: 360,
      transition: { duration: 20, repeat: Infinity, ease: 'linear' }
    })
  }, [controls])

  const features = [
    {
      icon: Brain,
      title: 'AI Exercise Detection',
      description: 'Automatically identifies your exercise in real-time',
      color: 'from-primary-500 to-primary-600',
      delay: 0.2
    },
    {
      icon: Zap,
      title: 'Real-Time Feedback',
      description: 'Instant posture correction and form guidance',
      color: 'from-accent-500 to-accent-600',
      delay: 0.3
    },
    {
      icon: Activity,
      title: 'Accurate Rep Counting',
      description: 'Smart AI counts only perfect form reps',
      color: 'from-primary-400 to-accent-400',
      delay: 0.4
    },
    {
      icon: Target,
      title: 'Form Analysis',
      description: 'Advanced pose tracking for perfect technique',
      color: 'from-accent-400 to-primary-400',
      delay: 0.5
    },
  ]

  return (
    <div className="min-h-screen bg-dark-900 overflow-hidden relative">
      {/* Navbar */}
      <Navbar />

      {/* Animated Background Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1890ff0a_1px,transparent_1px),linear-gradient(to_bottom,#1890ff0a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute w-2 h-2 rounded-full ${i % 3 === 0 ? 'bg-primary-500' : i % 3 === 1 ? 'bg-accent-500' : 'bg-primary-300'
              }`}
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              opacity: Math.random() * 0.5 + 0.2
            }}
            animate={{
              y: [null, Math.random() * window.innerHeight],
              x: [null, Math.random() * window.innerWidth],
              opacity: [null, Math.random() * 0.5 + 0.2]
            }}
            transition={{
              duration: Math.random() * 10 + 10,
              repeat: Infinity,
              repeatType: 'reverse',
              ease: 'linear'
            }}
          />
        ))}
      </div>

      {/* Gradient Orbs */}
      <div className="absolute inset-0">
        <motion.div
          className="absolute w-[500px] h-[500px] bg-primary-500/20 rounded-full blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
            scale: [1, 1.2, 1]
          }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
          style={{ top: '10%', left: '10%' }}
        />
        <motion.div
          className="absolute w-[600px] h-[600px] bg-accent-500/20 rounded-full blur-3xl"
          animate={{
            x: [0, -100, 0],
            y: [0, -50, 0],
            scale: [1, 1.3, 1]
          }}
          transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
          style={{ bottom: '10%', right: '10%' }}
        />
        <motion.div
          className="absolute w-[400px] h-[400px] bg-primary-400/10 rounded-full blur-3xl"
          animate={{
            x: [0, -50, 0],
            y: [0, 100, 0],
            scale: [1, 1.1, 1]
          }}
          transition={{ duration: 12, repeat: Infinity, ease: 'easeInOut' }}
          style={{ top: '50%', left: '50%' }}
        />
      </div>

      {/* Hero Section */}
      <div className="relative z-10 min-h-screen flex items-start justify-center px-4 pt-28">
        <div className="max-w-6xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Main Heading */}
            <motion.h1
              className="text-7xl md:text-9xl font-display font-black mb-6"
              style={{
                transform: `perspective(1000px) rotateX(${mousePosition.y * 0.05}deg) rotateY(${mousePosition.x * 0.05}deg)`
              }}
            >
              <motion.span
                className="inline-block gradient-text"
                animate={{
                  backgroundPosition: ['0% 50%', '100% 50%', '0% 50%']
                }}
                transition={{ duration: 5, repeat: Infinity, ease: 'linear' }}
              >
                ALPHA
              </motion.span>
              <motion.span
                className="inline-block text-white text-shadow"
                animate={{
                  textShadow: [
                    '0 0 20px rgba(24,144,255,0.5)',
                    '0 0 40px rgba(82,196,26,0.5)',
                    '0 0 20px rgba(24,144,255,0.5)'
                  ]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                REPS
              </motion.span>
            </motion.h1>

            <motion.p
              className="text-2xl md:text-3xl text-gray-300 mb-4 font-semibold"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <span className="text-primary-400">Train Smarter,</span>{' '}
              <span className="text-accent-400">Not Harder</span>
            </motion.p>
            <motion.p
              className="text-lg text-gray-400 mb-12 max-w-2xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              Real-time exercise detection, posture correction, and rep counting powered by advanced AI
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              className="flex flex-col sm:flex-row gap-4 justify-center mb-16"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
            >
              <Link to="/login">
                <motion.button
                  whileHover={{
                    scale: 1.05,
                    boxShadow: '0 0 30px rgba(24,144,255,0.6)'
                  }}
                  whileTap={{ scale: 0.95 }}
                  className="btn-primary px-12 py-4 text-lg relative overflow-hidden group"
                >
                  <span className="relative z-10 flex items-center gap-2">
                    <Zap className="w-5 h-5" />
                    Get Started
                  </span>
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-accent-500 to-primary-500"
                    initial={{ x: '-100%' }}
                    whileHover={{ x: 0 }}
                    transition={{ duration: 0.3 }}
                  />
                </motion.button>
              </Link>
            </motion.div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-8 max-w-3xl mx-auto mb-20">
              {[
                { value: '95%+', label: 'Accuracy', icon: Award },
                { value: '5', label: 'Exercises', icon: Dumbbell },
                { value: 'Real-Time', label: 'Feedback', icon: Zap }
              ].map((stat, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8 + i * 0.1 }}
                  whileHover={{
                    scale: 1.05,
                    boxShadow: i % 2 === 0
                      ? '0 0 30px rgba(24,144,255,0.3)'
                      : '0 0 30px rgba(82,196,26,0.3)'
                  }}
                  className="stat-box cursor-pointer group"
                >
                  <stat.icon className={`w-8 h-8 mx-auto mb-3 ${i % 2 === 0 ? 'text-primary-400' : 'text-accent-400'
                    } group-hover:scale-110 transition-transform`} />
                  <motion.div
                    className="text-3xl md:text-4xl font-black gradient-text mb-2"
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
                  >
                    {stat.value}
                  </motion.div>
                  <div className="text-sm text-gray-400 font-semibold">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mt-20">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 50, rotateX: -30 }}
                animate={{ opacity: 1, y: 0, rotateX: 0 }}
                transition={{
                  delay: 1 + feature.delay,
                  type: 'spring',
                  stiffness: 100
                }}
                whileHover={{
                  y: -10,
                  boxShadow: index % 2 === 0
                    ? '0 20px 40px rgba(24,144,255,0.3)'
                    : '0 20px 40px rgba(82,196,26,0.3)'
                }}
                className="card-hover group relative overflow-hidden"
              >
                <motion.div
                  className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-accent-500/10 opacity-0 group-hover:opacity-100 transition-opacity"
                  animate={{
                    backgroundPosition: ['0% 0%', '100% 100%']
                  }}
                  transition={{ duration: 3, repeat: Infinity, repeatType: 'reverse' }}
                />
                <div className="relative z-10">
                  <motion.div
                    className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mb-4 shadow-lg`}
                    whileHover={{
                      scale: 1.2,
                      rotate: 360
                    }}
                    transition={{ type: 'spring', stiffness: 200 }}
                  >
                    <feature.icon className="w-8 h-8 text-white" />
                  </motion.div>
                  <h3 className="text-xl font-bold mb-2 group-hover:text-primary-400 transition-colors">
                    {feature.title}
                  </h3>
                  <p className="text-gray-400 text-sm">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Supported Exercises */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.5 }}
            className="mt-20"
          >
            <motion.h3
              className="text-3xl font-bold mb-8 gradient-text"
              animate={{
                backgroundPosition: ['0% 50%', '100% 50%', '0% 50%']
              }}
              transition={{ duration: 5, repeat: Infinity }}
            >
              Supported Exercises
            </motion.h3>
            <div className="flex flex-wrap justify-center gap-4">
              {['Push-ups', 'Squats', 'Bicep Curls', 'Hammer Curls', 'Shoulder Press'].map((exercise, i) => (
                <motion.span
                  key={i}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.7 + i * 0.1, type: 'spring' }}
                  whileHover={{
                    scale: 1.1,
                    boxShadow: '0 0 20px rgba(82,196,26,0.5)'
                  }}
                  className="badge-success text-base px-6 py-3 cursor-pointer"
                >
                  {exercise}
                </motion.span>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

export default Landing
