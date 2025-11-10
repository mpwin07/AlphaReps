import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Dumbbell, TrendingUp, Target, Calendar,
  Play, BarChart3, User, LogOut 
} from 'lucide-react'
import { useAuthStore } from '../../store/authStore'

const UserDashboard = () => {
  const { user, logout } = useAuthStore()

  const quickStats = [
    { label: 'Total Workouts', value: '24', icon: Calendar, color: 'primary' },
    { label: 'Total Reps', value: '1,240', icon: Target, color: 'accent' },
    { label: 'Avg Accuracy', value: '94%', icon: TrendingUp, color: 'warning' },
    { label: 'Streak', value: '7 days', icon: Dumbbell, color: 'purple' },
  ]

  const recentWorkouts = [
    { exercise: 'Push-ups', reps: 50, accuracy: 96, date: 'Today', time: '10:30 AM' },
    { exercise: 'Squats', reps: 60, accuracy: 94, date: 'Today', time: '09:15 AM' },
    { exercise: 'Bicep Curls', reps: 40, accuracy: 92, date: 'Yesterday', time: '06:45 PM' },
  ]

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Header */}
      <div className="bg-dark-800 border-b border-dark-700">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-display font-black">
                <span className="gradient-text">ALPHA</span>
                <span className="text-white">REPS</span>
              </h1>
              <p className="text-gray-400 mt-1">Welcome back, {user?.name}!</p>
            </div>
            
            <div className="flex items-center gap-4">
              <Link to="/user/profile">
                <button className="btn-outline py-2 px-4 flex items-center gap-2">
                  <User className="w-4 h-4" />
                  Profile
                </button>
              </Link>
              <button 
                onClick={logout}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Quick Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {quickStats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card-hover"
            >
              <div className="flex items-center justify-between mb-3">
                <stat.icon className={`w-8 h-8 text-${stat.color}-500`} />
                <div className={`w-2 h-2 rounded-full bg-${stat.color}-500 animate-pulse`}></div>
              </div>
              <p className="text-3xl font-black mb-1">{stat.value}</p>
              <p className="text-sm text-gray-400">{stat.label}</p>
            </motion.div>
          ))}
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Start Workout CTA */}
          <div className="lg:col-span-2">
            <div className="workout-card h-full">
              <div className="flex flex-col md:flex-row items-center justify-between h-full">
                <div className="mb-6 md:mb-0">
                  <h2 className="text-4xl font-display font-black mb-4">
                    Ready to <span className="gradient-text">CRUSH</span> it?
                  </h2>
                  <p className="text-gray-400 mb-6 max-w-md">
                    Start your AI-powered workout session with real-time feedback and posture correction
                  </p>
                  <Link to="/user/workout">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="btn-primary px-8 py-4 flex items-center gap-3 text-lg"
                    >
                      <Play className="w-6 h-6" />
                      Start Workout
                    </motion.button>
                  </Link>
                </div>
                
                <div className="relative">
                  <div className="w-48 h-48 bg-gradient-to-br from-primary-500/20 to-accent-500/20 rounded-full flex items-center justify-center">
                    <Dumbbell className="w-24 h-24 text-primary-500" />
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-accent-500/10 rounded-full blur-2xl"></div>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card">
            <h3 className="text-xl font-bold mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <Link to="/user/analytics">
                <button className="w-full btn-secondary py-3 flex items-center justify-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  View Analytics
                </button>
              </Link>
              <Link to="/user/workout">
                <button className="w-full btn-outline py-3 flex items-center justify-center gap-2">
                  <Target className="w-5 h-5" />
                  Set Goals
                </button>
              </Link>
            </div>
          </div>
        </div>

        {/* Recent Workouts */}
        <div className="mt-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Recent Workouts</h2>
            <Link to="/user/analytics" className="text-primary-500 hover:text-primary-400 font-semibold">
              View All →
            </Link>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {recentWorkouts.map((workout, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                className="card"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-bold">{workout.exercise}</h3>
                    <p className="text-sm text-gray-400">{workout.date} • {workout.time}</p>
                  </div>
                  <Dumbbell className="w-6 h-6 text-primary-500" />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Reps</span>
                    <span className="text-2xl font-black text-accent-400">{workout.reps}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Accuracy</span>
                    <span className={`text-lg font-bold ${workout.accuracy >= 95 ? 'text-accent-400' : workout.accuracy >= 90 ? 'text-warning-400' : 'text-primary-400'}`}>
                      {workout.accuracy}%
                    </span>
                  </div>
                </div>

                <div className="mt-4 h-2 bg-dark-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-primary-500 to-accent-500"
                    style={{ width: `${workout.accuracy}%` }}
                  ></div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default UserDashboard
