import { Link } from 'react-router-dom'
import { ArrowLeft, TrendingUp, Calendar, Target } from 'lucide-react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const Analytics = () => {
  const weeklyData = [
    { day: 'Mon', reps: 120, accuracy: 92 },
    { day: 'Tue', reps: 150, accuracy: 94 },
    { day: 'Wed', reps: 100, accuracy: 90 },
    { day: 'Thu', reps: 180, accuracy: 96 },
    { day: 'Fri', reps: 160, accuracy: 95 },
    { day: 'Sat', reps: 200, accuracy: 97 },
    { day: 'Sun', reps: 140, accuracy: 93 },
  ]

  const exerciseBreakdown = [
    { name: 'Push-ups', value: 35 },
    { name: 'Squats', value: 28 },
    { name: 'Curls', value: 22 },
    { name: 'Press', value: 15 },
  ]

  return (
    <div className="min-h-screen bg-dark-900 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <Link to="/user/dashboard" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Dashboard</span>
          </Link>
          <h1 className="text-3xl font-display font-bold">
            <span className="gradient-text">ANALYTICS</span>
          </h1>
          <div className="w-32"></div>
        </div>

        {/* Weekly Progress */}
        <div className="card mb-6">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-primary-500" />
            Weekly Progress
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={weeklyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="day" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Line type="monotone" dataKey="reps" stroke="#FF3B3B" strokeWidth={3} />
              <Line type="monotone" dataKey="accuracy" stroke="#14B8A6" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Exercise Distribution */}
          <div className="card">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Target className="w-6 h-6 text-accent-500" />
              Exercise Distribution
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={exerciseBreakdown}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Bar dataKey="value" fill="url(#colorGradient)" radius={[8, 8, 0, 0]} />
                <defs>
                  <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#FF3B3B" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#14B8A6" stopOpacity={0.8}/>
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Monthly Stats */}
          <div className="card">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Calendar className="w-6 h-6 text-warning-500" />
              This Month
            </h2>
            <div className="space-y-6">
              <div className="stat-box">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-400">Total Workouts</span>
                  <span className="text-2xl font-black text-primary-500">24</span>
                </div>
                <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                  <div className="h-full bg-primary-500" style={{ width: '80%' }}></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">Goal: 30 workouts</p>
              </div>

              <div className="stat-box">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-400">Total Reps</span>
                  <span className="text-2xl font-black text-accent-500">1,240</span>
                </div>
                <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                  <div className="h-full bg-accent-500" style={{ width: '90%' }}></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">Goal: 1,500 reps</p>
              </div>

              <div className="stat-box">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-400">Average Accuracy</span>
                  <span className="text-2xl font-black text-warning-500">94%</span>
                </div>
                <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                  <div className="h-full bg-warning-500" style={{ width: '94%' }}></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">Goal: 95% accuracy</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics
