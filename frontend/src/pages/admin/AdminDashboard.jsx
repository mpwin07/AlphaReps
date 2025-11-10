import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Users, Activity, TrendingUp, Award,
  Search, Filter, MoreVertical, LogOut,
  Eye, Trash2, UserCheck
} from 'lucide-react'
import { useAuthStore } from '../../store/authStore'
import { Link } from 'react-router-dom'

const AdminDashboard = () => {
  const { user, logout } = useAuthStore()
  const [searchTerm, setSearchTerm] = useState('')

  const gymStats = [
    { label: 'Total Members', value: '142', change: '+12', icon: Users, color: 'primary' },
    { label: 'Active Today', value: '48', change: '+8', icon: Activity, color: 'accent' },
    { label: 'Total Workouts', value: '1,248', change: '+156', icon: TrendingUp, color: 'warning' },
    { label: 'Avg Accuracy', value: '93%', change: '+2%', icon: Award, color: 'purple' },
  ]

  const recentMembers = [
    { id: 1, name: 'John Doe', email: 'john@example.com', workouts: 24, accuracy: 95, status: 'active', joined: '2 days ago' },
    { id: 2, name: 'Sarah Smith', email: 'sarah@example.com', workouts: 18, accuracy: 92, status: 'active', joined: '1 week ago' },
    { id: 3, name: 'Mike Johnson', email: 'mike@example.com', workouts: 32, accuracy: 97, status: 'active', joined: '2 weeks ago' },
    { id: 4, name: 'Emily Davis', email: 'emily@example.com', workouts: 12, accuracy: 89, status: 'inactive', joined: '3 weeks ago' },
    { id: 5, name: 'Alex Brown', email: 'alex@example.com', workouts: 28, accuracy: 94, status: 'active', joined: '1 month ago' },
  ]

  const topPerformers = [
    { name: 'Mike Johnson', reps: 1240, accuracy: 97 },
    { name: 'John Doe', reps: 1180, accuracy: 95 },
    { name: 'Alex Brown', reps: 1095, accuracy: 94 },
  ]

  return (
    <div className="min-h-screen bg-dark-900">
      {/* Header */}
      <div className="bg-dark-800 border-b border-dark-700">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-display font-black">
                <span className="gradient-text">ADMIN</span>
                <span className="text-white"> DASHBOARD</span>
              </h1>
              <p className="text-gray-400 mt-1">Manage your gym members and analytics</p>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="font-semibold">{user?.name}</p>
                <p className="text-sm text-gray-400">Gym Owner</p>
              </div>
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
        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {gymStats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card-hover"
            >
              <div className="flex items-center justify-between mb-3">
                <stat.icon className={`w-8 h-8 text-${stat.color}-500`} />
                <span className="text-xs font-bold text-accent-400 bg-accent-500/20 px-2 py-1 rounded-full">
                  {stat.change}
                </span>
              </div>
              <p className="text-3xl font-black mb-1">{stat.value}</p>
              <p className="text-sm text-gray-400">{stat.label}</p>
            </motion.div>
          ))}
        </div>

        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* Members Table */}
          <div className="lg:col-span-2 card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">Gym Members</h2>
              
              <div className="flex gap-3">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search members..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="input-field pl-10 py-2 w-64"
                  />
                </div>
                <button className="btn-outline py-2 px-4 flex items-center gap-2">
                  <Filter className="w-4 h-4" />
                  Filter
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-700">
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-400">Member</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-400">Workouts</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-400">Accuracy</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-400">Status</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-gray-400">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {recentMembers.map((member) => (
                    <tr key={member.id} className="border-b border-dark-700/50 hover:bg-dark-700/30 transition-colors">
                      <td className="py-4 px-4">
                        <div>
                          <p className="font-semibold">{member.name}</p>
                          <p className="text-sm text-gray-400">{member.email}</p>
                        </div>
                      </td>
                      <td className="py-4 px-4">
                        <span className="text-accent-400 font-bold">{member.workouts}</span>
                      </td>
                      <td className="py-4 px-4">
                        <span className={`font-bold ${
                          member.accuracy >= 95 ? 'text-accent-400' :
                          member.accuracy >= 90 ? 'text-warning-400' :
                          'text-primary-400'
                        }`}>
                          {member.accuracy}%
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <span className={`badge ${
                          member.status === 'active' ? 'badge-success' : 'badge-warning'
                        }`}>
                          {member.status}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <div className="flex gap-2">
                          <button className="p-2 hover:bg-dark-600 rounded-lg transition-colors" title="View Details">
                            <Eye className="w-4 h-4 text-gray-400" />
                          </button>
                          <button className="p-2 hover:bg-dark-600 rounded-lg transition-colors" title="Delete">
                            <Trash2 className="w-4 h-4 text-gray-400" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Top Performers */}
          <div className="card">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
              <Award className="w-5 h-5 text-warning-500" />
              Top Performers
            </h3>
            <div className="space-y-4">
              {topPerformers.map((performer, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 + index * 0.1 }}
                  className="flex items-center gap-4 p-4 bg-dark-700 rounded-lg hover:bg-dark-600 transition-colors"
                >
                  <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${
                    index === 0 ? 'from-warning-500 to-warning-600' :
                    index === 1 ? 'from-gray-400 to-gray-500' :
                    'from-amber-600 to-amber-700'
                  } flex items-center justify-center font-black text-xl`}>
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold">{performer.name}</p>
                    <p className="text-sm text-gray-400">{performer.reps} reps</p>
                  </div>
                  <div className="text-right">
                    <p className={`text-lg font-bold ${
                      performer.accuracy >= 95 ? 'text-accent-400' : 'text-warning-400'
                    }`}>
                      {performer.accuracy}%
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>

            <button className="w-full mt-6 btn-outline py-3">
              View All Rankings
            </button>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid md:grid-cols-3 gap-6">
          <div className="workout-card cursor-pointer">
            <Users className="w-12 h-12 text-primary-500 mb-4" />
            <h3 className="text-xl font-bold mb-2">Add New Member</h3>
            <p className="text-gray-400 text-sm">Register a new gym member with face recognition</p>
          </div>
          
          <div className="workout-card cursor-pointer">
            <Activity className="w-12 h-12 text-accent-500 mb-4" />
            <h3 className="text-xl font-bold mb-2">View Analytics</h3>
            <p className="text-gray-400 text-sm">Detailed workout analytics and reports</p>
          </div>
          
          <div className="workout-card cursor-pointer">
            <TrendingUp className="w-12 h-12 text-warning-500 mb-4" />
            <h3 className="text-xl font-bold mb-2">Generate Report</h3>
            <p className="text-gray-400 text-sm">Export member performance reports</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AdminDashboard
