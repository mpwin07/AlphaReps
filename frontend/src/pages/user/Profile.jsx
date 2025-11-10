import { useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Camera, Save, Mail, Phone, Calendar } from 'lucide-react'
import { useAuthStore } from '../../store/authStore'

const Profile = () => {
  const { user, updateUser } = useAuthStore()
  const [isEditing, setIsEditing] = useState(false)
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    phone: user?.phone || '',
    joinDate: user?.joinDate || 'Jan 2025'
  })

  const handleSave = () => {
    updateUser(formData)
    setIsEditing(false)
  }

  return (
    <div className="min-h-screen bg-dark-900 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link to="/user/dashboard" className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors">
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Dashboard</span>
          </Link>
        </div>

        {/* Profile Card */}
        <div className="card">
          <div className="flex flex-col md:flex-row gap-8">
            {/* Avatar Section */}
            <div className="flex flex-col items-center">
              <div className="relative">
                <div className="w-32 h-32 rounded-full bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center text-4xl font-black text-white">
                  {user?.name?.charAt(0) || 'U'}
                </div>
                <button className="absolute bottom-0 right-0 w-10 h-10 bg-primary-500 rounded-full flex items-center justify-center hover:bg-primary-600 transition-colors">
                  <Camera className="w-5 h-5 text-white" />
                </button>
              </div>
              <h2 className="text-2xl font-bold mt-4">{user?.name}</h2>
              <p className="text-gray-400">{user?.role === 'admin' ? 'Admin' : 'Member'}</p>
            </div>

            {/* Info Section */}
            <div className="flex-1">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold">Personal Information</h3>
                <button
                  onClick={() => setIsEditing(!isEditing)}
                  className="btn-outline py-2 px-4"
                >
                  {isEditing ? 'Cancel' : 'Edit Profile'}
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    <Mail className="w-4 h-4 inline mr-2" />
                    Email
                  </label>
                  {isEditing ? (
                    <input
                      type="email"
                      value={formData.email}
                      onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                      className="input-field w-full"
                    />
                  ) : (
                    <p className="text-lg">{user?.email || 'Not set'}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    <Phone className="w-4 h-4 inline mr-2" />
                    Phone
                  </label>
                  {isEditing ? (
                    <input
                      type="tel"
                      value={formData.phone}
                      onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                      className="input-field w-full"
                    />
                  ) : (
                    <p className="text-lg">{formData.phone || 'Not set'}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    <Calendar className="w-4 h-4 inline mr-2" />
                    Member Since
                  </label>
                  <p className="text-lg">{formData.joinDate}</p>
                </div>
              </div>

              {isEditing && (
                <button
                  onClick={handleSave}
                  className="btn-primary mt-6 w-full flex items-center justify-center gap-2"
                >
                  <Save className="w-5 h-5" />
                  Save Changes
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid md:grid-cols-3 gap-6 mt-6">
          <div className="card text-center">
            <p className="text-4xl font-black text-primary-500 mb-2">24</p>
            <p className="text-gray-400">Total Workouts</p>
          </div>
          <div className="card text-center">
            <p className="text-4xl font-black text-accent-500 mb-2">1,240</p>
            <p className="text-gray-400">Total Reps</p>
          </div>
          <div className="card text-center">
            <p className="text-4xl font-black text-warning-500 mb-2">94%</p>
            <p className="text-gray-400">Avg Accuracy</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Profile
