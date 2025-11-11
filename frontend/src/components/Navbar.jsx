import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import Logo from './Logo'
import { Menu, X } from 'lucide-react'
import { useState } from 'react'

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 bg-dark-900/80 backdrop-blur-lg border-b border-dark-800"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <Link to="/" className="flex items-center gap-3 group">
            <Logo size="md" animated={true} />
            <span className="text-2xl font-display font-black">
              <span className="gradient-text">ALPHA</span>
              <span className="text-white">REPS</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            <Link 
              to="/features" 
              className="text-gray-300 hover:text-primary-400 transition-colors font-medium"
            >
              Features
            </Link>
            <Link 
              to="/about" 
              className="text-gray-300 hover:text-primary-400 transition-colors font-medium"
            >
              About
            </Link>
            <Link 
              to="/pricing" 
              className="text-gray-300 hover:text-primary-400 transition-colors font-medium"
            >
              Pricing
            </Link>
            <Link to="/login">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-gradient-to-r from-primary-500 to-accent-500 text-white px-6 py-2 rounded-lg font-bold"
              >
                Get Started
              </motion.button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-white p-2"
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="md:hidden bg-dark-800 border-t border-dark-700"
        >
          <div className="px-4 py-4 space-y-3">
            <Link 
              to="/features" 
              className="block text-gray-300 hover:text-primary-400 transition-colors py-2"
            >
              Features
            </Link>
            <Link 
              to="/about" 
              className="block text-gray-300 hover:text-primary-400 transition-colors py-2"
            >
              About
            </Link>
            <Link 
              to="/pricing" 
              className="block text-gray-300 hover:text-primary-400 transition-colors py-2"
            >
              Pricing
            </Link>
            <Link to="/login" className="block">
              <button className="w-full bg-gradient-to-r from-primary-500 to-accent-500 text-white px-6 py-3 rounded-lg font-bold">
                Get Started
              </button>
            </Link>
          </div>
        </motion.div>
      )}
    </motion.nav>
  )
}

export default Navbar
