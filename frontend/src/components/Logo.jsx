import { motion } from 'framer-motion'

const Logo = ({ size = 'md', animated = true, className = '' }) => {
  const sizes = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16',
    xl: 'w-24 h-24',
    '2xl': 'w-32 h-32'
  }

  const LogoImage = () => (
    <img 
      src="/assets/logo.svg" 
      alt="AlphaReps Logo"
      className={`${sizes[size]} ${className}`}
    />
  )

  if (!animated) {
    return <LogoImage />
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.5, rotate: -180 }}
      animate={{ opacity: 1, scale: 1, rotate: 0 }}
      transition={{ 
        type: 'spring',
        stiffness: 200,
        damping: 15
      }}
      whileHover={{ 
        scale: 1.1,
        rotate: 5,
        transition: { duration: 0.2 }
      }}
      className="inline-block"
    >
      <LogoImage />
    </motion.div>
  )
}

export default Logo
