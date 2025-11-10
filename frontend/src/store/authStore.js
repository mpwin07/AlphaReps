import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useAuthStore = create(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      faceDescriptor: null,
      
      login: (userData, faceDesc) => set({ 
        user: userData, 
        isAuthenticated: true,
        faceDescriptor: faceDesc 
      }),
      
      logout: () => set({ 
        user: null, 
        isAuthenticated: false,
        faceDescriptor: null 
      }),
      
      updateUser: (userData) => set((state) => ({ 
        user: { ...state.user, ...userData } 
      })),
    }),
    {
      name: 'alphareps-auth',
    }
  )
)
