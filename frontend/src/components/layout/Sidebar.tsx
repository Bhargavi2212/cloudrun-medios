import React from 'react'
import { NavLink } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import { Heart, Users, UserCheck, Activity, Settings, User } from 'lucide-react'

interface MenuItem {
  icon: React.ComponentType<{ className?: string }>
  label: string
  href: string
}

export const Sidebar: React.FC = () => {
  const { user, getRoleDisplayName } = useAuthStore()

  const getMenuItems = (): MenuItem[] => {
    const role = user?.role
    switch (role) {
      case 'RECEPTIONIST':
        return [
          { icon: Users, label: 'Main Queue', href: '/receptionist' },
          { icon: UserCheck, label: 'Check-In', href: '/check-in' },
        ]
      case 'NURSE':
        return [
          { icon: Activity, label: 'Triage', href: '/nurse' },
          { icon: Users, label: 'Patient Queue', href: '/nurse/queue' },
        ]
      case 'DOCTOR':
        return [
          { icon: UserCheck, label: 'Consultations', href: '/doctor' },
          { icon: Users, label: 'Patient List', href: '/doctor/patients' },
        ]
      case 'ADMIN':
        return [
          { icon: Users, label: 'Dashboard', href: '/admin' },
          { icon: Settings, label: 'System Settings', href: '/admin/settings' },
        ]
      default:
        return []
    }
  }

  const menuItems = getMenuItems()

  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <Heart className="w-8 h-8 text-blue-600" />
          <div>
            <h2 className="text-lg font-bold text-gray-900">MediOS</h2>
            <p className="text-sm text-gray-600">Healthcare System</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {menuItems.map(({ icon: Icon, label, href }) => (
            <NavLink
              key={href}
              to={href}
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 font-medium'
                    : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                }`
              }
            >
              <Icon className="w-5 h-5" />
              <span>{label}</span>
            </NavLink>
          ))}
        </div>
      </nav>

      <div className="p-4 border-t border-gray-200 space-y-3">
        <NavLink
          to="/account"
          className={({ isActive }) =>
            `flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors ${
              isActive ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            }`
          }
        >
          <Settings className="w-4 h-4" />
          Account settings
        </NavLink>
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-blue-600" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900">{user?.full_name}</p>
            <p className="text-xs text-gray-500">{getRoleDisplayName()}</p>
          </div>
        </div>
      </div>
    </aside>
  )
}