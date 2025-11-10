import { useAuthStore, UserRole } from '../../store/authStore';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { 
  Users, 
  Stethoscope, 
  UserCheck, 
  ClipboardList, 
  LogOut, 
  Hospital,
  Activity,
  FileText,
  Calendar,
  Settings
} from 'lucide-react';
import { useLocation } from 'wouter';

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  roles: UserRole[];
}

const navigationItems: NavItem[] = [
  {
    label: 'Live Patient Queue',
    href: '/dashboard/queue',
    icon: Users,
    roles: ['RECEPTIONIST'],
  },
  {
    label: 'Check-In Patient',
    href: '/dashboard/checkin',
    icon: UserCheck,
    roles: ['RECEPTIONIST'],
  },
  {
    label: 'Triage Queue',
    href: '/dashboard/triage',
    icon: Activity,
    roles: ['NURSE'],
  },
  {
    label: 'My Patients',
    href: '/dashboard/my-patients',
    icon: Stethoscope,
    roles: ['DOCTOR'],
  },
  {
    label: 'Schedule',
    href: '/dashboard/schedule',
    icon: Calendar,
    roles: ['DOCTOR'],
  },
  {
    label: 'Hospital Overview',
    href: '/dashboard/overview',
    icon: Hospital,
    roles: ['ADMIN'],
  },
  {
    label: 'Reports',
    href: '/dashboard/reports',
    icon: FileText,
    roles: ['ADMIN'],
  },
  {
    label: 'Settings',
    href: '/dashboard/settings',
    icon: Settings,
    roles: ['ADMIN'],
  },
];

export default function Sidebar() {
  const { user, logout } = useAuthStore();
  const [location, navigate] = useLocation();

  if (!user) return null;

  const userNavItems = navigationItems.filter(item => 
    item.roles.includes(user.role)
  );

  return (
    <div className="w-64 bg-white border-r border-gray-200 h-screen flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
            <Hospital className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">Medi OS</h1>
            <p className="text-sm text-gray-500">Hospital System</p>
          </div>
        </div>
      </div>

      {/* User Info */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
            <span className="text-sm font-medium text-gray-700">
              {user.name.split(' ').map(n => n[0]).join('')}
            </span>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900">{user.name}</p>
            <p className="text-xs text-gray-500">{user.role}</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {userNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = location === item.href;
            
            return (
              <li key={item.href}>
                <button
                  onClick={() => navigate(item.href)}
                  className={cn(
                    "w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors",
                    isActive
                      ? "bg-blue-50 text-blue-700"
                      : "text-gray-700 hover:bg-gray-50"
                  )}
                  data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Logout */}
      <div className="p-4 border-t border-gray-200">
        <Button 
          variant="ghost" 
          onClick={logout}
          className="w-full justify-start"
          data-testid="button-logout"
        >
          <LogOut className="w-5 h-5 mr-3" />
          Sign Out
        </Button>
      </div>
    </div>
  );
}