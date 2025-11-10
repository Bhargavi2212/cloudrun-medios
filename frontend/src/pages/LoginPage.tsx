import React, { useState, useEffect } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useToast } from '@/components/ui/use-toast'
import { useAuthStore } from '@/store/authStore'
import { Eye, EyeOff, Shield, Heart, Mail, Lock } from 'lucide-react'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const { login, isLoading, error, isAuthenticated, user, getDefaultRoute, clearError } = useAuthStore()
  const { toast } = useToast()
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    if (isAuthenticated && user) {
      const redirectTo = location.state?.from?.pathname || getDefaultRoute()
      navigate(redirectTo, { replace: true })
    }
  }, [isAuthenticated, user, navigate, location, getDefaultRoute])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await login(email.trim(), password)
      const currentUser = useAuthStore.getState().user
      const defaultRoute = useAuthStore.getState().getDefaultRoute()
      toast({
        title: 'Login Successful',
        description: `Welcome back, ${currentUser?.full_name ?? 'clinician'}!`,
      })
      const redirectTo = location.state?.from?.pathname || defaultRoute
      navigate(redirectTo, { replace: true })
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Invalid credentials. Please try again.'
      toast({
        title: 'Login Failed',
        description: errorMessage,
        variant: 'destructive',
      })
    }
  }

  useEffect(() => {
    if (error) {
      toast({ title: 'Authentication Required', description: error, variant: 'destructive' })
      clearError()
    }
  }, [error, toast, clearError])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl">
            <Heart className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">MediOS</h1>
          <p className="text-gray-600 text-lg">AI-Powered Healthcare Platform</p>
          <p className="text-gray-500 text-sm mt-1">Role-Based Medical Workflow System</p>
        </div>

        <Card className="shadow-xl border-0">
          <CardHeader className="space-y-1 pb-6">
            <CardTitle className="text-2xl text-center font-semibold">Welcome Back</CardTitle>
            <CardDescription className="text-center text-gray-600">
              Sign in to access your medical workspace
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="email" className="text-sm font-medium text-gray-700">
                  Email Address
                </label>
                <div className="relative">
                  <Input
                    id="email"
                    type="email"
                    placeholder="name@hospital.org"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="w-full h-12 pl-10"
                  />
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="password" className="text-sm font-medium text-gray-700">
                  Password
                </label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="w-full h-12 pl-10 pr-10"
                  />
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4 text-gray-400" /> : <Eye className="h-4 w-4 text-gray-400" />}
                  </Button>
                </div>
              </div>

              <div className="flex items-center justify-between text-sm">
                <Link to="/forgot-password" className="text-blue-600 hover:text-blue-700 font-medium">
                  Forgot password?
                </Link>
                <Link to="/register" className="text-blue-600 hover:text-blue-700 font-medium">
                  Create account
                </Link>
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-12 bg-blue-600 hover:bg-blue-700 text-white font-medium"
              >
                {isLoading ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Signing In...</span>
                  </div>
                ) : (
                  'Sign In to Continue'
                )}
              </Button>
            </form>

            <div className="pt-4 border-t border-gray-200">
              <div className="text-center">
                <p className="text-sm font-medium text-gray-700 mb-3">Demo Credentials</p>
                <div className="space-y-1 text-xs text-gray-600">
                  <p>
                    <span className="font-medium">Receptionist:</span> receptionist@medios.ai / password
                  </p>
                  <p>
                    <span className="font-medium">Nurse:</span> nurse@medios.ai / password
                  </p>
                  <p>
                    <span className="font-medium">Doctor:</span> doctor@medios.ai / password
                  </p>
                  <p>
                    <span className="font-medium">Admin:</span> admin@medios.ai / password
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="text-center mt-6">
          <p className="text-sm text-gray-500">MediOS v1.0 - Powered by AI Technology</p>
          <div className="flex items-center justify-center mt-2 space-x-1">
            <Shield className="w-3 h-3 text-gray-400" />
            <span className="text-xs text-gray-400">Secure Medical Platform</span>
          </div>
        </div>
      </div>
    </div>
  )
}
