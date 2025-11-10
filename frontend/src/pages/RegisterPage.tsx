import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useToast } from '@/components/ui/use-toast'
import { authAPI } from '@/services/api'
import { ROLE_CONFIG } from '@/store/authStore'
import { UserRole } from '@/types'

const roleOptions: UserRole[] = ['RECEPTIONIST', 'NURSE', 'DOCTOR', 'ADMIN']

const RegisterPage: React.FC = () => {
  const [form, setForm] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    role: 'RECEPTIONIST' as UserRole,
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const { toast } = useToast()
  const navigate = useNavigate()

  const handleChange = (field: keyof typeof form) => (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setForm((prev) => ({ ...prev, [field]: event.target.value }))
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (form.password !== form.confirmPassword) {
      toast({ title: 'Passwords do not match', variant: 'destructive' })
      return
    }

    try {
      setIsSubmitting(true)
      const payload = {
        email: form.email.trim().toLowerCase(),
        password: form.password,
        first_name: form.firstName || undefined,
        last_name: form.lastName || undefined,
        roles: [form.role],
      }
      const response = await authAPI.register(payload)
      if (!response.data?.success) {
        throw new Error(response.data?.error || 'Registration failed')
      }

      toast({
        title: 'Account Created',
        description: 'You can now sign in with your credentials.',
      })
      navigate('/login', { replace: true })
    } catch (error) {
      const errorMessage = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { error?: string } } })?.response?.data?.error || 'Could not create account.'
      toast({
        title: 'Registration Failed',
        description: errorMessage,
        variant: 'destructive',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="w-full max-w-2xl">
        <Card className="shadow-xl border-0">
          <CardHeader className="space-y-1 pb-6">
            <CardTitle className="text-2xl text-center font-semibold">Create Your MediOS Account</CardTitle>
            <CardDescription className="text-center text-gray-600">
              Invite-only registration for healthcare team members
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="md:col-span-1">
                <label className="text-sm font-medium text-gray-700" htmlFor="firstName">
                  First Name
                </label>
                <Input id="firstName" value={form.firstName} onChange={handleChange('firstName')} placeholder="Jane" />
              </div>

              <div className="md:col-span-1">
                <label className="text-sm font-medium text-gray-700" htmlFor="lastName">
                  Last Name
                </label>
                <Input id="lastName" value={form.lastName} onChange={handleChange('lastName')} placeholder="Doe" />
              </div>

              <div className="md:col-span-2">
                <label className="text-sm font-medium text-gray-700" htmlFor="email">
                  Email Address
                </label>
                <Input
                  id="email"
                  type="email"
                  value={form.email}
                  onChange={handleChange('email')}
                  required
                  placeholder="name@hospital.org"
                />
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700" htmlFor="password">
                  Password
                </label>
                <Input
                  id="password"
                  type="password"
                  value={form.password}
                  onChange={handleChange('password')}
                  required
                  minLength={8}
                  placeholder="Minimum 8 characters"
                />
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700" htmlFor="confirmPassword">
                  Confirm Password
                </label>
                <Input
                  id="confirmPassword"
                  type="password"
                  value={form.confirmPassword}
                  onChange={handleChange('confirmPassword')}
                  required
                  minLength={8}
                  placeholder="Re-enter password"
                />
              </div>

              <div className="md:col-span-2">
                <label className="text-sm font-medium text-gray-700" htmlFor="role">
                  Role
                </label>
                <select
                  id="role"
                  value={form.role}
                  onChange={handleChange('role')}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 h-12"
                >
                  {roleOptions.map((role) => (
                    <option key={role} value={role}>
                      {ROLE_CONFIG[role].displayName}
                    </option>
                  ))}
                </select>
                <p className="mt-2 text-xs text-gray-500">
                  Access is restricted based on assigned role. Administrators can update permissions later.
                </p>
              </div>

              <div className="md:col-span-2">
                <Button type="submit" disabled={isSubmitting} className="w-full h-12 bg-blue-600 hover:bg-blue-700">
                  {isSubmitting ? 'Creating Account...' : 'Create Account'}
                </Button>
              </div>
            </form>

            <div className="text-center mt-6 text-sm text-gray-600">
              Already have an account?{' '}
              <Link to="/login" className="text-blue-600 hover:text-blue-700 font-medium">
                Sign in instead
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default RegisterPage
