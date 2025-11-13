import React, { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useToast } from '@/components/ui/use-toast'
import { authAPI } from '@/services/api'
import { ROLE_CONFIG } from '@/store/authStore'
import { UserRole } from '@/types'
import { Check, ShieldAlert } from 'lucide-react'

const roleOptions: UserRole[] = ['RECEPTIONIST', 'NURSE', 'DOCTOR', 'ADMIN']

const roleDescriptions: Record<UserRole, string> = {
  RECEPTIONIST: 'Manage patient intake, new registrations, and queue flow at the front desk.',
  NURSE: 'Capture vitals, triage patients, and upload medical documentation.',
  DOCTOR: 'Review consultations, leverage AI Scribe, and finalize clinical documentation.',
  ADMIN: 'Oversee system-wide configuration, user management, and analytics.',
}

const minPasswordLength = 8

const calculatePasswordStrength = (password: string) => {
  const trimmed = password.trim()
  if (!trimmed) {
    return { label: 'None', score: 0, color: 'text-gray-400' }
  }

  const baseChecks = passwordRequirementChecks.map((requirement) => requirement.test(trimmed))
  const extendedChecks = [trimmed.length >= 12]
  const score = [...baseChecks, ...extendedChecks].filter(Boolean).length

  if (score >= 5) {
    return { label: 'Strong', score, color: 'text-emerald-600' }
  }
  if (score >= 3) {
    return { label: 'Medium', score, color: 'text-amber-600' }
  }
  return { label: 'Weak', score, color: 'text-rose-600' }
}

const passwordRequirementChecks: Array<{ label: string; test: (value: string) => boolean }> = [
  { label: `At least ${minPasswordLength} characters`, test: (value) => value.length >= minPasswordLength },
  { label: 'One uppercase letter', test: (value) => /[A-Z]/.test(value) },
  { label: 'One lowercase letter', test: (value) => /[a-z]/.test(value) },
  { label: 'One number', test: (value) => /\d/.test(value) },
  { label: 'One special character', test: (value) => /[^A-Za-z0-9]/.test(value) },
]

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

    if (!form.firstName.trim() || !form.lastName.trim()) {
      toast({ title: 'Missing name', description: 'Please enter both first and last name.', variant: 'destructive' })
      return
    }

    const passwordStrength = calculatePasswordStrength(form.password)
    if (passwordStrength.label === 'Weak') {
      toast({
        title: 'Password too weak',
        description: 'Please choose a stronger password that meets all security requirements.',
        variant: 'destructive',
      })
      return
    }

    try {
      setIsSubmitting(true)
      const payload = {
        email: form.email.trim().toLowerCase(),
        password: form.password,
        first_name: form.firstName.trim(),
        last_name: form.lastName.trim(),
        role: form.role,
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

  const passwordStrength = useMemo(() => calculatePasswordStrength(form.password), [form.password])
  const selectedRoleDescription = roleDescriptions[form.role]

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="w-full max-w-3xl space-y-4">
        <div className="rounded-xl bg-white/90 border border-blue-200 shadow-md p-5 md:p-6">
          <div className="flex items-start space-x-3">
            <div className="mt-0.5 text-blue-600">
              <ShieldAlert className="h-6 w-6" />
            </div>
            <div className="space-y-2">
              <p className="font-semibold text-blue-800 text-sm uppercase tracking-wide">Demo Accounts</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-y-1 gap-x-6 text-sm text-gray-700">
                <span>
                  <span className="font-medium text-gray-900">Receptionist:</span> receptionist@medios.ai / Password123!
                </span>
                <span>
                  <span className="font-medium text-gray-900">Nurse:</span> nurse@medios.ai / Password123!
                </span>
                <span>
                  <span className="font-medium text-gray-900">Doctor:</span> doctor@medios.ai / Password123!
                </span>
                <span>
                  <span className="font-medium text-gray-900">Admin:</span> admin@medios.ai / Password123!
                </span>
              </div>
              <p className="text-xs text-gray-500">
                Use a demo credential for quick testing or create your own account with the role selector below.
              </p>
            </div>
          </div>
        </div>

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
                <div className="mt-2 flex items-center justify-between text-xs font-medium">
                  <span className={`uppercase tracking-wide ${passwordStrength.color}`}>Strength: {passwordStrength.label}</span>
                  <span className="text-gray-500">Use a mix of letters, numbers & symbols</span>
                </div>
                <ul className="mt-2 space-y-1 text-xs text-gray-500">
                  {passwordRequirementChecks.map(({ label, test }) => {
                    const requirementMet = test(form.password.trim())
                    return (
                      <li key={label} className="flex items-center space-x-2">
                        <Check className={`h-3.5 w-3.5 ${requirementMet ? 'text-emerald-600' : 'text-gray-400'}`} />
                        <span className={requirementMet ? 'text-emerald-600' : undefined}>{label}</span>
                      </li>
                    )
                  })}
                </ul>
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
              <p className="mt-3 text-sm text-gray-600">{selectedRoleDescription}</p>
              <p className="mt-2 text-xs text-gray-500">
                Access is restricted based on the role you choose. Administrators can elevate permissions later if needed.
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
