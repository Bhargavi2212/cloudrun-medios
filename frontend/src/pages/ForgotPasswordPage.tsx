import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useToast } from '@/components/ui/use-toast'
import { authAPI } from '@/services/api'

const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [resetToken, setResetToken] = useState<string | null>(null)
  const { toast } = useToast()
  const navigate = useNavigate()

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    try {
      setIsSubmitting(true)
      const response = await authAPI.forgotPassword(email.trim().toLowerCase())
      const data = response.data
      
      if (!data?.success) {
        throw new Error(data?.error || 'Failed to send reset link')
      }
      
      // In development, the token is returned in the response
      // In production, this would be sent via email
      const token = data.data?.reset_token
      if (token) {
        // For development: show token and allow navigation to reset page
        setResetToken(token)
        toast({
          title: 'Reset Token Generated',
          description: 'In production, this would be sent via email. Redirecting to reset page...',
        })
        // Navigate to reset page with token
        setTimeout(() => {
          navigate(`/reset-password?token=${encodeURIComponent(token)}`, { replace: true })
        }, 1500)
      } else {
        toast({
          title: 'Reset Email Sent',
          description: 'If the email exists in our system, you will receive reset instructions shortly.',
        })
        setEmail('')
      }
    } catch (error: any) {
      toast({
        title: 'Unable to send reset link',
        description: error?.response?.data?.error || error?.message || 'Please try again later.',
        variant: 'destructive',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="w-full max-w-lg">
        <Card className="shadow-xl border-0">
          <CardHeader className="space-y-1 pb-6">
            <CardTitle className="text-2xl text-center font-semibold">Forgot Password</CardTitle>
            <CardDescription className="text-center text-gray-600">
              Enter the email associated with your MediOS account and we will send reset instructions.
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label htmlFor="email" className="text-sm font-medium text-gray-700">
                  Email Address
                </label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="name@hospital.org"
                  required
                  className="h-12"
                />
              </div>

              <Button type="submit" disabled={isSubmitting} className="w-full h-12 bg-blue-600 hover:bg-blue-700">
                {isSubmitting ? 'Sending reset link...' : 'Send reset link'}
              </Button>
            </form>

            <div className="text-center mt-6 text-sm text-gray-600">
              Remembered your password?{' '}
              <Link to="/login" className="text-blue-600 hover:text-blue-700 font-medium">
                Return to sign in
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default ForgotPasswordPage
