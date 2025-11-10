import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { useToast } from '@/components/ui/use-toast'
import { useAuthStore } from '@/store/authStore'

const AccountSettingsPage: React.FC = () => {
  const { user, updateProfile, changePassword, refreshUser } = useAuthStore()
  const { toast } = useToast()

  const [profileForm, setProfileForm] = useState({
    first_name: user?.first_name ?? '',
    last_name: user?.last_name ?? '',
    phone: user?.phone ?? '',
  })

  const [passwordForm, setPasswordForm] = useState({
    current_password: '',
    new_password: '',
    confirm_password: '',
  })

  const [isUpdatingProfile, setIsUpdatingProfile] = useState(false)
  const [isUpdatingPassword, setIsUpdatingPassword] = useState(false)

  const handleProfileSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    try {
      setIsUpdatingProfile(true)
      await updateProfile({
        first_name: profileForm.first_name || undefined,
        last_name: profileForm.last_name || undefined,
        phone: profileForm.phone || undefined,
      })
      await refreshUser()
      toast({ title: 'Profile updated', description: 'Your account details were saved.' })
    } catch (error: any) {
      toast({
        title: 'Unable to update profile',
        description: error?.message || 'Please try again later.',
        variant: 'destructive',
      })
    } finally {
      setIsUpdatingProfile(false)
    }
  }

  const handlePasswordSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (passwordForm.new_password !== passwordForm.confirm_password) {
      toast({ title: 'Passwords do not match', variant: 'destructive' })
      return
    }
    try {
      setIsUpdatingPassword(true)
      await changePassword({
        current_password: passwordForm.current_password,
        new_password: passwordForm.new_password,
      })
      setPasswordForm({ current_password: '', new_password: '', confirm_password: '' })
      toast({ title: 'Password updated', description: 'Use your new password next time you sign in.' })
    } catch (error: any) {
      toast({
        title: 'Unable to change password',
        description: error?.response?.data?.error || error?.message || 'Please verify your current password.',
        variant: 'destructive',
      })
    } finally {
      setIsUpdatingPassword(false)
    }
  }

  return (
    <div className="space-y-8">
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Profile Information</CardTitle>
          <CardDescription>Update your personal details so your care team recognises you.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleProfileSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="first_name" className="text-sm font-medium text-gray-700">
                First name
              </label>
              <Input
                id="first_name"
                value={profileForm.first_name}
                onChange={(event) => setProfileForm((prev) => ({ ...prev, first_name: event.target.value }))}
                placeholder="Jane"
              />
            </div>
            <div>
              <label htmlFor="last_name" className="text-sm font-medium text-gray-700">
                Last name
              </label>
              <Input
                id="last_name"
                value={profileForm.last_name}
                onChange={(event) => setProfileForm((prev) => ({ ...prev, last_name: event.target.value }))}
                placeholder="Doe"
              />
            </div>
            <div className="md:col-span-2">
              <label htmlFor="email" className="text-sm font-medium text-gray-700">
                Email address
              </label>
              <Input id="email" value={user?.email ?? ''} disabled className="bg-gray-100" />
            </div>
            <div className="md:col-span-2">
              <label htmlFor="phone" className="text-sm font-medium text-gray-700">
                Contact phone
              </label>
              <Input
                id="phone"
                value={profileForm.phone}
                onChange={(event) => setProfileForm((prev) => ({ ...prev, phone: event.target.value }))}
                placeholder="+1 (555) 000-0000"
              />
            </div>
            <div className="md:col-span-2 flex justify-end">
              <Button type="submit" disabled={isUpdatingProfile}>
                {isUpdatingProfile ? 'Saving changes...' : 'Save changes'}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <Separator />

      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Password & Security</CardTitle>
          <CardDescription>Keep your account secure by using a strong, unique password.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handlePasswordSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="md:col-span-2">
              <label htmlFor="current_password" className="text-sm font-medium text-gray-700">
                Current password
              </label>
              <Input
                id="current_password"
                type="password"
                required
                value={passwordForm.current_password}
                onChange={(event) =>
                  setPasswordForm((prev) => ({ ...prev, current_password: event.target.value }))
                }
              />
            </div>
            <div>
              <label htmlFor="new_password" className="text-sm font-medium text-gray-700">
                New password
              </label>
              <Input
                id="new_password"
                type="password"
                required
                minLength={8}
                value={passwordForm.new_password}
                onChange={(event) =>
                  setPasswordForm((prev) => ({ ...prev, new_password: event.target.value }))
                }
                placeholder="Minimum 8 characters"
              />
            </div>
            <div>
              <label htmlFor="confirm_password" className="text-sm font-medium text-gray-700">
                Confirm new password
              </label>
              <Input
                id="confirm_password"
                type="password"
                required
                minLength={8}
                value={passwordForm.confirm_password}
                onChange={(event) =>
                  setPasswordForm((prev) => ({ ...prev, confirm_password: event.target.value }))
                }
              />
            </div>
            <div className="md:col-span-2 flex justify-end">
              <Button type="submit" variant="outline" disabled={isUpdatingPassword}>
                {isUpdatingPassword ? 'Updating password...' : 'Update password'}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

export default AccountSettingsPage
