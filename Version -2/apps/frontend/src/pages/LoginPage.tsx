import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  TextField,
  Typography,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useAuthStore } from '../store/authStore';

export const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { login, isLoading, error, getDefaultRoute } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    
    try {
      await login(email, password);
      const defaultRoute = getDefaultRoute();
      navigate(defaultRoute);
    } catch (err: unknown) {
      setLocalError(err instanceof Error ? err.message : 'Login failed');
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        bgcolor: 'background.default',
      }}
    >
      <Card sx={{ maxWidth: 400, width: '100%', m: 2 }}>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom align="center">
            Medi OS Portal
          </Typography>
          <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 3 }}>
            Sign in to your account
          </Typography>

          {(error || localError) && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {localError || error}
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              margin="normal"
              required
              autoComplete="email"
              disabled={isLoading}
            />
            <TextField
              fullWidth
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              margin="normal"
              required
              autoComplete="current-password"
              disabled={isLoading}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              disabled={isLoading}
            >
              {isLoading ? <CircularProgress size={24} /> : 'Sign In'}
            </Button>
          </form>

          <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Demo Credentials
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box>
                <Typography variant="caption" fontWeight="bold">Receptionist:</Typography>
                <Typography
                  variant="caption"
                  sx={{ ml: 1, cursor: 'pointer', color: 'primary.main', textDecoration: 'underline' }}
                  onClick={() => {
                    setEmail('receptionist@hospital.com');
                    setPassword('demo123');
                  }}
                >
                  receptionist@hospital.com / demo123
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" fontWeight="bold">Nurse:</Typography>
                <Typography
                  variant="caption"
                  sx={{ ml: 1, cursor: 'pointer', color: 'primary.main', textDecoration: 'underline' }}
                  onClick={() => {
                    setEmail('nurse@hospital.com');
                    setPassword('demo123');
                  }}
                >
                  nurse@hospital.com / demo123
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" fontWeight="bold">Doctor:</Typography>
                <Typography
                  variant="caption"
                  sx={{ ml: 1, cursor: 'pointer', color: 'primary.main', textDecoration: 'underline' }}
                  onClick={() => {
                    setEmail('doctor@hospital.com');
                    setPassword('demo123');
                  }}
                >
                  doctor@hospital.com / demo123
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" fontWeight="bold">Admin:</Typography>
                <Typography
                  variant="caption"
                  sx={{ ml: 1, cursor: 'pointer', color: 'primary.main', textDecoration: 'underline' }}
                  onClick={() => {
                    setEmail('admin@hospital.com');
                    setPassword('demo123');
                  }}
                >
                  admin@hospital.com / demo123
                </Typography>
              </Box>
            </Box>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
              Click any credential to auto-fill
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

