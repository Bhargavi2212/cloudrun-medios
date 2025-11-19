import { Box, Card, CardContent, Typography, Grid } from '@mui/material';
import { AppLayout } from '../../layout/AppLayout';

export const AdminDashboard = () => {
  return (
    <AppLayout>
      <Box>
        <Typography variant="h4" gutterBottom>
          Admin Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          System Administration and Management
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Overview
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Admin features coming soon...
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </AppLayout>
  );
};

