import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Clock, CheckCircle } from "lucide-react";

interface HealthcareCardProps {
  title: string;
  description?: string;
  priority?: 'low' | 'normal' | 'high' | 'critical';
  status?: 'waiting' | 'in-progress' | 'completed' | 'cancelled';
  children: React.ReactNode;
  className?: string;
}

export function HealthcareCard({ 
  title, 
  description, 
  priority, 
  status, 
  children, 
  className = "" 
}: HealthcareCardProps) {
  const getPriorityColor = (priority?: string) => {
    switch (priority) {
      case 'critical':
        return 'border-red-500 bg-red-50';
      case 'high':
        return 'border-orange-500 bg-orange-50';
      case 'normal':
        return 'border-blue-500 bg-blue-50';
      case 'low':
        return 'border-green-500 bg-green-50';
      default:
        return 'border-gray-200 bg-white';
    }
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'waiting':
        return <Clock className="w-4 h-4 text-yellow-600" />;
      case 'in-progress':
        return <AlertTriangle className="w-4 h-4 text-blue-600" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      default:
        return null;
    }
  };

  return (
    <Card className={`${getPriorityColor(priority)} ${className}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </div>
          <div className="flex items-center gap-2">
            {priority && priority !== 'normal' && (
              <Badge 
                variant="outline"
                className={
                  priority === 'critical' ? 'border-red-500 text-red-700' :
                  priority === 'high' ? 'border-orange-500 text-orange-700' :
                  'border-green-500 text-green-700'
                }
              >
                {priority.toUpperCase()}
              </Badge>
            )}
            {status && (
              <div className="flex items-center gap-1">
                {getStatusIcon(status)}
                <span className="text-sm capitalize">{status.replace('-', ' ')}</span>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {children}
      </CardContent>
    </Card>
  );
}