interface DoctorStatusProps {
  doctors: any[];
  isLoading: boolean;
}

export default function DoctorStatus({ doctors, isLoading }: DoctorStatusProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'available':
        return 'bg-green-100 text-green-800';
      case 'busy':
        return 'bg-red-100 text-red-800';
      case 'on_break':
        return 'bg-yellow-100 text-yellow-800';
      case 'emergency':
        return 'bg-orange-100 text-orange-800';
      case 'off_duty':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatStatus = (status: string) => {
    switch (status) {
      case 'on_break':
        return 'On Break';
      case 'off_duty':
        return 'Off Duty';
      default:
        return status.charAt(0).toUpperCase() + status.slice(1);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-6">
        <div className="h-6 bg-gray-200 rounded w-32 mb-6 animate-pulse"></div>
        <div className="space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="flex items-center justify-between animate-pulse">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gray-200 rounded-full"></div>
                <div className="space-y-2">
                  <div className="h-4 bg-gray-200 rounded w-20"></div>
                  <div className="h-3 bg-gray-200 rounded w-16"></div>
                </div>
              </div>
              <div className="h-6 bg-gray-200 rounded w-16"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-6">Doctor Availability</h3>
      
      <div className="space-y-4">
        {doctors.length === 0 ? (
          <div className="text-center py-8">
            <i className="fas fa-user-md text-4xl text-gray-300 mb-4"></i>
            <p className="text-gray-500">No doctors found</p>
          </div>
        ) : (
          doctors.slice(0, 6).map((doctor) => (
            <div key={doctor.id} className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <img 
                  src="https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&h=100" 
                  alt="Doctor Profile" 
                  className="w-10 h-10 rounded-full object-cover"
                />
                <div>
                  <p className="font-medium text-gray-900">Dr. {doctor.userId}</p>
                  <p className="text-sm text-gray-600">{doctor.specialty}</p>
                  {doctor.roomNumber && (
                    <p className="text-xs text-gray-500">Room {doctor.roomNumber}</p>
                  )}
                </div>
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(doctor.status)}`}>
                {formatStatus(doctor.status)}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
