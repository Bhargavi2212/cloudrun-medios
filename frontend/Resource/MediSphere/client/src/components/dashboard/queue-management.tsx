interface QueueManagementProps {
  queue: any[];
  isLoading: boolean;
  userRole: string;
}

export default function QueueManagement({ queue, isLoading, userRole }: QueueManagementProps) {
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'emergency':
        return 'bg-red-100 text-red-800';
      case 'urgent':
        return 'bg-orange-100 text-orange-800';
      case 'high':
        return 'bg-yellow-100 text-yellow-800';
      case 'normal':
        return 'bg-green-100 text-green-800';
      case 'low':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="h-6 bg-gray-200 rounded w-32 animate-pulse"></div>
          <div className="h-4 bg-gray-200 rounded w-20 animate-pulse"></div>
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="p-4 bg-gray-50 rounded-lg animate-pulse">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gray-200 rounded-full"></div>
                  <div className="space-y-2">
                    <div className="h-4 bg-gray-200 rounded w-24"></div>
                    <div className="h-3 bg-gray-200 rounded w-16"></div>
                  </div>
                </div>
                <div className="h-6 bg-gray-200 rounded w-16"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Live Patient Queue</h3>
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
          <span className="text-sm text-gray-600">Live Updates</span>
        </div>
      </div>
      
      <div className="space-y-4">
        {queue.length === 0 ? (
          <div className="text-center py-8">
            <i className="fas fa-clipboard-list text-4xl text-gray-300 mb-4"></i>
            <p className="text-gray-500">No patients in queue</p>
          </div>
        ) : (
          queue.slice(0, 5).map((item, index) => (
            <div key={item.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className="w-10 h-10 bg-medical-blue rounded-full flex items-center justify-center">
                  <span className="font-semibold text-white">{item.queuePosition || index + 1}</span>
                </div>
                <div>
                  <p className="font-medium text-gray-900">{item.patientId}</p>
                  <p className="text-sm text-gray-600">{item.reasonForVisit}</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${getPriorityColor(item.priority)}`}>
                  {item.priority}
                </span>
                <span className="text-sm text-gray-600">
                  {item.estimatedWaitTime ? `${item.estimatedWaitTime} min` : 'N/A'}
                </span>
                {item.doctorId && (
                  <span className="text-sm font-medium text-medical-blue">
                    Dr. Assigned
                  </span>
                )}
              </div>
            </div>
          ))
        )}
        
        {queue.length > 5 && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <button className="w-full py-2 text-medical-blue hover:bg-blue-50 rounded-lg font-medium">
              View All Queue Items ({queue.length} total)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
