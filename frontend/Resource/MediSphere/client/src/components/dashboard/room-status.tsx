interface RoomStatusProps {
  rooms: any[];
  isLoading: boolean;
}

export default function RoomStatus({ rooms, isLoading }: RoomStatusProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'available':
        return 'bg-green-50 border-green-200 text-green-800';
      case 'occupied':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'cleaning':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'maintenance':
        return 'bg-gray-50 border-gray-200 text-gray-800';
      case 'reserved':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  const getStatusDot = (status: string) => {
    switch (status) {
      case 'available':
        return 'bg-green-500';
      case 'occupied':
        return 'bg-red-500';
      case 'cleaning':
        return 'bg-yellow-500';
      case 'maintenance':
        return 'bg-gray-500';
      case 'reserved':
        return 'bg-blue-500';
      default:
        return 'bg-gray-500';
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-6">
        <div className="h-6 bg-gray-200 rounded w-32 mb-6 animate-pulse"></div>
        <div className="grid grid-cols-4 gap-3">
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <div key={i} className="p-3 border border-gray-200 rounded-lg animate-pulse">
              <div className="flex items-center justify-between mb-2">
                <div className="h-4 bg-gray-200 rounded w-8"></div>
                <div className="w-2 h-2 bg-gray-200 rounded-full"></div>
              </div>
              <div className="h-3 bg-gray-200 rounded w-12"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-6">Room Occupancy</h3>
      
      <div className="grid grid-cols-4 gap-3">
        {rooms.slice(0, 16).map((room) => (
          <div key={room.id} className={`relative p-3 rounded-lg text-center border ${getStatusColor(room.status)}`}>
            <div className="text-xs font-medium mb-1">{room.roomNumber}</div>
            <div className={`w-2 h-2 rounded-full mx-auto mt-1 ${getStatusDot(room.status)}`}></div>
            <div className="text-xs mt-1 capitalize">{room.status}</div>
            {room.totalBeds > 1 && (
              <div className="text-xs text-gray-600 mt-1">
                {room.occupiedBeds}/{room.totalBeds}
              </div>
            )}
          </div>
        ))}
      </div>
      
      <div className="mt-4 flex items-center justify-between text-sm">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 bg-green-500 rounded-full"></span>
            <span className="text-gray-600">Available</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 bg-red-500 rounded-full"></span>
            <span className="text-gray-600">Occupied</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 bg-yellow-500 rounded-full"></span>
            <span className="text-gray-600">Cleaning</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 bg-gray-500 rounded-full"></span>
            <span className="text-gray-600">Maintenance</span>
          </div>
        </div>
        {rooms.length > 16 && (
          <button className="text-medical-blue hover:underline">
            View All Rooms
          </button>
        )}
      </div>
    </div>
  );
}
