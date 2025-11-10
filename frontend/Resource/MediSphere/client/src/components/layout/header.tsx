interface HeaderProps {
  title: string;
  subtitle: string;
  user: any;
}

export default function Header({ title, subtitle, user }: HeaderProps) {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-medical-blue rounded-lg flex items-center justify-center">
              <i className="fas fa-heartbeat text-white"></i>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">{title}</h1>
              <p className="text-sm text-medical-gray">{subtitle}</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="relative">
              <i className="fas fa-bell text-medical-gray text-lg"></i>
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
            </div>
            <div className="flex items-center space-x-2">
              <img 
                src={user?.profileImageUrl || "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&h=100"} 
                alt="Profile" 
                className="w-8 h-8 rounded-full object-cover"
              />
              <span className="text-sm font-medium text-gray-700">
                {user?.firstName} {user?.lastName}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
