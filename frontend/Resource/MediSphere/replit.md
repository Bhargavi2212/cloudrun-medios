# Hospital Management System (Medi OS)

## Overview

This is a **frontend-only** hospital management system built with modern web technologies following a "Role-First" design philosophy. The application supports role-based authentication with tailored interfaces for different healthcare roles (RECEPTIONIST, NURSE, DOCTOR, ADMIN). Each user sees only the information and tools necessary for their immediate tasks.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (January 17, 2025)

✓ **Complete pivot to frontend-only architecture** per user request
✓ **Implemented Role-First design philosophy** with tailored interfaces
✓ **Created role-based authentication** using Zustand for state management
✓ **Built comprehensive patient workflow** with realistic demo data
✓ **Designed modern, professional UI** following attached specifications

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Routing**: Wouter for client-side routing
- **State Management**: TanStack Query (React Query) for server state management
- **UI Framework**: shadcn/ui components with Radix UI primitives
- **Styling**: Tailwind CSS with custom medical theme colors
- **Build Tool**: Vite for development and bundling

### Backend Architecture
- **Runtime**: Node.js with Express.js
- **Language**: TypeScript with ES modules
- **Database ORM**: Drizzle ORM
- **Database**: PostgreSQL (configured for Neon serverless)
- **Authentication**: Replit OIDC with Passport.js
- **Session Management**: Express sessions with PostgreSQL store

### Project Structure
- **Monorepo Layout**: Client, server, and shared code in separate directories
- **Client**: `/client` - React frontend application
- **Server**: `/server` - Express backend API
- **Shared**: `/shared` - Shared TypeScript schemas and types

## Key Components

### Authentication System
- **Provider**: Replit OIDC integration
- **Strategy**: OpenID Connect with Passport.js
- **Session Storage**: PostgreSQL-backed sessions using connect-pg-simple
- **User Management**: Role-based access control with user upserts

### Database Schema
- **Users**: Core user accounts with role-based permissions
- **Doctors/Patients/Staff**: Role-specific profile tables
- **Queue Management**: Patient check-in and triage system
- **Rooms**: Real-time room status tracking
- **Vitals**: Patient vital signs recording
- **Appointments**: Scheduling system
- **Medical Records**: Patient history and documentation
- **Staff Tasks**: Nurse and staff task management

### API Architecture
- **RESTful Design**: Standard HTTP methods with JSON payloads
- **Authentication Middleware**: Protected routes requiring valid sessions
- **Role-based Access**: Different endpoints accessible by different user roles
- **Real-time Updates**: Query invalidation for live data updates

### UI Components
- **Design System**: Consistent medical-themed UI components
- **Responsive Design**: Mobile-first approach with desktop optimization
- **Role-based Dashboards**: Customized interfaces for each user type
- **Modals and Forms**: Patient check-in, vitals recording, task management

## Data Flow

1. **Authentication Flow**: Users authenticate via Replit OIDC, sessions stored in PostgreSQL
2. **Role Resolution**: User roles determine accessible features and data
3. **API Requests**: Frontend makes authenticated requests to Express API
4. **Database Queries**: Drizzle ORM handles PostgreSQL interactions
5. **Real-time Updates**: TanStack Query manages cache invalidation and refetching

## External Dependencies

### Core Dependencies
- **Database**: Neon PostgreSQL serverless database
- **Authentication**: Replit OIDC service
- **UI Components**: Radix UI primitives
- **Icons**: Font Awesome (referenced in CSS)
- **Styling**: Tailwind CSS with PostCSS

### Development Tools
- **TypeScript**: Type safety across frontend and backend
- **Vite**: Fast development server and build tool
- **ESBuild**: Production bundling for server code
- **Drizzle Kit**: Database migrations and schema management

## Deployment Strategy

### Build Process
1. **Frontend Build**: Vite builds client code to `/dist/public`
2. **Backend Build**: ESBuild bundles server code to `/dist/index.js`
3. **Database Setup**: Drizzle migrations run on deployment

### Environment Configuration
- **DATABASE_URL**: PostgreSQL connection string (required)
- **SESSION_SECRET**: Session encryption key (required)
- **REPL_ID**: Replit environment identifier
- **ISSUER_URL**: OIDC issuer URL (defaults to Replit)

### Production Considerations
- **Session Security**: Secure cookie settings for HTTPS
- **Database Connection**: Connection pooling with Neon serverless
- **Static Assets**: Vite-built assets served via Express
- **Error Handling**: Centralized error middleware

### Development vs Production
- **Development**: Vite dev server with HMR and middleware mode
- **Production**: Express serves static files and API routes
- **Database**: Same PostgreSQL setup for both environments