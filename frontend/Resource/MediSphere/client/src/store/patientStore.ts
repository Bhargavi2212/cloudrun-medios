import { create } from 'zustand';

export type PatientStatus = 
  | 'AWAITING_VITALS' 
  | 'AWAITING_DOCTOR_ASSIGNMENT' 
  | 'AWAITING_DOCTOR' 
  | 'IN_CONSULTATION' 
  | 'AWAITING_DISCHARGE'
  | 'COMPLETED';

export interface Patient {
  id: string;
  name: string;
  age: number;
  phone: string;
  chiefComplaint: string;
  status: PatientStatus;
  assignedDoctorId?: string;
  assignedDoctorName?: string;
  checkinTime: Date;
  estimatedWaitTime?: number;
  triageLevel?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  vitals?: {
    heartRate: number;
    bloodPressure: string;
    temperature: number;
    weight: number;
    oxygenSaturation: number;
  };
}

interface PatientStore {
  patients: Patient[];
  addPatient: (patient: Omit<Patient, 'id' | 'checkinTime' | 'status'>) => void;
  updatePatientStatus: (patientId: string, status: PatientStatus) => void;
  updatePatientVitals: (patientId: string, vitals: Patient['vitals']) => void;
  assignDoctor: (patientId: string, doctorId: string, doctorName: string) => void;
  getPatientsByStatus: (status: PatientStatus) => Patient[];
  getPatientsByDoctor: (doctorId: string) => Patient[];
}

export const usePatientStore = create<PatientStore>((set, get) => ({
  patients: [
    {
      id: '1',
      name: 'John Smith',
      age: 45,
      phone: '555-0123',
      chiefComplaint: 'Chest pain',
      status: 'AWAITING_VITALS',
      checkinTime: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
      estimatedWaitTime: 20,
    },
    {
      id: '2',
      name: 'Maria Garcia',
      age: 32,
      phone: '555-0456',
      chiefComplaint: 'Fever and cough',
      status: 'AWAITING_DOCTOR_ASSIGNMENT',
      checkinTime: new Date(Date.now() - 30 * 60 * 1000), // 30 minutes ago
      estimatedWaitTime: 35,
      vitals: {
        heartRate: 88,
        bloodPressure: '120/80',
        temperature: 101.2,
        weight: 140,
        oxygenSaturation: 98,
      },
      triageLevel: 'MEDIUM',
    },
    {
      id: '3',
      name: 'Robert Johnson',
      age: 67,
      phone: '555-0789',
      chiefComplaint: 'Shortness of breath',
      status: 'AWAITING_DOCTOR',
      assignedDoctorId: '3',
      assignedDoctorName: 'Dr. Emily Rodriguez',
      checkinTime: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
      estimatedWaitTime: 10,
      vitals: {
        heartRate: 110,
        bloodPressure: '140/90',
        temperature: 98.6,
        weight: 180,
        oxygenSaturation: 94,
      },
      triageLevel: 'HIGH',
    },
  ],

  addPatient: (patientData) => {
    const newPatient: Patient = {
      ...patientData,
      id: Date.now().toString(),
      checkinTime: new Date(),
      status: 'AWAITING_VITALS',
    };
    set((state) => ({
      patients: [...state.patients, newPatient],
    }));
  },

  updatePatientStatus: (patientId, status) => {
    set((state) => ({
      patients: state.patients.map((patient) =>
        patient.id === patientId ? { ...patient, status } : patient
      ),
    }));
  },

  updatePatientVitals: (patientId, vitals) => {
    set((state) => {
      const triageLevel = calculateTriageLevel(vitals);
      return {
        patients: state.patients.map((patient) =>
          patient.id === patientId 
            ? { 
                ...patient, 
                vitals,
                status: 'AWAITING_DOCTOR',
                triageLevel,
                assignedDoctorId: '3', // Auto-assign to Dr. Emily Rodriguez for demo
                assignedDoctorName: 'Dr. Emily Rodriguez',
              } 
            : patient
        ),
      };
    });
  },

  assignDoctor: (patientId, doctorId, doctorName) => {
    set((state) => ({
      patients: state.patients.map((patient) =>
        patient.id === patientId
          ? {
              ...patient,
              assignedDoctorId: doctorId,
              assignedDoctorName: doctorName,
              status: 'AWAITING_DOCTOR',
            }
          : patient
      ),
    }));
  },

  getPatientsByStatus: (status) => {
    return get().patients.filter((patient) => patient.status === status);
  },

  getPatientsByDoctor: (doctorId) => {
    return get().patients.filter((patient) => patient.assignedDoctorId === doctorId);
  },
}));

function calculateTriageLevel(vitals?: Patient['vitals']): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
  if (!vitals) return 'LOW';
  
  const { heartRate, temperature, oxygenSaturation } = vitals;
  
  // Critical conditions
  if (oxygenSaturation < 90 || heartRate > 120 || temperature > 103) {
    return 'CRITICAL';
  }
  
  // High priority conditions
  if (oxygenSaturation < 95 || heartRate > 100 || temperature > 101) {
    return 'HIGH';
  }
  
  // Medium priority conditions
  if (heartRate > 80 || temperature > 99) {
    return 'MEDIUM';
  }
  
  return 'LOW';
}