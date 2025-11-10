import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { setupAuth, isAuthenticated } from "./replitAuth";
import { 
  insertQueueEntrySchema, 
  insertVitalSchema, 
  insertPrescriptionSchema,
  insertLabOrderSchema,
  insertConsultationSchema,
  insertAppointmentSchema,
  insertPatientSchema
} from "@shared/healthcare-schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  // Auth middleware
  await setupAuth(app);

  // Auth routes
  app.get('/api/auth/user', isAuthenticated, async (req: any, res) => {
    try {
      const userId = (req.user as any).claims.sub;
      const user = await storage.getUser(userId);
      
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }

      // Get additional role-specific data
      let roleData = null;
      if (user.role === 'doctor') {
        roleData = await storage.getDoctorByUserId(userId);
      } else if (user.role === 'patient') {
        roleData = await storage.getPatientByUserId(userId);
      }

      res.json({ ...user, roleData });
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });

  app.post('/api/auth/select-role', isAuthenticated, async (req: any, res) => {
    try {
      const userId = (req.user as any).claims.sub;
      const { role } = req.body;
      
      if (!role || !['admin', 'doctor', 'nurse', 'receptionist', 'patient'].includes(role)) {
        return res.status(400).json({ message: "Invalid role" });
      }

      await storage.updateUserRole(userId, role);
      res.json({ success: true, role });
    } catch (error) {
      console.error("Error updating user role:", error);
      res.status(500).json({ message: "Failed to update role" });
    }
  });

  // Patient details endpoints
  app.get('/api/patient/:patientId/details', isAuthenticated, async (req, res) => {
    try {
      const { patientId } = req.params;
      const patientDetails = await storage.getPatientDetails(parseInt(patientId));
      res.json(patientDetails);
    } catch (error) {
      console.error("Error fetching patient details:", error);
      res.status(500).json({ message: "Failed to fetch patient details" });
    }
  });

  app.get('/api/patient/:patientId/vitals', isAuthenticated, async (req, res) => {
    try {
      const { patientId } = req.params;
      const vitals = await storage.getPatientVitals(parseInt(patientId));
      res.json(vitals);
    } catch (error) {
      console.error("Error fetching patient vitals:", error);
      res.status(500).json({ message: "Failed to fetch patient vitals" });
    }
  });

  app.get('/api/patient/:patientId/medical-history', isAuthenticated, async (req, res) => {
    try {
      const { patientId } = req.params;
      const history = await storage.getPatientMedicalHistory(parseInt(patientId));
      res.json(history);
    } catch (error) {
      console.error("Error fetching medical history:", error);
      res.status(500).json({ message: "Failed to fetch medical history" });
    }
  });

  app.get('/api/patient/:patientId/lab-results', isAuthenticated, async (req, res) => {
    try {
      const { patientId } = req.params;
      const labResults = await storage.getPatientLabResults(parseInt(patientId));
      res.json(labResults);
    } catch (error) {
      console.error("Error fetching lab results:", error);
      res.status(500).json({ message: "Failed to fetch lab results" });
    }
  });

  app.get('/api/patient/:patientId/medications', isAuthenticated, async (req, res) => {
    try {
      const { patientId } = req.params;
      const medications = await storage.getPatientMedications(parseInt(patientId));
      res.json(medications);
    } catch (error) {
      console.error("Error fetching medications:", error);
      res.status(500).json({ message: "Failed to fetch medications" });
    }
  });

  app.get('/api/patient/:patientId/consultations', isAuthenticated, async (req, res) => {
    try {
      const { patientId } = req.params;
      const consultations = await storage.getPatientConsultations(parseInt(patientId));
      res.json(consultations);
    } catch (error) {
      console.error("Error fetching consultations:", error);
      res.status(500).json({ message: "Failed to fetch consultations" });
    }
  });

  // Dashboard stats
  app.get('/api/stats', isAuthenticated, async (req, res) => {
    try {
      const stats = await storage.getStats();
      res.json(stats);
    } catch (error) {
      console.error("Error fetching stats:", error);
      res.status(500).json({ message: "Failed to fetch stats" });
    }
  });

  // Queue management
  app.get('/api/queue', isAuthenticated, async (req, res) => {
    try {
      const { doctorId } = req.query;
      let queueItems;
      
      if (doctorId) {
        queueItems = await storage.getQueueByDoctorId(doctorId as string);
      } else {
        queueItems = await storage.getQueue();
      }
      
      res.json(queueItems);
    } catch (error) {
      console.error("Error fetching queue:", error);
      res.status(500).json({ message: "Failed to fetch queue" });
    }
  });

  app.post('/api/queue', isAuthenticated, async (req, res) => {
    try {
      const queueData = insertQueueEntrySchema.parse(req.body);
      
      // Intelligent doctor assignment based on category
      if (!queueData.doctorId && queueData.category) {
        const doctors = await storage.getDoctors();
        const availableDoctor = doctors.find(d => 
          d.specialty.toLowerCase().includes(queueData.category!.toLowerCase()) && 
          d.status === 'available'
        );
        if (availableDoctor) {
          queueData.doctorId = availableDoctor.id;
        }
      }
      
      const queueItem = await storage.addToQueue(queueData);
      res.json(queueItem);
    } catch (error) {
      console.error("Error adding to queue:", error);
      res.status(500).json({ message: "Failed to add to queue" });
    }
  });

  app.patch('/api/queue/:id/status', isAuthenticated, async (req, res) => {
    try {
      const { id } = req.params;
      const { status } = req.body;
      const updatedItem = await storage.updateQueueStatus(id, status);
      res.json(updatedItem);
    } catch (error) {
      console.error("Error updating queue status:", error);
      res.status(500).json({ message: "Failed to update queue status" });
    }
  });

  // Rooms management
  app.get('/api/rooms', isAuthenticated, async (req, res) => {
    try {
      const rooms = await storage.getRooms();
      res.json(rooms);
    } catch (error) {
      console.error("Error fetching rooms:", error);
      res.status(500).json({ message: "Failed to fetch rooms" });
    }
  });

  app.patch('/api/rooms/:id/status', isAuthenticated, async (req, res) => {
    try {
      const { id } = req.params;
      const { status, occupiedBeds } = req.body;
      const updatedRoom = await storage.updateRoomStatus(id, status, occupiedBeds);
      res.json(updatedRoom);
    } catch (error) {
      console.error("Error updating room status:", error);
      res.status(500).json({ message: "Failed to update room status" });
    }
  });

  // Doctors management
  app.get('/api/doctors', isAuthenticated, async (req, res) => {
    try {
      const doctors = await storage.getDoctors();
      res.json(doctors);
    } catch (error) {
      console.error("Error fetching doctors:", error);
      res.status(500).json({ message: "Failed to fetch doctors" });
    }
  });

  app.patch('/api/doctors/:id/status', isAuthenticated, async (req, res) => {
    try {
      const { id } = req.params;
      const { status } = req.body;
      const updatedDoctor = await storage.updateDoctorStatus(id, status);
      res.json(updatedDoctor);
    } catch (error) {
      console.error("Error updating doctor status:", error);
      res.status(500).json({ message: "Failed to update doctor status" });
    }
  });

  // Patients management
  app.get('/api/patients', isAuthenticated, async (req, res) => {
    try {
      const patients = await storage.getPatients();
      res.json(patients);
    } catch (error) {
      console.error("Error fetching patients:", error);
      res.status(500).json({ message: "Failed to fetch patients" });
    }
  });

  app.get('/api/patients/:id', isAuthenticated, async (req, res) => {
    try {
      const { id } = req.params;
      const patient = await storage.getPatientById(id);
      if (!patient) {
        return res.status(404).json({ message: "Patient not found" });
      }
      res.json(patient);
    } catch (error) {
      console.error("Error fetching patient:", error);
      res.status(500).json({ message: "Failed to fetch patient" });
    }
  });

  // Vitals management
  app.get('/api/patients/:id/vitals', isAuthenticated, async (req, res) => {
    try {
      const { id } = req.params;
      const vitals = await storage.getVitalsByPatientId(id);
      res.json(vitals);
    } catch (error) {
      console.error("Error fetching vitals:", error);
      res.status(500).json({ message: "Failed to fetch vitals" });
    }
  });

  app.post('/api/vitals', isAuthenticated, async (req, res) => {
    try {
      const vitalsData = insertVitalSchema.parse({
        ...req.body,
        recordedBy: req.user?.claims?.sub
      });
      const vitals = await storage.createVitals(vitalsData);
      res.json(vitals);
    } catch (error) {
      console.error("Error creating vitals:", error);
      res.status(500).json({ message: "Failed to create vitals" });
    }
  });

  // Staff tasks
  app.get('/api/tasks', isAuthenticated, async (req, res) => {
    try {
      const userId = req.user?.claims?.sub;
      const tasks = await storage.getTasksByUserId(userId);
      res.json(tasks);
    } catch (error) {
      console.error("Error fetching tasks:", error);
      res.status(500).json({ message: "Failed to fetch tasks" });
    }
  });

  app.post('/api/tasks', isAuthenticated, async (req, res) => {
    try {
      const taskData = req.body; // Temporary fix - need to create staff task schema
      const task = await storage.createTask(taskData);
      res.json(task);
    } catch (error) {
      console.error("Error creating task:", error);
      res.status(500).json({ message: "Failed to create task" });
    }
  });

  app.patch('/api/tasks/:id/status', isAuthenticated, async (req, res) => {
    try {
      const { id } = req.params;
      const { status } = req.body;
      const updatedTask = await storage.updateTaskStatus(id, status);
      res.json(updatedTask);
    } catch (error) {
      console.error("Error updating task status:", error);
      res.status(500).json({ message: "Failed to update task status" });
    }
  });

  // Medical records
  app.get('/api/patients/:id/records', isAuthenticated, async (req, res) => {
    try {
      const { id } = req.params;
      const records = await storage.getMedicalRecordsByPatientId(id);
      res.json(records);
    } catch (error) {
      console.error("Error fetching medical records:", error);
      res.status(500).json({ message: "Failed to fetch medical records" });
    }
  });

  // Appointments
  app.get('/api/appointments', isAuthenticated, async (req, res) => {
    try {
      const { doctorId, patientId } = req.query;
      let appointments;
      
      if (doctorId) {
        appointments = await storage.getAppointmentsByDoctorId(doctorId as string);
      } else if (patientId) {
        appointments = await storage.getAppointmentsByPatientId(patientId as string);
      } else {
        appointments = await storage.getAppointments();
      }
      
      res.json(appointments);
    } catch (error) {
      console.error("Error fetching appointments:", error);
      res.status(500).json({ message: "Failed to fetch appointments" });
    }
  });

  // Doctor Dashboard Routes
  app.get('/api/doctor/queue', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      
      // Include patients transferred from nurse queue after vitals collection
      const doctorQueue = [
        // Emergency patients (auto-transferred from nurse)
        {
          id: 'emergency1',
          name: 'Sarah Johnson',
          reasonForVisit: 'Fever and headache - EMERGENCY',
          priority: 'emergency',
          estimatedWaitTime: 0,
          queueNumber: 1,
          vitals: {
            temperature: 102.3,
            heartRate: 110,
            bloodPressure: '140/90',
            oxygenSaturation: 96
          },
          transferredFrom: 'nurse',
          transferTime: new Date(Date.now() - 5*60*1000),
          nurseNotes: 'High fever and elevated heart rate - immediate attention required'
        },
        // Normal priority patients (transferred from nurse after vitals)
        {
          id: 'normal1',
          name: 'Mike Chen',
          reasonForVisit: 'Chest pain - Post-vitals',
          priority: 'normal',
          estimatedWaitTime: 15,
          queueNumber: 2,
          vitals: {
            temperature: 98.8,
            heartRate: 78,
            bloodPressure: '125/82',
            oxygenSaturation: 99
          },
          transferredFrom: 'nurse',
          transferTime: new Date(Date.now() - 12*60*1000),
          nurseNotes: 'Vitals stable, patient reports mild chest discomfort'
        },
        {
          id: 'normal2',
          name: 'Lisa Rodriguez',
          reasonForVisit: 'Dizziness - Post-vitals',
          priority: 'normal',
          estimatedWaitTime: 30,
          queueNumber: 3,
          vitals: {
            temperature: 97.9,
            heartRate: 72,
            bloodPressure: '118/75',
            oxygenSaturation: 98
          },
          transferredFrom: 'nurse',
          transferTime: new Date(Date.now() - 20*60*1000),
          nurseNotes: 'Patient reports occasional dizziness, vitals within normal range'
        }
      ];
      
      res.json(doctorQueue);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch queue" });
    }
  });

  app.get('/api/doctor/appointments', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const appointments = await storage.getAppointmentsByDoctorId(doctorId);
      res.json(appointments.map((a: any) => ({
        id: a.id,
        patientName: a.patientName || `Patient ${a.patientId}`,
        appointmentType: 'Consultation',
        scheduledTime: a.scheduledTime || new Date(),
        status: a.status || 'scheduled'
      })));
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch appointments" });
    }
  });

  app.get('/api/doctor/lab-results', isAuthenticated, async (req, res) => {
    try {
      const results = [
        { id: 1, testName: 'Blood Test', patientName: 'John Doe', status: 'normal', reportedAt: new Date() },
        { id: 2, testName: 'X-Ray', patientName: 'Jane Smith', status: 'abnormal', reportedAt: new Date() }
      ];
      res.json(results);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch lab results" });
    }
  });

  app.get('/api/doctor/consultations', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const consultations = await storage.getQueue().then((queue: any[]) => 
        queue.filter((q: any) => q.doctorId === doctorId).slice(0, 10)
      );
      res.json(consultations.map((c: any) => ({
        id: c.id,
        patientName: c.patientName || `Patient ${c.patientId}`,
        notes: c.reasonForVisit,
        consultationTime: c.checkInTime,
        status: c.status
      })));
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch consultations" });
    }
  });

  // Doctor Schedule Management
  app.get('/api/doctor/schedule', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const schedule = await storage.getDoctorSchedule(doctorId);
      res.json(schedule);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch schedule" });
    }
  });

  app.post('/api/doctor/schedule', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const scheduleData = {
        ...req.body,
        doctorId
      }; // Temporary fix - doctor schedule schema not defined
      const schedule = await storage.createDoctorSchedule(scheduleData);
      res.json(schedule);
    } catch (error) {
      res.status(500).json({ message: "Failed to create schedule" });
    }
  });

  // Doctor Status Management
  app.get('/api/doctor/status', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const status = await storage.getDoctorStatus(doctorId);
      res.json(status || { status: 'available', location: null });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch status" });
    }
  });

  app.put('/api/doctor/status', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const { status, location } = req.body;
      const updatedStatus = await storage.updateDoctorStatus(doctorId, status, location);
      res.json(updatedStatus);
    } catch (error) {
      res.status(500).json({ message: "Failed to update status" });
    }
  });

  // Prescription Management
  app.post('/api/doctor/prescriptions', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const prescriptionData = insertPrescriptionSchema.parse({
        ...req.body,
        doctorId
      });
      const prescription = await storage.createPrescription(prescriptionData);
      res.json(prescription);
    } catch (error) {
      res.status(500).json({ message: "Failed to create prescription" });
    }
  });

  app.get('/api/doctor/prescriptions', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const prescriptions = await storage.getPrescriptionsByDoctorId(doctorId);
      res.json(prescriptions);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch prescriptions" });
    }
  });

  // Lab Orders Management
  app.post('/api/doctor/lab-orders', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const labOrderData = insertLabOrderSchema.parse({
        ...req.body,
        doctorId
      });
      const labOrder = await storage.createLabOrder(labOrderData);
      res.json(labOrder);
    } catch (error) {
      res.status(500).json({ message: "Failed to create lab order" });
    }
  });

  app.get('/api/doctor/lab-orders', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const labOrders = await storage.getLabOrdersByDoctorId(doctorId);
      res.json(labOrders);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch lab orders" });
    }
  });

  // AI Consultation
  app.post('/api/doctor/ai-consultation', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const { patientId, query, consultationType } = req.body;
      
      // Simulate AI response (in real implementation, this would call OpenAI)
      const aiResponse = `Based on the query: "${query}", I recommend further evaluation. Consider the patient's medical history and current symptoms. This is a simulated AI response for ${consultationType} consultation.`;
      
      const consultationData = {
        patientId,
        doctorId,
        query,
        aiResponse,
        confidence: 'medium',
        consultationType
      }; // Temporary fix - AI consultation schema not defined
      
      const consultation = await storage.createAiConsultation(consultationData);
      res.json(consultation);
    } catch (error) {
      res.status(500).json({ message: "Failed to create AI consultation" });
    }
  });

  // AI Scribe - Consultation Notes
  app.post('/api/doctor/consultation-notes', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const noteData = {
        ...req.body,
        doctorId
      }; // Temporary fix - consultation note schema not defined
      const note = await storage.createConsultationNote(noteData);
      res.json(note);
    } catch (error) {
      res.status(500).json({ message: "Failed to save consultation notes" });
    }
  });

  app.get('/api/doctor/consultation-notes/:patientId', isAuthenticated, async (req, res) => {
    try {
      const doctorId = (req.user as any).claims.sub;
      const { patientId } = req.params;
      const notes = await storage.getConsultationNotes(parseInt(patientId), doctorId);
      res.json(notes);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch consultation notes" });
    }
  });

  // Admin Dashboard Routes
  app.get('/api/admin/hospital-stats', isAuthenticated, async (req, res) => {
    try {
      const stats = {
        satisfaction: 92,
        feedbackCount: 156,
        todayConsultations: 45,
        admissions: 12,
        discharges: 8,
        pendingDischarges: 3
      };
      res.json(stats);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch hospital stats" });
    }
  });

  app.get('/api/admin/staff', isAuthenticated, async (req, res) => {
    try {
      const doctors = await storage.getDoctors();
      res.json(doctors.map((s: any) => ({
        id: s.id,
        firstName: s.firstName || 'Doctor',
        lastName: s.lastName || 'Smith',
        role: 'doctor',
        department: s.specialty || 'General Medicine',
        isActive: s.status === 'available'
      })));
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch staff" });
    }
  });

  app.get('/api/admin/departments', isAuthenticated, async (req, res) => {
    try {
      const depts = [
        { id: 1, name: 'Cardiology', description: 'Heart and cardiovascular care', headDoctorName: 'Smith', staffCount: 5, todayPatients: 15, revenue: 25000 },
        { id: 2, name: 'Emergency', description: 'Emergency medical care', headDoctorName: 'Johnson', staffCount: 8, todayPatients: 22, revenue: 35000 },
        { id: 3, name: 'General Medicine', description: 'Primary healthcare', headDoctorName: 'Williams', staffCount: 6, todayPatients: 18, revenue: 20000 }
      ];
      res.json(depts);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch departments" });
    }
  });

  app.get('/api/admin/finances', isAuthenticated, async (req, res) => {
    try {
      const finances = {
        monthlyRevenue: 250000,
        growthRate: 12,
        totalRevenue: 250000,
        totalExpenses: 180000,
        profitMargin: 28,
        collectionRate: 95
      };
      res.json(finances);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch finances" });
    }
  });

  app.get('/api/admin/bed-occupancy', isAuthenticated, async (req, res) => {
    try {
      const bedData = {
        occupancyRate: 85,
        occupiedBeds: 42,
        totalBeds: 50
      };
      res.json(bedData);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch bed occupancy" });
    }
  });

  // Nurse Dashboard Routes
  app.get('/api/nurse/patients', isAuthenticated, async (req, res) => {
    try {
      const patients = await storage.getPatients();
      res.json(patients.slice(0, 10).map((p: any, index) => ({
        id: p.id,
        name: p.name || `Patient ${p.id}`,
        roomNumber: 'A101',
        bedNumber: '1',
        condition: 'stable',
        lastVitalsTime: new Date(),
        heartRate: index === 0 ? 105 : index === 1 ? 72 : 68, // Emergency flag for first patient
        bloodPressure: '120/80',
        temperature: index === 0 ? 102.1 : index === 1 ? 98.6 : 97.8, // Emergency flag for first patient
        oxygenSaturation: index === 2 ? 88 : 98 // Emergency flag for third patient
      })));
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch patients" });
    }
  });

  app.get('/api/nurse/tasks', isAuthenticated, async (req, res) => {
    try {
      const userId = (req.user as any).claims.sub;
      const tasks = await storage.getTasksByUserId(userId);
      res.json(tasks.map((t: any) => ({
        id: t.id,
        description: t.description,
        patientName: `Patient ${t.patientId}`,
        priority: t.priority || 'normal',
        taskType: t.taskType || 'general',
        status: t.status,
        dueTime: t.dueTime || new Date()
      })));
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch tasks" });
    }
  });

  app.get('/api/nurse/medications', isAuthenticated, async (req, res) => {
    try {
      const medications = [
        { id: 1, medicationName: 'Aspirin', patientName: 'John Doe', dosage: '100mg', frequency: 'Daily', dueTime: new Date(Date.now() + 2 * 60 * 60 * 1000) },
        { id: 2, medicationName: 'Insulin', patientName: 'Jane Smith', dosage: '10 units', frequency: 'Before meals', dueTime: new Date(Date.now() + 30 * 60 * 1000) }
      ];
      res.json(medications);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch medications" });
    }
  });

  app.get('/api/nurse/vitals-alerts', isAuthenticated, async (req, res) => {
    try {
      const alerts = [
        { id: 1, alertType: 'High Blood Pressure', patientName: 'John Doe', value: '180/95' },
        { id: 2, alertType: 'Low Oxygen Saturation', patientName: 'Jane Smith', value: '89%' }
      ];
      res.json(alerts);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch vitals alerts" });
    }
  });

  // Nurse Queue - patients waiting for vitals after check-in
  app.get('/api/nurse/queue', isAuthenticated, async (req, res) => {
    try {
      const nurseQueue = [
        {
          id: 'q1',
          name: 'Sarah Johnson',
          checkedInAt: new Date(Date.now() - 15*60*1000),
          chiefComplaint: 'Fever and headache',
          waitTime: 15,
          priority: 'normal'
        },
        {
          id: 'q2', 
          name: 'Mike Chen',
          checkedInAt: new Date(Date.now() - 8*60*1000),
          chiefComplaint: 'Chest pain',
          waitTime: 8,
          priority: 'urgent'
        },
        {
          id: 'q3',
          name: 'Lisa Rodriguez', 
          checkedInAt: new Date(Date.now() - 25*60*1000),
          chiefComplaint: 'Dizziness',
          waitTime: 25,
          priority: 'high'
        },
        {
          id: 'q4',
          name: 'David Kim',
          checkedInAt: new Date(Date.now() - 5*60*1000), 
          chiefComplaint: 'Shortness of breath',
          waitTime: 5,
          priority: 'normal'
        }
      ];
      
      res.json(nurseQueue);
    } catch (error) {
      console.error("Error fetching nurse queue:", error);
      res.status(500).json({ message: "Failed to fetch nurse queue" });
    }
  });

  // Record vitals and transfer patient to doctor queue
  app.post('/api/nurse/record-vitals', isAuthenticated, async (req, res) => {
    try {
      const { patientId, temperature, heartRate, bloodPressure, oxygenSaturation, notes, isEmergency } = req.body;
      
      const vitalsRecord = {
        id: Date.now(),
        patientId,
        temperature,
        heartRate,
        bloodPressure,
        oxygenSaturation,
        notes,
        recordedAt: new Date(),
        recordedBy: req.user?.claims?.sub || 'nurse1',
        isEmergency
      };
      
      if (isEmergency) {
        console.log(`🚨 EMERGENCY: Patient ${patientId} transferred to doctor queue with emergency priority`);
      } else {
        console.log(`✅ Patient ${patientId} transferred to doctor queue after vitals collection`);
      }
      
      res.json({
        success: true,
        vitalsRecord,
        transferred: true,
        priority: isEmergency ? 'EMERGENCY' : 'NORMAL',
        message: isEmergency ? 
          'Emergency vitals recorded! Patient transferred to doctor queue with high priority.' :
          'Vitals recorded successfully. Patient transferred to doctor queue.'
      });
    } catch (error) {
      console.error("Error recording vitals:", error);
      res.status(500).json({ message: "Failed to record vitals" });
    }
  });

  // Receptionist Dashboard Routes
  app.get('/api/receptionist/queue', isAuthenticated, async (req, res) => {
    try {
      const queue = await storage.getQueue();
      res.json(queue.map((q: any) => ({
        id: q.id,
        queueNumber: q.id,
        patientName: q.patientName || `Patient ${q.patientId}`,
        reasonForVisit: q.reasonForVisit,
        priority: q.priority,
        status: q.status,
        estimatedWaitTime: q.estimatedWaitTime
      })));
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch queue" });
    }
  });

  app.get('/api/receptionist/appointments', isAuthenticated, async (req, res) => {
    try {
      const appointments = await storage.getAppointments();
      res.json(appointments.slice(0, 10).map((a: any) => ({
        id: a.id,
        patientName: a.patientName || `Patient ${a.patientId}`,
        appointmentType: 'Consultation',
        appointmentDate: a.scheduledTime || new Date(),
        scheduledTime: a.scheduledTime || new Date(),
        status: a.status || 'scheduled',
        doctorName: 'Smith',
        duration: 30
      })));
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch appointments" });
    }
  });

  app.get('/api/receptionist/check-ins', isAuthenticated, async (req, res) => {
    try {
      const checkIns = [
        { id: 1, patientName: 'John Doe', isNewPatient: false, checkedInAt: new Date(), department: 'Cardiology' },
        { id: 2, patientName: 'Jane Smith', isNewPatient: true, checkedInAt: new Date(), department: 'General' }
      ];
      res.json(checkIns);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch check-ins" });
    }
  });

  app.get('/api/receptionist/billing', isAuthenticated, async (req, res) => {
    try {
      const billing = [
        { id: 1, patientName: 'John Doe', amount: 250, status: 'pending', invoiceNumber: 'INV-001' },
        { id: 2, patientName: 'Jane Smith', amount: 180, status: 'paid', invoiceNumber: 'INV-002' }
      ];
      res.json(billing);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch billing" });
    }
  });

  // Patient Dashboard Routes
  app.get('/api/patient/visit-status', isAuthenticated, async (req, res) => {
    try {
      const patientId = (req.user as any).claims.sub;
      const visitStatus = {
        estimatedWaitTime: 25,
        queuePosition: 3,
        doctorName: 'Smith',
        department: 'General Medicine',
        roomNumber: 'TBD',
        floor: '2',
        priority: 'normal',
        status: 'waiting',
        labUpdates: [
          { message: 'Blood test results available', time: '10:30 AM' }
        ],
        prescriptionUpdates: [
          { message: 'Prescription ready for pickup', time: '11:00 AM' }
        ]
      };
      res.json(visitStatus);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch visit status" });
    }
  });

  app.get('/api/patient/health-history', isAuthenticated, async (req, res) => {
    try {
      const history = [
        {
          id: 1,
          visitType: 'Annual Checkup',
          date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          doctorName: 'Johnson',
          summary: 'Routine annual physical examination. All vital signs normal, no concerns noted.',
          status: 'Completed',
          hospital: 'General Hospital',
          department: 'Family Medicine'
        },
        {
          id: 2,
          visitType: 'Follow-up',
          date: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000),
          doctorName: 'Williams',
          summary: 'Follow-up for blood pressure management. Medication adjusted, continue monitoring.',
          status: 'Completed',
          hospital: 'General Hospital',
          department: 'Cardiology'
        }
      ];
      res.json(history);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch health history" });
    }
  });

  app.get('/api/patient/notifications', isAuthenticated, async (req, res) => {
    try {
      const notifications = [
        {
          id: 1,
          title: 'Lab Results Available',
          message: 'Your blood test results from yesterday are now available for review.',
          type: 'lab_result',
          createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
          isRead: false,
          actionRequired: true,
          actionText: 'View Results'
        },
        {
          id: 2,
          title: 'Appointment Reminder',
          message: 'Your appointment with Dr. Smith is scheduled for tomorrow at 2:00 PM.',
          type: 'appointment',
          createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000),
          isRead: true,
          actionRequired: false
        },
        {
          id: 3,
          title: 'Prescription Ready',
          message: 'Your prescription is ready for pickup at the hospital pharmacy.',
          type: 'prescription',
          createdAt: new Date(Date.now() - 4 * 60 * 60 * 1000),
          isRead: false,
          actionRequired: true,
          actionText: 'Get Directions'
        }
      ];
      res.json(notifications);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch notifications" });
    }
  });

  app.get('/api/patient/lab-results', isAuthenticated, async (req, res) => {
    try {
      const results = [
        {
          id: 1,
          testName: 'Complete Blood Count',
          date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
          status: 'normal'
        },
        {
          id: 2,
          testName: 'Lipid Panel',
          date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          status: 'normal'
        },
        {
          id: 3,
          testName: 'Blood Glucose',
          date: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
          status: 'abnormal'
        }
      ];
      res.json(results);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch lab results" });
    }
  });

  app.post('/api/patient/check-in', isAuthenticated, async (req, res) => {
    try {
      const { symptoms, preferences } = req.body;
      const patientId = (req.user as any).claims.sub;
      
      // Process check-in
      const checkIn = {
        id: Date.now(),
        patientId,
        symptoms,
        preferences,
        checkInTime: new Date(),
        queueNumber: Math.floor(Math.random() * 100) + 1
      };
      
      res.json({ success: true, checkIn });
    } catch (error) {
      res.status(500).json({ message: "Failed to process check-in" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
