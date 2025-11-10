import {
  users,
  patients,
  rooms,
  vitals,
  appointments,
  medicalRecords,
  prescriptions,
  labOrders,
  labResults,
  labTests,
  consultations,
  queueEntries,
  type User,
  type UpsertUser,
  type Patient,
  type InsertPatient,
  type Room,
  type QueueEntry,
  type InsertQueueEntry,
  type Vital,
  type InsertVital,
  type Appointment,
  type InsertAppointment,
  type Prescription,
  type LabOrder,
  type Consultation,
  type InsertConsultation,
} from "@shared/healthcare-schema";
import { db } from "./db";
import { eq, desc, asc, and, or, sql, count } from "drizzle-orm";

export interface IStorage {
  // User operations (mandatory for Replit Auth)
  getUser(id: string): Promise<User | undefined>;
  upsertUser(user: UpsertUser): Promise<User>;
  updateUserRole(id: string, role: string): Promise<User>;
  
  // Patient operations
  getPatients(): Promise<Patient[]>;
  getPatientById(patientId: number): Promise<Patient | undefined>;
  createPatient(patient: InsertPatient): Promise<Patient>;
  
  // Room operations
  getRooms(): Promise<Room[]>;
  
  // Queue operations
  getQueue(): Promise<QueueEntry[]>;
  
  // Patient details operations
  getPatientDetails(patientId: number): Promise<any>;
  getPatientVitals(patientId: number): Promise<any[]>;
  getPatientMedicalHistory(patientId: number): Promise<any[]>;
  getPatientLabResults(patientId: number): Promise<any[]>;
  getPatientMedications(patientId: number): Promise<any[]>;
  getPatientConsultations(patientId: number): Promise<any[]>;
  
  // Appointment operations
  getAppointments(): Promise<Appointment[]>;
  
  // Stats operations
  getStats(): Promise<any>;
  addToQueue(queueItem: InsertQueue): Promise<QueueItem>;
  updateQueueStatus(queueId: string, status: string): Promise<QueueItem>;
  assignDoctorToQueue(queueId: string, doctorId: string): Promise<QueueItem>;
  
  // Vitals operations
  getVitalsByPatientId(patientId: string): Promise<Vitals[]>;
  createVitals(vitals: InsertVitals): Promise<Vitals>;
  
  // Appointment operations
  getAppointments(): Promise<Appointment[]>;
  getAppointmentsByDoctorId(doctorId: string): Promise<Appointment[]>;
  getAppointmentsByPatientId(patientId: string): Promise<Appointment[]>;
  createAppointment(appointment: InsertAppointment): Promise<Appointment>;
  
  // Staff task operations
  getTasksByUserId(userId: string): Promise<StaffTask[]>;
  createTask(task: InsertStaffTask): Promise<StaffTask>;
  updateTaskStatus(taskId: string, status: string): Promise<StaffTask>;
  
  // Medical record operations
  getMedicalRecordsByPatientId(patientId: string): Promise<MedicalRecord[]>;
  createMedicalRecord(record: InsertMedicalRecord): Promise<MedicalRecord>;
  
  // Statistics
  getStats(): Promise<{
    currentPatients: number;
    waitingQueue: number;
    availableRooms: number;
    totalRooms: number;
    activeStaff: number;
    totalStaff: number;
  }>;

  // Doctor-specific operations
  getDoctorSchedule(doctorId: string): Promise<DoctorSchedule[]>;
  createDoctorSchedule(schedule: InsertDoctorSchedule): Promise<DoctorSchedule>;
  getDoctorStatus(doctorId: string): Promise<{ status: string; location?: string } | undefined>;
  updateDoctorStatus(doctorId: string, status: string, location?: string): Promise<{ status: string; location?: string }>;
  
  // Prescription operations
  createPrescription(prescription: InsertPrescription): Promise<Prescription>;
  getPrescriptionsByDoctorId(doctorId: string): Promise<Prescription[]>;
  
  // Lab order operations
  createLabOrder(labOrder: InsertLabOrder): Promise<LabOrder>;
  getLabOrdersByDoctorId(doctorId: string): Promise<LabOrder[]>;
  
  // AI consultation operations
  createAiConsultation(consultation: InsertAiConsultation): Promise<AiConsultation>;
  
  // Consultation notes operations
  createConsultationNote(note: InsertConsultationNote): Promise<ConsultationNote>;
  getConsultationNotes(patientId: number, doctorId: string): Promise<ConsultationNote[]>;
}

export class DatabaseStorage implements IStorage {
  // User operations
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async upsertUser(userData: UpsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(userData)
      .onConflictDoUpdate({
        target: users.id,
        set: {
          ...userData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return user;
  }

  async updateUserRole(id: string, role: string): Promise<User> {
    const [user] = await db
      .update(users)
      .set({ role: role as any, updatedAt: new Date() })
      .where(eq(users.id, id))
      .returning();
    return user;
  }

  // Patient operations
  async getPatients(): Promise<Patient[]> {
    return await db.select().from(patients);
  }

  async getPatientById(patientId: number): Promise<Patient | undefined> {
    const [patient] = await db.select().from(patients).where(eq(patients.id, patientId));
    return patient;
  }

  async createPatient(patient: InsertPatient): Promise<Patient> {
    const [newPatient] = await db.insert(patients).values(patient).returning();
    return newPatient;
  }

  // Room operations
  async getRooms(): Promise<Room[]> {
    return await db.select().from(rooms);
  }

  // Queue operations
  async getQueue(): Promise<QueueEntry[]> {
    return await db.select().from(queueEntries).orderBy(asc(queueEntries.queueNumber));
  }

  // Appointment operations
  async getAppointments(): Promise<Appointment[]> {
    return await db.select().from(appointments).orderBy(asc(appointments.appointmentDate));
  }

  // Stats operations
  async getStats(): Promise<any> {
    const totalPatients = await db.select({ count: count() }).from(patients);
    const totalAppointments = await db.select({ count: count() }).from(appointments);
    const totalQueue = await db.select({ count: count() }).from(queueEntries);
    
    return {
      totalPatients: totalPatients[0]?.count || 0,
      totalAppointments: totalAppointments[0]?.count || 0,
      queueLength: totalQueue[0]?.count || 0,
    };
  }

  // Doctor operations
  async getDoctors(): Promise<Doctor[]> {
    return await db.select().from(doctors).orderBy(asc(doctors.createdAt));
  }

  async getDoctorByUserId(userId: string): Promise<Doctor | undefined> {
    const [doctor] = await db.select().from(doctors).where(eq(doctors.userId, userId));
    return doctor;
  }

  async createDoctor(doctor: InsertDoctor): Promise<Doctor> {
    const [newDoctor] = await db.insert(doctors).values(doctor).returning();
    return newDoctor;
  }

  async updateDoctorStatus(doctorId: string, status: string): Promise<Doctor> {
    const [updatedDoctor] = await db
      .update(doctors)
      .set({ status: status as any, updatedAt: new Date() })
      .where(eq(doctors.id, doctorId))
      .returning();
    return updatedDoctor;
  }

  // Patient operations
  async getPatients(): Promise<Patient[]> {
    return await db.select().from(patients).orderBy(desc(patients.createdAt));
  }

  async getPatientById(patientId: string): Promise<Patient | undefined> {
    const [patient] = await db.select().from(patients).where(eq(patients.id, patientId));
    return patient;
  }

  async getPatientByUserId(userId: string): Promise<Patient | undefined> {
    const [patient] = await db.select().from(patients).where(eq(patients.userId, userId));
    return patient;
  }

  async getPatientByPatientId(patientId: string): Promise<Patient | undefined> {
    const [patient] = await db.select().from(patients).where(eq(patients.patientId, patientId));
    return patient;
  }

  async createPatient(patient: InsertPatient): Promise<Patient> {
    const [newPatient] = await db.insert(patients).values(patient).returning();
    return newPatient;
  }

  // Room operations
  async getRooms(): Promise<Room[]> {
    return await db.select().from(rooms).orderBy(asc(rooms.roomNumber));
  }

  async getRoomById(roomId: string): Promise<Room | undefined> {
    const [room] = await db.select().from(rooms).where(eq(rooms.id, roomId));
    return room;
  }

  async createRoom(room: InsertRoom): Promise<Room> {
    const [newRoom] = await db.insert(rooms).values(room).returning();
    return newRoom;
  }

  async updateRoomStatus(roomId: string, status: string, occupiedBeds?: number): Promise<Room> {
    const updateData: any = { status: status as any, updatedAt: new Date() };
    if (occupiedBeds !== undefined) {
      updateData.occupiedBeds = occupiedBeds;
    }
    const [updatedRoom] = await db
      .update(rooms)
      .set(updateData)
      .where(eq(rooms.id, roomId))
      .returning();
    return updatedRoom;
  }

  // Queue operations
  async getQueue(): Promise<QueueItem[]> {
    return await db.select().from(queue).where(eq(queue.status, "waiting")).orderBy(asc(queue.queuePosition));
  }

  async getQueueByDoctorId(doctorId: string): Promise<QueueItem[]> {
    return await db
      .select()
      .from(queue)
      .where(and(eq(queue.doctorId, doctorId), eq(queue.status, "waiting")))
      .orderBy(asc(queue.queuePosition));
  }

  async getQueueByPatientId(patientId: string): Promise<QueueItem[]> {
    return await db
      .select()
      .from(queue)
      .where(eq(queue.patientId, patientId))
      .orderBy(desc(queue.createdAt));
  }

  async addToQueue(queueItem: InsertQueue): Promise<QueueItem> {
    // Get the next queue position
    const [maxPosition] = await db
      .select({ max: sql<number>`COALESCE(MAX(${queue.queuePosition}), 0)` })
      .from(queue)
      .where(eq(queue.status, "waiting"));
    
    const nextPosition = (maxPosition.max || 0) + 1;
    
    const [newQueueItem] = await db
      .insert(queue)
      .values({ ...queueItem, queuePosition: nextPosition })
      .returning();
    return newQueueItem;
  }

  async updateQueueStatus(queueId: string, status: string): Promise<QueueItem> {
    const [updatedQueueItem] = await db
      .update(queue)
      .set({ status: status as any, updatedAt: new Date() })
      .where(eq(queue.id, queueId))
      .returning();
    return updatedQueueItem;
  }

  async assignDoctorToQueue(queueId: string, doctorId: string): Promise<QueueItem> {
    const [updatedQueueItem] = await db
      .update(queue)
      .set({ doctorId, updatedAt: new Date() })
      .where(eq(queue.id, queueId))
      .returning();
    return updatedQueueItem;
  }

  // Vitals operations
  async getVitalsByPatientId(patientId: string): Promise<Vitals[]> {
    return await db
      .select()
      .from(vitals)
      .where(eq(vitals.patientId, patientId))
      .orderBy(desc(vitals.recordedAt));
  }

  async createVitals(vitalsData: InsertVitals): Promise<Vitals> {
    const [newVitals] = await db.insert(vitals).values(vitalsData).returning();
    return newVitals;
  }

  // Appointment operations
  async getAppointments(): Promise<Appointment[]> {
    return await db.select().from(appointments).orderBy(asc(appointments.scheduledTime));
  }

  async getAppointmentsByDoctorId(doctorId: string): Promise<Appointment[]> {
    return await db
      .select()
      .from(appointments)
      .where(eq(appointments.doctorId, doctorId))
      .orderBy(asc(appointments.scheduledTime));
  }

  async getAppointmentsByPatientId(patientId: string): Promise<Appointment[]> {
    return await db
      .select()
      .from(appointments)
      .where(eq(appointments.patientId, patientId))
      .orderBy(asc(appointments.scheduledTime));
  }

  async createAppointment(appointment: InsertAppointment): Promise<Appointment> {
    const [newAppointment] = await db.insert(appointments).values(appointment).returning();
    return newAppointment;
  }

  // Staff task operations
  async getTasksByUserId(userId: string): Promise<StaffTask[]> {
    return await db
      .select()
      .from(staffTasks)
      .where(eq(staffTasks.assignedTo, userId))
      .orderBy(asc(staffTasks.dueTime));
  }

  async createTask(task: InsertStaffTask): Promise<StaffTask> {
    const [newTask] = await db.insert(staffTasks).values(task).returning();
    return newTask;
  }

  async updateTaskStatus(taskId: string, status: string): Promise<StaffTask> {
    const updateData: any = { status, updatedAt: new Date() };
    if (status === "completed") {
      updateData.completedAt = new Date();
    }
    const [updatedTask] = await db
      .update(staffTasks)
      .set(updateData)
      .where(eq(staffTasks.id, taskId))
      .returning();
    return updatedTask;
  }

  // Medical record operations
  async getMedicalRecordsByPatientId(patientId: string): Promise<MedicalRecord[]> {
    return await db
      .select()
      .from(medicalRecords)
      .where(eq(medicalRecords.patientId, patientId))
      .orderBy(desc(medicalRecords.visitDate));
  }

  async createMedicalRecord(record: InsertMedicalRecord): Promise<MedicalRecord> {
    const [newRecord] = await db.insert(medicalRecords).values(record).returning();
    return newRecord;
  }

  // Statistics
  async getStats(): Promise<{
    currentPatients: number;
    waitingQueue: number;
    availableRooms: number;
    totalRooms: number;
    activeStaff: number;
    totalStaff: number;
  }> {
    const [queueCount] = await db
      .select({ count: count() })
      .from(queue)
      .where(eq(queue.status, "waiting"));

    const [roomStats] = await db
      .select({
        total: count(),
        available: sql<number>`COUNT(CASE WHEN ${rooms.status} = 'available' THEN 1 END)`,
      })
      .from(rooms);

    const [staffStats] = await db
      .select({
        total: count(),
        active: sql<number>`COUNT(CASE WHEN ${users.role} IN ('doctor', 'nurse', 'receptionist') THEN 1 END)`,
      })
      .from(users);

    const [patientCount] = await db
      .select({ count: count() })
      .from(queue)
      .where(or(eq(queue.status, "waiting"), eq(queue.status, "in_progress")));

    return {
      currentPatients: patientCount.count,
      waitingQueue: queueCount.count,
      availableRooms: roomStats.available || 0,
      totalRooms: roomStats.total,
      activeStaff: staffStats.active || 0,
      totalStaff: staffStats.total,
    };
  }

  // Doctor-specific operations
  async getDoctorSchedule(doctorId: string): Promise<DoctorSchedule[]> {
    const schedules = await db.select().from(doctorSchedule).where(eq(doctorSchedule.doctorId, doctorId));
    return schedules;
  }

  async createDoctorSchedule(schedule: InsertDoctorSchedule): Promise<DoctorSchedule> {
    const [newSchedule] = await db.insert(doctorSchedule).values(schedule).returning();
    return newSchedule;
  }

  async getDoctorStatus(doctorId: string): Promise<{ status: string; location?: string } | undefined> {
    const [doctor] = await db.select().from(doctors).where(eq(doctors.userId, doctorId));
    if (!doctor) return undefined;
    return { status: doctor.status || 'available', location: doctor.roomNumber || undefined };
  }

  async updateDoctorStatus(doctorId: string, status: string, location?: string): Promise<{ status: string; location?: string }> {
    const [doctor] = await db
      .update(doctors)
      .set({ 
        status: status as any,
        roomNumber: location,
        updatedAt: new Date() 
      })
      .where(eq(doctors.userId, doctorId))
      .returning();
    
    return { status: doctor.status || status, location: doctor.roomNumber || location };
  }

  // Prescription operations
  async createPrescription(prescription: InsertPrescription): Promise<Prescription> {
    const [newPrescription] = await db.insert(prescriptions).values(prescription).returning();
    return newPrescription;
  }

  async getPrescriptionsByDoctorId(doctorId: string): Promise<Prescription[]> {
    const doctorPrescriptions = await db.select().from(prescriptions).where(eq(prescriptions.doctorId, doctorId));
    return doctorPrescriptions;
  }

  // Lab order operations
  async createLabOrder(labOrder: InsertLabOrder): Promise<LabOrder> {
    const [newLabOrder] = await db.insert(labOrders).values(labOrder).returning();
    return newLabOrder;
  }

  async getLabOrdersByDoctorId(doctorId: string): Promise<LabOrder[]> {
    // Using existing labOrders table which has different structure
    const doctorLabOrders = await db.select().from(labOrders).where(eq(labOrders.doctorId, doctorId));
    return doctorLabOrders;
  }

  // AI consultation operations
  async createAiConsultation(consultation: InsertAiConsultation): Promise<AiConsultation> {
    const [newConsultation] = await db.insert(aiConsultations).values(consultation).returning();
    return newConsultation;
  }

  // Consultation notes operations
  async createConsultationNote(note: InsertConsultationNote): Promise<ConsultationNote> {
    const [newNote] = await db.insert(consultationNotes).values(note).returning();
    return newNote;
  }

  async getConsultationNotes(patientId: number, doctorId: string): Promise<ConsultationNote[]> {
    const notes = await db
      .select()
      .from(consultationNotes)
      .where(and(
        eq(consultationNotes.patientId, patientId),
        eq(consultationNotes.doctorId, doctorId)
      ))
      .orderBy(desc(consultationNotes.consultationDate));
    return notes;
  }

  // Patient details operations
  async getPatientDetails(patientId: number): Promise<any> {
    const [patient] = await db.select().from(patients).where(eq(patients.id, patientId));
    return patient;
  }

  async getPatientVitals(patientId: number): Promise<any[]> {
    const patientVitals = await db
      .select()
      .from(vitals)
      .where(eq(vitals.patientId, patientId.toString()))
      .orderBy(desc(vitals.recordedAt))
      .limit(5);
    return patientVitals;
  }

  async getPatientMedicalHistory(patientId: number): Promise<any[]> {
    // Get medical records and diagnoses
    const records = await db
      .select()
      .from(medicalRecords)
      .where(eq(medicalRecords.patientId, patientId))
      .orderBy(desc(medicalRecords.createdAt));
    
    return records;
  }

  async getPatientLabResults(patientId: number): Promise<any[]> {
    // Join lab results with lab tests to get test names
    const results = await db
      .select({
        id: labResults.id,
        testName: labTests.testName,
        resultValue: labResults.resultValue,
        referenceRange: labResults.referenceRange,
        unit: labResults.unit,
        status: labResults.status,
        reportedAt: labResults.reportedAt
      })
      .from(labResults)
      .innerJoin(labTests, eq(labResults.labTestId, labTests.id))
      .where(eq(labResults.patientId, patientId))
      .orderBy(desc(labResults.reportedAt))
      .limit(10);
    
    return results;
  }

  async getPatientMedications(patientId: number): Promise<any[]> {
    const medications = await db
      .select()
      .from(prescriptions)
      .where(and(
        eq(prescriptions.patientId, patientId),
        eq(prescriptions.isActive, true)
      ))
      .orderBy(desc(prescriptions.prescribedAt));
    
    return medications;
  }

  async getPatientConsultations(patientId: number): Promise<any[]> {
    // Get consultations with doctor names
    const consultationHistory = await db
      .select({
        id: consultations.id,
        consultationTime: consultations.consultationTime,
        notes: consultations.notes,
        status: consultations.status,
        doctorFirstName: users.firstName,
        doctorLastName: users.lastName
      })
      .from(consultations)
      .innerJoin(users, eq(consultations.doctorId, users.id))
      .where(eq(consultations.patientId, patientId))
      .orderBy(desc(consultations.consultationTime))
      .limit(10);
    
    return consultationHistory;
  }
}

export const storage = new DatabaseStorage();
