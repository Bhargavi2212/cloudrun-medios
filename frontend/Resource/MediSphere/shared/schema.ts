import { sql, relations } from "drizzle-orm";
import {
  index,
  jsonb,
  pgTable,
  timestamp,
  varchar,
  text,
  integer,
  boolean,
  pgEnum,
  serial,
  numeric,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Session storage table for Replit Auth
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// User roles enum
export const userRoleEnum = pgEnum("user_role", [
  "admin",
  "doctor",
  "nurse", 
  "receptionist",
  "patient"
]);

// Room status enum
export const roomStatusEnum = pgEnum("room_status", [
  "available",
  "occupied", 
  "cleaning",
  "maintenance",
  "reserved"
]);

// Room type enum
export const roomTypeEnum = pgEnum("room_type", [
  "icu",
  "general",
  "private", 
  "emergency",
  "operation_theater",
  "consultation"
]);

// Doctor status enum
export const doctorStatusEnum = pgEnum("doctor_status", [
  "available",
  "busy",
  "on_break",
  "emergency",
  "off_duty"
]);

// Queue status enum
export const queueStatusEnum = pgEnum("queue_status", [
  "waiting",
  "in_progress",
  "completed",
  "cancelled"
]);

// General status enum
export const statusEnum = pgEnum("status", [
  "scheduled",
  "in_progress", 
  "completed",
  "cancelled",
  "waiting",
  "called",
  "pending",
  "approved",
  "rejected"
]);

// Priority enum
export const priorityEnum = pgEnum("priority", [
  "low",
  "normal", 
  "high",
  "urgent",
  "emergency"
]);

// Core Healthcare Entity Tables
export const hospitals = pgTable("hospitals", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 255 }).notNull(),
  address: varchar("address", { length: 500 }).notNull(),
  branchCode: varchar("branch_code", { length: 50 }).notNull(),
  contactInfo: varchar("contact_info", { length: 255 }).notNull(),
  location: varchar("location", { length: 255 }),
  phone: varchar("phone", { length: 20 }),
  hipaaCompliant: boolean("hipaa_compliant").default(true),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const departments = pgTable("departments", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 255 }).notNull(),
  description: varchar("description", { length: 500 }),
  headDoctorId: varchar("head_doctor_id"),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Users table for Replit Auth - Extended for Healthcare
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  role: userRoleEnum("role"),
  departmentId: integer("department_id").references(() => departments.id),
  phone: varchar("phone", { length: 20 }),
  salary: numeric("salary", { precision: 10, scale: 2 }),
  shift: varchar("shift", { length: 50 }),
  isActive: boolean("is_active").default(true).notNull(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Doctors table
export const doctors = pgTable("doctors", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  specialty: varchar("specialty").notNull(),
  licenseNumber: varchar("license_number").unique(),
  status: doctorStatusEnum("status").default("available"),
  roomNumber: varchar("room_number"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Patients table - Extended for Healthcare System
export const patients = pgTable("patients", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }).unique(),
  patientId: varchar("patient_id").unique(),
  name: varchar("name", { length: 255 }),
  age: integer("age"),
  phone: varchar("phone", { length: 20 }),
  abhaId: varchar("abha_id", { length: 50 }),
  mrn: varchar("mrn", { length: 50 }),
  dateOfBirth: timestamp("date_of_birth"),
  gender: varchar("gender"),
  bloodType: varchar("blood_type"),
  height: numeric("height", { precision: 5, scale: 2 }),
  weight: numeric("weight", { precision: 5, scale: 2 }),
  allergies: text("allergies"),
  emergencyContactName: varchar("emergency_contact_name", { length: 255 }),
  emergencyContactPhone: varchar("emergency_contact_phone", { length: 20 }),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  deletedAt: timestamp("deleted_at"),
});

// Facility Management Tables
export const rooms = pgTable("rooms", {
  id: serial("id").primaryKey(),
  roomNumber: varchar("room_number", { length: 50 }).notNull(),
  roomType: roomTypeEnum("room_type").notNull(),
  capacity: integer("capacity").default(0).notNull(),
  totalBeds: integer("total_beds").default(1),
  occupiedBeds: integer("occupied_beds").default(0),
  isAvailable: boolean("is_available").default(true).notNull(),
  status: roomStatusEnum("status").default("available"),
  floor: integer("floor"),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  departmentId: integer("department_id").references(() => departments.id),
  equipment: text("equipment").array(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const beds = pgTable("beds", {
  id: serial("id").primaryKey(),
  bedNumber: varchar("bed_number", { length: 50 }).notNull(),
  isOccupied: boolean("is_occupied").default(false).notNull(),
  patientId: integer("patient_id").references(() => patients.id),
  roomId: integer("room_id").notNull().references(() => rooms.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const wards = pgTable("wards", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 255 }).notNull(),
  capacity: integer("capacity").notNull(),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  deletedAt: timestamp("deleted_at"),
});

// Clinical Tables
export const consultations = pgTable("consultations", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  consultationTime: timestamp("consultation_time").notNull(),
  notes: text("notes"),
  status: statusEnum("status").default("scheduled").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const prescriptions = pgTable("prescriptions", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  consultationId: integer("consultation_id").references(() => consultations.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  medicationName: varchar("medication_name", { length: 255 }).notNull(),
  dosage: varchar("dosage", { length: 100 }).notNull(),
  frequency: varchar("frequency", { length: 100 }).notNull(),
  duration: varchar("duration", { length: 100 }).notNull(),
  instructions: varchar("instructions", { length: 500 }),
  isActive: boolean("is_active").default(true).notNull(),
  prescribedAt: timestamp("prescribed_at").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const diagnoses = pgTable("diagnoses", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  consultationId: integer("consultation_id").references(() => consultations.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  diagnosisCode: varchar("diagnosis_code", { length: 50 }).notNull(),
  diagnosisName: varchar("diagnosis_name", { length: 255 }).notNull(),
  description: text("description"),
  severity: varchar("severity", { length: 50 }),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Laboratory Tables
export const labTests = pgTable("lab_tests", {
  id: serial("id").primaryKey(),
  testName: varchar("test_name", { length: 255 }).notNull(),
  testCode: varchar("test_code", { length: 50 }).notNull(),
  description: text("description"),
  price: numeric("price", { precision: 10, scale: 2 }).notNull(),
  departmentId: integer("department_id").references(() => departments.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const labOrders = pgTable("lab_orders", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  labTestId: integer("lab_test_id").notNull().references(() => labTests.id),
  orderedBy: varchar("ordered_by").notNull().references(() => users.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  orderDate: timestamp("order_date").notNull(),
  status: statusEnum("status").default("pending").notNull(),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const labResults = pgTable("lab_results", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  labTestId: integer("lab_test_id").notNull().references(() => labTests.id),
  consultationId: integer("consultation_id").references(() => consultations.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  resultValue: varchar("result_value", { length: 255 }).notNull(),
  referenceRange: varchar("reference_range", { length: 100 }),
  unit: varchar("unit", { length: 50 }),
  status: varchar("status", { length: 50 }).default("normal").notNull(),
  reportedAt: timestamp("reported_at").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Queue table - Enhanced for Healthcare
export const queue = pgTable("queue", {
  id: serial("id").primaryKey(),
  queueNumber: varchar("queue_number", { length: 50 }).notNull(),
  patientId: integer("patient_id").references(() => patients.id, { onDelete: "cascade" }),
  doctorId: varchar("doctor_id").references(() => doctors.id, { onDelete: "set null" }),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  queuePosition: integer("queue_position"),
  reasonForVisit: varchar("reason_for_visit"),
  category: varchar("category"),
  priority: priorityEnum("priority").default("normal"),
  status: queueStatusEnum("status").default("waiting"),
  checkinTime: timestamp("checkin_time").defaultNow(),
  calledAt: timestamp("called_at"),
  completedAt: timestamp("completed_at"),
  estimatedWaitTime: integer("estimated_wait_time"),
  roomAssigned: varchar("room_assigned"),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Supporting Tables
export const notifications = pgTable("notifications", {
  id: serial("id").primaryKey(),
  title: varchar("title", { length: 255 }).notNull(),
  message: text("message").notNull(),
  type: varchar("type", { length: 50 }).notNull(),
  isRead: boolean("is_read").default(false).notNull(),
  userId: varchar("user_id").references(() => users.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const invoices = pgTable("invoices", {
  id: serial("id").primaryKey(),
  invoiceNumber: varchar("invoice_number", { length: 100 }).notNull(),
  amount: numeric("amount", { precision: 10, scale: 2 }).notNull(),
  status: statusEnum("status").default("pending").notNull(),
  dueDate: timestamp("due_date").notNull(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insuranceClaims = pgTable("insurance_claims", {
  id: serial("id").primaryKey(),
  claimNumber: varchar("claim_number", { length: 100 }).notNull(),
  amount: numeric("amount", { precision: 10, scale: 2 }).notNull(),
  status: statusEnum("status").default("pending").notNull(),
  submittedAt: timestamp("submitted_at").notNull(),
  processedAt: timestamp("processed_at"),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Vitals table - Enhanced for Healthcare
export const vitals = pgTable("vitals", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").references(() => patients.id, { onDelete: "cascade" }),
  queueId: integer("queue_id").references(() => queue.id, { onDelete: "cascade" }),
  doctorId: varchar("doctor_id").references(() => users.id),
  consultationId: integer("consultation_id").references(() => consultations.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  heightCm: numeric("height_cm", { precision: 5, scale: 2 }),
  weightKg: numeric("weight_kg", { precision: 5, scale: 2 }),
  bloodPressure: varchar("blood_pressure", { length: 20 }),
  bloodPressureSystolic: integer("blood_pressure_systolic"),
  bloodPressureDiastolic: integer("blood_pressure_diastolic"),
  heartRate: integer("heart_rate"),
  respiratoryRate: integer("respiratory_rate"),
  temperature: integer("temperature"), // in Fahrenheit * 10
  temperatureC: numeric("temperature_c", { precision: 4, scale: 1 }),
  oxygenSaturation: numeric("oxygen_saturation", { precision: 5, scale: 2 }),
  weight: integer("weight"), // legacy weight in pounds * 10
  height: integer("height"), // legacy height in inches
  recordedBy: varchar("recorded_by").references(() => users.id),
  recordedAt: timestamp("recorded_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Appointments table
export const appointments = pgTable("appointments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").references(() => patients.id, { onDelete: "cascade" }),
  doctorId: varchar("doctor_id").references(() => doctors.id, { onDelete: "cascade" }),
  scheduledTime: timestamp("scheduled_time").notNull(),
  appointmentType: varchar("appointment_type").notNull(),
  duration: integer("duration").default(30), // in minutes
  status: varchar("status").default("scheduled"), // scheduled, completed, cancelled, no_show
  roomNumber: varchar("room_number"),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Staff tasks table
export const staffTasks = pgTable("staff_tasks", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  assignedTo: varchar("assigned_to").references(() => users.id, { onDelete: "cascade" }),
  taskType: varchar("task_type").notNull(), // medication, vitals, assessment, etc.
  description: text("description").notNull(),
  patientId: varchar("patient_id").references(() => patients.id, { onDelete: "cascade" }),
  priority: priorityEnum("priority").default("normal"),
  status: varchar("status").default("pending"), // pending, in_progress, completed
  dueTime: timestamp("due_time"),
  completedAt: timestamp("completed_at"),
  createdBy: varchar("created_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Medical records table
export const medicalRecords = pgTable("medical_records", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").references(() => patients.id, { onDelete: "cascade" }),
  doctorId: varchar("doctor_id").references(() => doctors.id),
  visitDate: timestamp("visit_date").defaultNow(),
  diagnosis: text("diagnosis"),
  prescription: text("prescription"),
  notes: text("notes"),
  followUpRequired: boolean("follow_up_required").default(false),
  followUpDate: timestamp("follow_up_date"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Relations
export const usersRelations = relations(users, ({ one, many }) => ({
  doctor: one(doctors, {
    fields: [users.id],
    references: [doctors.userId],
  }),
  patient: one(patients, {
    fields: [users.id],
    references: [patients.userId],
  }),
  assignedTasks: many(staffTasks),
  createdTasks: many(staffTasks),
  vitalsRecorded: many(vitals),
}));

export const doctorsRelations = relations(doctors, ({ one, many }) => ({
  user: one(users, {
    fields: [doctors.userId],
    references: [users.id],
  }),
  queueItems: many(queue),
  appointments: many(appointments),
  medicalRecords: many(medicalRecords),
}));

export const patientsRelations = relations(patients, ({ one, many }) => ({
  user: one(users, {
    fields: [patients.userId],
    references: [users.id],
  }),
  queueItems: many(queue),
  vitals: many(vitals),
  appointments: many(appointments),
  tasks: many(staffTasks),
  medicalRecords: many(medicalRecords),
}));

export const queueRelations = relations(queue, ({ one, many }) => ({
  patient: one(patients, {
    fields: [queue.patientId],
    references: [patients.id],
  }),
  doctor: one(doctors, {
    fields: [queue.doctorId],
    references: [doctors.id],
  }),
  vitals: many(vitals),
}));

export const vitalsRelations = relations(vitals, ({ one }) => ({
  patient: one(patients, {
    fields: [vitals.patientId],
    references: [patients.id],
  }),
  queueItem: one(queue, {
    fields: [vitals.queueId],
    references: [queue.id],
  }),
  recordedByUser: one(users, {
    fields: [vitals.recordedBy],
    references: [users.id],
  }),
}));

export const appointmentsRelations = relations(appointments, ({ one }) => ({
  patient: one(patients, {
    fields: [appointments.patientId],
    references: [patients.id],
  }),
  doctor: one(doctors, {
    fields: [appointments.doctorId],
    references: [doctors.id],
  }),
}));

export const staffTasksRelations = relations(staffTasks, ({ one }) => ({
  assignedToUser: one(users, {
    fields: [staffTasks.assignedTo],
    references: [users.id],
  }),
  patient: one(patients, {
    fields: [staffTasks.patientId],
    references: [patients.id],
  }),
  createdByUser: one(users, {
    fields: [staffTasks.createdBy],
    references: [users.id],
  }),
}));

export const medicalRecordsRelations = relations(medicalRecords, ({ one }) => ({
  patient: one(patients, {
    fields: [medicalRecords.patientId],
    references: [patients.id],
  }),
  doctor: one(doctors, {
    fields: [medicalRecords.doctorId],
    references: [doctors.id],
  }),
}));

// Insert schemas
export const insertUserSchema = createInsertSchema(users).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertDoctorSchema = createInsertSchema(doctors).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertPatientSchema = createInsertSchema(patients).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertRoomSchema = createInsertSchema(rooms).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertQueueSchema = createInsertSchema(queue).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertVitalsSchema = createInsertSchema(vitals).omit({
  id: true,
});

export const insertAppointmentSchema = createInsertSchema(appointments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertStaffTaskSchema = createInsertSchema(staffTasks).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertMedicalRecordSchema = createInsertSchema(medicalRecords).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

// Healthcare System Insert Schemas
export const insertHospitalSchema = createInsertSchema(hospitals).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertDepartmentSchema = createInsertSchema(departments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertConsultationSchema = createInsertSchema(consultations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertPrescriptionSchema = createInsertSchema(prescriptions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertDiagnosisSchema = createInsertSchema(diagnoses).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertLabTestSchema = createInsertSchema(labTests).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertLabOrderSchema = createInsertSchema(labOrders).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertLabResultSchema = createInsertSchema(labResults).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertNotificationSchema = createInsertSchema(notifications).omit({
  id: true,
  createdAt: true,
});

export const insertInvoiceSchema = createInsertSchema(invoices).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertInsuranceClaimSchema = createInsertSchema(insuranceClaims).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertBedSchema = createInsertSchema(beds).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertWardSchema = createInsertSchema(wards).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

// Types
export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect & {
  role?: 'admin' | 'doctor' | 'nurse' | 'receptionist' | 'patient';
};
export type InsertDoctor = z.infer<typeof insertDoctorSchema>;
export type Doctor = typeof doctors.$inferSelect;
export type InsertPatient = z.infer<typeof insertPatientSchema>;
export type Patient = typeof patients.$inferSelect;
export type InsertRoom = z.infer<typeof insertRoomSchema>;
export type Room = typeof rooms.$inferSelect;
export type InsertQueue = z.infer<typeof insertQueueSchema>;
export type QueueItem = typeof queue.$inferSelect;
export type InsertVitals = z.infer<typeof insertVitalsSchema>;
export type Vitals = typeof vitals.$inferSelect;
export type InsertAppointment = z.infer<typeof insertAppointmentSchema>;
export type Appointment = typeof appointments.$inferSelect;
export type InsertStaffTask = z.infer<typeof insertStaffTaskSchema>;
export type StaffTask = typeof staffTasks.$inferSelect;
export type InsertMedicalRecord = z.infer<typeof insertMedicalRecordSchema>;
export type MedicalRecord = typeof medicalRecords.$inferSelect;

// Healthcare System Types
export type Hospital = typeof hospitals.$inferSelect;
export type InsertHospital = z.infer<typeof insertHospitalSchema>;
export type Department = typeof departments.$inferSelect;
export type InsertDepartment = z.infer<typeof insertDepartmentSchema>;
export type Consultation = typeof consultations.$inferSelect;
export type InsertConsultation = z.infer<typeof insertConsultationSchema>;
export type Prescription = typeof prescriptions.$inferSelect;
export type InsertPrescription = z.infer<typeof insertPrescriptionSchema>;
export type Diagnosis = typeof diagnoses.$inferSelect;
export type InsertDiagnosis = z.infer<typeof insertDiagnosisSchema>;
export type LabTest = typeof labTests.$inferSelect;
export type InsertLabTest = z.infer<typeof insertLabTestSchema>;
export type LabOrder = typeof labOrders.$inferSelect;
export type InsertLabOrder = z.infer<typeof insertLabOrderSchema>;
export type LabResult = typeof labResults.$inferSelect;
export type InsertLabResult = z.infer<typeof insertLabResultSchema>;
export type Notification = typeof notifications.$inferSelect;
export type InsertNotification = z.infer<typeof insertNotificationSchema>;
export type Invoice = typeof invoices.$inferSelect;
export type InsertInvoice = z.infer<typeof insertInvoiceSchema>;
export type InsuranceClaim = typeof insuranceClaims.$inferSelect;
export type InsertInsuranceClaim = z.infer<typeof insertInsuranceClaimSchema>;
export type Bed = typeof beds.$inferSelect;
export type InsertBed = z.infer<typeof insertBedSchema>;
export type Ward = typeof wards.$inferSelect;
export type InsertWard = z.infer<typeof insertWardSchema>;

// Additional Doctor Management Tables
export const doctorSchedule = pgTable("doctor_schedule", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  date: timestamp("date").notNull(),
  startTime: timestamp("start_time").notNull(),
  endTime: timestamp("end_time").notNull(),
  eventType: varchar("event_type").notNull(), // appointment, rounds, break, surgery
  patientId: integer("patient_id").references(() => patients.id),
  title: varchar("title").notNull(),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const aiConsultations = pgTable("ai_consultations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  query: text("query").notNull(),
  aiResponse: text("ai_response").notNull(),
  confidence: varchar("confidence"), // high, medium, low
  consultationType: varchar("consultation_type").notNull(), // diagnosis, treatment, drug_interaction
  createdAt: timestamp("created_at").defaultNow(),
});

export const consultationNotes = pgTable("consultation_notes", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  transcription: text("transcription").notNull(),
  summary: text("summary"),
  recordings: jsonb("recordings"), // Audio file references
  isAiGenerated: boolean("is_ai_generated").notNull().default(false),
  consultationDate: timestamp("consultation_date").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

// Insert schemas for new tables
export const insertDoctorScheduleSchema = createInsertSchema(doctorSchedule).omit({
  id: true,
  createdAt: true,
});

export const insertAiConsultationSchema = createInsertSchema(aiConsultations).omit({
  id: true,
  createdAt: true,
});

export const insertConsultationNoteSchema = createInsertSchema(consultationNotes).omit({
  id: true,
  createdAt: true,
});

// Types for new tables
export type DoctorSchedule = typeof doctorSchedule.$inferSelect;
export type InsertDoctorSchedule = z.infer<typeof insertDoctorScheduleSchema>;
export type AiConsultation = typeof aiConsultations.$inferSelect;
export type InsertAiConsultation = z.infer<typeof insertAiConsultationSchema>;
export type ConsultationNote = typeof consultationNotes.$inferSelect;
export type InsertConsultationNote = z.infer<typeof insertConsultationNoteSchema>;
