import { sql } from 'drizzle-orm';
import { relations } from 'drizzle-orm';
import {
  index,
  jsonb,
  pgTable,
  timestamp,
  varchar,
  integer,
  text,
  boolean,
  numeric,
  pgEnum,
  serial,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Enums
export const userRoleEnum = pgEnum("user_role", ["admin", "doctor", "nurse", "receptionist", "patient"]);
export const roomTypeEnum = pgEnum("room_type", ["icu", "general", "private", "emergency", "operation_theater", "consultation"]);
export const priorityEnum = pgEnum("priority", ["low", "normal", "high", "critical"]);
export const statusEnum = pgEnum("status", ["scheduled", "in_progress", "completed", "cancelled", "waiting", "called", "pending", "approved", "rejected"]);

// Session storage table.
// (IMPORTANT) This table is mandatory for Replit Auth, don't drop it.
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// Core Entity Tables
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
  headDoctorId: varchar("head_doctor_id").notNull(),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const roles = pgTable("roles", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 100 }).notNull(),
  description: varchar("description", { length: 500 }),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const permissions = pgTable("permissions", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 100 }).notNull(),
  description: varchar("description", { length: 500 }),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// User storage table.
// (IMPORTANT) This table is mandatory for Replit Auth, don't drop it.
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: varchar("username", { length: 100 }).unique(),
  email: varchar("email", { length: 255 }).notNull(),
  firstName: varchar("first_name", { length: 100 }),
  lastName: varchar("last_name", { length: 100 }),
  profileImageUrl: varchar("profile_image_url", { length: 500 }),
  role: userRoleEnum("role"),
  departmentId: integer("department_id").references(() => departments.id),
  roleId: integer("role_id").references(() => roles.id),
  phone: varchar("phone", { length: 20 }),
  salary: numeric("salary", { precision: 10, scale: 2 }),
  shift: varchar("shift", { length: 50 }),
  isActive: boolean("is_active").default(true).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
  deletedAt: timestamp("deleted_at"),
});

export const patients = pgTable("patients", {
  id: serial("id").primaryKey(),
  name: varchar("name", { length: 255 }),
  age: integer("age"),
  phone: varchar("phone", { length: 20 }),
  abhaId: varchar("abha_id", { length: 50 }),
  mrn: varchar("mrn", { length: 50 }),
  height: numeric("height", { precision: 5, scale: 2 }),
  weight: numeric("weight", { precision: 5, scale: 2 }),
  emergencyContactName: varchar("emergency_contact_name", { length: 255 }),
  emergencyContactPhone: varchar("emergency_contact_phone", { length: 20 }),
  deletedAt: timestamp("deleted_at"),
});

// Relationship Tables
export const userHospitals = pgTable("user_hospitals", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull().references(() => users.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  roleId: integer("role_id").notNull().references(() => roles.id),
  isPrimary: boolean("is_primary").default(false).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const rolePermissions = pgTable("role_permissions", {
  id: serial("id").primaryKey(),
  roleId: integer("role_id").notNull().references(() => roles.id),
  permissionId: integer("permission_id").notNull().references(() => permissions.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const staffAssignments = pgTable("staff_assignments", {
  id: serial("id").primaryKey(),
  userId: varchar("user_id").notNull().references(() => users.id),
  departmentId: integer("department_id").notNull().references(() => departments.id),
  shift: varchar("shift", { length: 50 }),
  startDate: timestamp("start_date"),
  endDate: timestamp("end_date"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  deletedAt: timestamp("deleted_at"),
});

// Clinical Tables
export const consultations = pgTable("consultations", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  consultationTime: timestamp("consultation_time").notNull(),
  notes: text("notes"),
  status: statusEnum("status").default("scheduled").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const vitals = pgTable("vitals", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").references(() => users.id),
  consultationId: integer("consultation_id").references(() => consultations.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  heightCm: numeric("height_cm", { precision: 5, scale: 2 }),
  weightKg: numeric("weight_kg", { precision: 5, scale: 2 }),
  bloodPressure: varchar("blood_pressure", { length: 20 }),
  heartRate: integer("heart_rate"),
  respiratoryRate: integer("respiratory_rate"),
  temperatureC: numeric("temperature_c", { precision: 4, scale: 1 }),
  oxygenSaturation: numeric("oxygen_saturation", { precision: 5, scale: 2 }),
  recordedAt: timestamp("recorded_at").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const appointments = pgTable("appointments", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  appointmentDate: timestamp("appointment_date").notNull(),
  duration: integer("duration").default(30).notNull(),
  status: statusEnum("status").default("scheduled").notNull(),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const prescriptions = pgTable("prescriptions", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  consultationId: integer("consultation_id").references(() => consultations.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
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

export const medicalRecords = pgTable("medical_records", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  consultationId: integer("consultation_id").references(() => consultations.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  recordType: varchar("record_type", { length: 100 }).notNull(),
  content: text("content").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const diagnoses = pgTable("diagnoses", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  consultationId: integer("consultation_id").references(() => consultations.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
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
  departmentId: integer("department_id").notNull().references(() => departments.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const labOrders = pgTable("lab_orders", {
  id: serial("id").primaryKey(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  labTestId: integer("lab_test_id").notNull().references(() => labTests.id),
  orderedBy: varchar("ordered_by").notNull().references(() => users.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
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
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  resultValue: varchar("result_value", { length: 255 }).notNull(),
  referenceRange: varchar("reference_range", { length: 100 }),
  unit: varchar("unit", { length: 50 }),
  status: varchar("status", { length: 50 }).default("normal").notNull(),
  reportedAt: timestamp("reported_at").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Facility Management Tables
export const rooms = pgTable("rooms", {
  id: serial("id").primaryKey(),
  roomNumber: varchar("room_number", { length: 50 }).notNull(),
  roomType: roomTypeEnum("room_type").notNull(),
  capacity: integer("capacity").default(0).notNull(),
  isAvailable: boolean("is_available").default(true).notNull(),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  departmentId: integer("department_id").references(() => departments.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
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

// Operational Tables
export const queueEntries = pgTable("queue_entries", {
  id: serial("id").primaryKey(),
  queueNumber: varchar("queue_number", { length: 50 }).notNull(),
  priority: priorityEnum("priority").default("normal").notNull(),
  status: statusEnum("status").default("waiting").notNull(),
  estimatedWaitTime: integer("estimated_wait_time"),
  checkedInAt: timestamp("checked_in_at").notNull(),
  calledAt: timestamp("called_at"),
  completedAt: timestamp("completed_at"),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

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

// Financial Tables  
export const invoices = pgTable("invoices", {
  id: serial("id").primaryKey(),
  invoiceNumber: varchar("invoice_number", { length: 100 }).notNull(),
  amount: numeric("amount", { precision: 10, scale: 2 }).notNull(),
  status: statusEnum("status").default("pending").notNull(),
  dueDate: timestamp("due_date").notNull(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
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
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// Supporting Tables
export const documents = pgTable("documents", {
  id: serial("id").primaryKey(),
  fileName: varchar("file_name", { length: 255 }).notNull(),
  filePath: varchar("file_path", { length: 500 }).notNull(),
  fileType: varchar("file_type", { length: 100 }).notNull(),
  fileSize: integer("file_size").notNull(),
  patientId: integer("patient_id").notNull().references(() => patients.id),
  hospitalId: integer("hospital_id").notNull().references(() => hospitals.id),
  uploadedBy: varchar("uploaded_by").notNull().references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const auditLogs = pgTable("audit_logs", {
  id: serial("id").primaryKey(),
  action: varchar("action", { length: 100 }).notNull(),
  tableName: varchar("table_name", { length: 100 }).notNull(),
  recordId: integer("record_id"),
  oldValues: jsonb("old_values"),
  newValues: jsonb("new_values"),
  userId: varchar("user_id").references(() => users.id),
  hospitalId: integer("hospital_id").references(() => hospitals.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Insert schemas
export const insertPatientSchema = createInsertSchema(patients);
export const insertConsultationSchema = createInsertSchema(consultations);
export const insertAppointmentSchema = createInsertSchema(appointments);
export const insertQueueEntrySchema = createInsertSchema(queueEntries);
export const insertVitalSchema = createInsertSchema(vitals);
export const insertPrescriptionSchema = createInsertSchema(prescriptions);
export const insertLabOrderSchema = createInsertSchema(labOrders);
export const insertLabResultSchema = createInsertSchema(labResults);

// Type exports
export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;
export type Hospital = typeof hospitals.$inferSelect;
export type Patient = typeof patients.$inferSelect;
export type InsertPatient = z.infer<typeof insertPatientSchema>;
export type Department = typeof departments.$inferSelect;
export type Consultation = typeof consultations.$inferSelect;
export type InsertConsultation = z.infer<typeof insertConsultationSchema>;
export type Appointment = typeof appointments.$inferSelect;
export type InsertAppointment = z.infer<typeof insertAppointmentSchema>;
export type QueueEntry = typeof queueEntries.$inferSelect;
export type InsertQueueEntry = z.infer<typeof insertQueueEntrySchema>;
export type Room = typeof rooms.$inferSelect;
export type Bed = typeof beds.$inferSelect;
export type Vital = typeof vitals.$inferSelect;
export type InsertVital = z.infer<typeof insertVitalSchema>;
export type Prescription = typeof prescriptions.$inferSelect;
export type LabOrder = typeof labOrders.$inferSelect;
export type LabResult = typeof labResults.$inferSelect;
export type Notification = typeof notifications.$inferSelect;