import 'server-only';

import { neon } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-http';
import {
  pgTable,
  text,
  numeric,
  integer,
  timestamp,
  pgEnum,
  serial,
  jsonb
} from 'drizzle-orm/pg-core';
import { eq, and } from 'drizzle-orm';
import { createInsertSchema } from 'drizzle-zod';

export const db = drizzle(neon(process.env.POSTGRES_URL!));

export const statusEnum = pgEnum('status', ['active', 'inactive', 'archived']);

export const annotations = pgTable('annotations', {
  id: serial('id').primaryKey(),
  dataset: text('dataset').notNull(),
  numberOfAnnotations: integer('number_of_annotations').notNull(),
  username: text('username').notNull(),
  evaluations: jsonb('evaluations').notNull(),
  mapping: jsonb('mapping').notNull(),
  tests: jsonb('tests').notNull(),
  createdAt: timestamp('created_at').defaultNow().notNull()
});

export async function getAnnotations() {
  return await db.select().from(annotations);
}

export async function saveAnnotation(annotationData: any) {
  await db.insert(annotations).values(annotationData);
}

export async function updateAnnotation(id: string, annotationData: any) {
  await db
    .update(annotations)
    .set(annotationData)
    .where(eq(annotations.id, parseInt(id)));
}
