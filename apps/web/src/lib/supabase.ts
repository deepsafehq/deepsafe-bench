import { createBrowserClient } from '@supabase/ssr';

export type SupabaseClient = ReturnType<typeof createBrowserClient>;

/**
 * Create a Supabase browser client, or null when auth is not configured.
 *
 * The public build ships without Supabase credentials. Returning null rather
 * than throwing lets the static site build and render while auth-gated routes
 * stay inert. Self-hosters enable auth by setting NEXT_PUBLIC_SUPABASE_URL and
 * NEXT_PUBLIC_SUPABASE_ANON_KEY.
 */
export function createClient(): SupabaseClient | null {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anonKey) {
    return null;
  }
  return createBrowserClient(url, anonKey);
}
