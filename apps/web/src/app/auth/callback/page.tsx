"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/auth-provider";

/**
 * OAuth callback page. Exchanges the auth code for a session.
 * API key auto-creation is handled by the AuthProvider's onAuthStateChange.
 */
export default function AuthCallbackPage() {
  const router = useRouter();
  const { supabase } = useAuth();

  useEffect(() => {
    const handleCallback = async () => {
      const params = new URLSearchParams(window.location.search);
      const code = params.get("code");

      if (code) {
        const { error } = await supabase.auth.exchangeCodeForSession(code);
        if (!error) {
          router.replace("/");
          return;
        }
      }

      router.replace("/login?error=auth_failed");
    };

    handleCallback();
  }, [router, supabase]);

  return (
    <div
      data-theme="app"
      className="min-h-screen bg-background flex items-center justify-center"
    >
      <div className="text-text-secondary text-sm">Completing sign in...</div>
    </div>
  );
}
