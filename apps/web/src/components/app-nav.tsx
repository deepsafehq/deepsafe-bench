"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { DeepSafeLogo } from "@/components/logo";
import { Menu, X, LogOut, User } from "lucide-react";
import { useAuth } from "@/components/auth-provider";

export function AppNav() {
  const pathname = usePathname();
  const { user, signOut } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);

  const links = [
    { href: "/", label: "Analyze" },
    { href: "/dashboard", label: "Dashboard" },
    { href: "/api-keys", label: "API Keys" },
  ];

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/" || pathname === "/demo";
    return pathname === href;
  };

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-surface backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div className="flex items-center">
            <DeepSafeLogo size="sm" showWordmark={false} variant="dark" />
          </div>
          <nav className="hidden sm:flex items-center gap-1">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                  isActive(link.href)
                    ? "text-accent bg-accent-light"
                    : "text-text-secondary hover:text-text-primary hover:bg-surface-raised"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-3">
          {user?.email && (
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 text-sm text-text-tertiary">
              <User className="w-3.5 h-3.5" />
              <span className="max-w-[180px] truncate">{user.email}</span>
            </div>
          )}
          <button
            onClick={signOut}
            className="hidden sm:flex items-center gap-2 px-3 py-1.5 text-sm text-text-secondary hover:text-text-primary transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="sm:hidden p-2 text-text-secondary hover:text-text-primary"
            aria-label="Toggle menu"
          >
            {mobileOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
      {mobileOpen && (
        <div className="sm:hidden border-t border-border bg-surface backdrop-blur-md">
          <nav className="flex flex-col px-4 py-3 gap-1">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileOpen(false)}
                className={`px-3 py-2 text-sm rounded-md ${
                  isActive(link.href)
                    ? "text-accent bg-accent-light"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                {link.label}
              </Link>
            ))}
            {user?.email && (
              <div className="px-3 py-2 text-xs text-text-tertiary truncate mt-2 border-t border-border pt-3">
                {user.email}
              </div>
            )}
            <button
              onClick={signOut}
              className="px-3 py-2 text-sm text-text-secondary hover:text-text-primary text-left flex items-center gap-2"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </button>
          </nav>
        </div>
      )}
    </header>
  );
}
