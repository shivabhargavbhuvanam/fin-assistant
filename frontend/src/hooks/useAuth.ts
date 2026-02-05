/**
 * Authentication hook for Smart Financial Coach.
 * 
 * Integrates Clerk with the API client for authenticated requests.
 * Handles auth state and token management.
 */

import { useEffect, useState } from 'react';
import { useAuth as useClerkAuth, useUser } from '@clerk/clerk-react';
import { setTokenGetter, getCurrentUser } from '../api/client';
import type { UserSession } from '../types';

interface AuthState {
  isLoaded: boolean;
  isSignedIn: boolean;
  userId: string | null;
  userInfo: UserSession | null;
  error: string | null;
}

/**
 * Hook to manage authentication state and Clerk integration.
 */
export function useAuth() {
  const { isLoaded, isSignedIn, userId, getToken } = useClerkAuth();
  const { user } = useUser();
  
  const [userInfo, setUserInfo] = useState<UserSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingUser, setIsLoadingUser] = useState(false);

  // Set up token getter for API client when Clerk is ready
  useEffect(() => {
    if (isLoaded && isSignedIn) {
      setTokenGetter(async () => {
        try {
          return await getToken();
        } catch (e) {
          console.error('Failed to get auth token:', e);
          return null;
        }
      });
    }
  }, [isLoaded, isSignedIn, getToken]);

  // Fetch user info when signed in
  useEffect(() => {
    if (isLoaded && isSignedIn && userId) {
      setIsLoadingUser(true);
      getCurrentUser()
        .then(setUserInfo)
        .catch((e) => {
          console.error('Failed to fetch user info:', e);
          setError(e.message);
        })
        .finally(() => setIsLoadingUser(false));
    } else if (isLoaded && !isSignedIn) {
      setUserInfo(null);
    }
  }, [isLoaded, isSignedIn, userId]);

  const state: AuthState = {
    isLoaded: isLoaded && !isLoadingUser,
    isSignedIn: isSignedIn ?? false,
    userId: userId ?? null,
    userInfo,
    error,
  };

  return {
    ...state,
    user,
    refreshUserInfo: async () => {
      if (isSignedIn) {
        try {
          const info = await getCurrentUser();
          setUserInfo(info);
          return info;
        } catch (e) {
          console.error('Failed to refresh user info:', e);
          throw e;
        }
      }
      return null;
    },
  };
}

/**
 * Hook for development/bypass mode when Clerk is not configured.
 * Returns a mock auth state.
 */
export function useBypassAuth() {
  const [userInfo, setUserInfo] = useState<UserSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Set up token getter that returns null (no auth)
    setTokenGetter(async () => null);
    
    // Try to fetch user info (will fail gracefully if backend requires auth)
    getCurrentUser()
      .then(setUserInfo)
      .catch(() => {
        // Expected in bypass mode - backend might still require auth
        setUserInfo(null);
      })
      .finally(() => setIsLoading(false));
  }, []);

  return {
    isLoaded: !isLoading,
    isSignedIn: true, // Always "signed in" in bypass mode
    userId: 'bypass_user',
    userInfo,
    user: null,
    error: null,
    refreshUserInfo: async () => {
      try {
        const info = await getCurrentUser();
        setUserInfo(info);
        return info;
      } catch {
        return null;
      }
    },
  };
}
