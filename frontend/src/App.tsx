/**
 * App Component
 * 
 * Main application component handling view routing:
 * Upload → Analyzing → Dashboard
 * 
 * Includes Clerk authentication integration.
 * Wrapped in ErrorBoundary for graceful error handling.
 */

import { useState, useCallback, useEffect } from 'react';
import { Loader2, LogIn, LogOut, User } from 'lucide-react';
import {
  SignInButton,
  SignOutButton,
  SignedIn,
  useAuth as useClerkAuth,
  useUser as useClerkUser
} from '@clerk/clerk-react';
import type { DashboardResponse, AppView } from './types';
import { Upload } from './components/Upload';
import { Dashboard } from './components/Dashboard';
import { ErrorBoundary } from './components/ErrorBoundary';
import {
  uploadAndAnalyze,
  loadSampleAndAnalyze,
  setTokenGetter,
  getSessions,
  getDashboard,
  analyzeSession
} from './api/client';
import type { Session } from './types';

// Check if Clerk is configured
const CLERK_ENABLED = !!import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

/**
 * Wrapper component that uses Clerk hooks.
 * Only rendered when Clerk is enabled.
 */
function AppWithClerk() {
  const { isLoaded, isSignedIn, getToken } = useClerkAuth();
  const { user } = useClerkUser();

  return (
    <AppContent 
      isLoaded={isLoaded} 
      isSignedIn={isSignedIn ?? false} 
      getToken={getToken}
      user={user}
      clerkEnabled={true}
    />
  );
}

/**
 * Wrapper component for non-Clerk mode (development).
 */
function AppWithoutClerk() {
  return (
    <AppContent 
      isLoaded={true} 
      isSignedIn={true} 
      getToken={async () => null}
      user={null}
      clerkEnabled={false}
    />
  );
}

interface AppContentProps {
  isLoaded: boolean;
  isSignedIn: boolean;
  getToken: () => Promise<string | null>;
  user: { firstName?: string | null; primaryEmailAddress?: { emailAddress?: string } } | null;
  clerkEnabled: boolean;
}

function AppContent({ isLoaded, isSignedIn, getToken, user, clerkEnabled }: AppContentProps) {

  // Application state
  const [view, setView] = useState<AppView>('upload');
  const [dashboardData, setDashboardData] = useState<DashboardResponse | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingStep, setLoadingStep] = useState<string>('');

  // Set up token getter and fetch sessions when authenticated
  useEffect(() => {
    if (isLoaded && isSignedIn) {
      // First, set up the token getter
      setTokenGetter(async () => {
        try {
          return await getToken();
        } catch {
          return null;
        }
      });
      
      // Then fetch sessions (small delay to ensure token getter is set)
      console.log('[App] Auth ready, fetching sessions...');
      getSessions()
        .then(data => {
          console.log('[App] Sessions loaded:', data);
          setSessions(data.sessions);
          if (data.active_session_id) {
            setActiveSessionId(data.active_session_id);
          }
        })
        .catch(err => {
          console.log('[App] No existing sessions:', err.message);
        });
    }
  }, [isLoaded, isSignedIn, getToken]);

  // Handle session switch
  const handleSwitchSession = useCallback(async (sessionId: string) => {
    setIsLoading(true);
    setLoadingStep('Loading session...');
    setView('analyzing');

    try {
      // If session isn't analyzed yet, analyze it
      const session = sessions.find(s => s.id === sessionId);
      if (session?.status === 'processing') {
        setLoadingStep('Analyzing transactions...');
        await analyzeSession(sessionId);
      }

      setLoadingStep('Loading dashboard...');
      const data = await getDashboard(sessionId);
      setDashboardData(data);
      setActiveSessionId(sessionId);
      setView('dashboard');
    } catch (err) {
      console.error('[App] Session switch error:', err);
      setError(err instanceof Error ? err.message : 'Failed to load session');
      setView('upload');
    } finally {
      setIsLoading(false);
      setLoadingStep('');
    }
  }, [sessions]);

  // Handle file upload
  const handleFileUpload = useCallback(async (file: File) => {
    console.log('[App] Starting file upload:', file.name);
    setIsLoading(true);
    setError(null);
    setView('analyzing');

    try {
      const data = await uploadAndAnalyze(file, (step) => {
        console.log('[App] Progress:', step);
        setLoadingStep(step);
      });
      console.log('[App] Upload complete, dashboard data:', data);
      setDashboardData(data);
      setActiveSessionId(data.session_id);
      // Refresh sessions list
      getSessions().then(data => setSessions(data.sessions)).catch(() => { });
      setView('dashboard');
    } catch (err) {
      console.error('[App] Upload error:', err);
      const errorMessage = err instanceof Error
        ? (err as { details?: string }).details || err.message
        : 'Failed to analyze file';
      setError(errorMessage);
      setView('upload');
    } finally {
      setIsLoading(false);
      setLoadingStep('');
    }
  }, []);

  // Handle sample data
  const handleUseSampleData = useCallback(async () => {
    console.log('[App] Loading sample data...');
    setIsLoading(true);
    setError(null);
    setView('analyzing');

    try {
      const data = await loadSampleAndAnalyze((step) => {
        console.log('[App] Progress:', step);
        setLoadingStep(step);
      });
      console.log('[App] Sample data loaded, dashboard data:', data);
      setDashboardData(data);
      setActiveSessionId(data.session_id);
      // Refresh sessions list
      getSessions().then(data => setSessions(data.sessions)).catch(() => { });
      setView('dashboard');
    } catch (err) {
      console.error('[App] Sample data error:', err);
      const errorMessage = err instanceof Error
        ? (err as { details?: string }).details || err.message
        : 'Failed to load sample data';
      setError(errorMessage);
      setView('upload');
    } finally {
      setIsLoading(false);
      setLoadingStep('');
    }
  }, []);

  // Reset to upload view
  const handleReset = useCallback(() => {
    setView('upload');
    setDashboardData(null);
    setError(null);
  }, []);

  // Show loading while Clerk initializes
  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-accent animate-spin" />
      </div>
    );
  }

  // Show sign-in for unauthenticated users (when Clerk is enabled)
  if (clerkEnabled && !isSignedIn) {
    return <SignInView />;
  }

  // Render based on current view
  switch (view) {
    case 'upload':
      return (
        <div className="relative">
          <UserHeader user={user} clerkEnabled={clerkEnabled} />
          <Upload
            onFileUpload={handleFileUpload}
            onUseSampleData={handleUseSampleData}
            isLoading={isLoading}
            error={error}
            sessions={sessions}
            onSwitchSession={handleSwitchSession}
            user={user ? { username: user.firstName || user.primaryEmailAddress?.emailAddress?.split('@')[0] } : undefined}
          />
        </div>
      );

    case 'analyzing':
      return (
        <div className="relative">
          <UserHeader user={user} clerkEnabled={clerkEnabled} />
          <AnalyzingView step={loadingStep} />
        </div>
      );

    case 'dashboard':
      if (!dashboardData) {
        return <AnalyzingView step="Loading dashboard..." />;
      }
      return (
        <div className="relative">
          <UserHeader user={user} clerkEnabled={clerkEnabled} />
          <Dashboard
            data={dashboardData}
            onReset={handleReset}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSwitchSession={handleSwitchSession}
            user={user ? { username: user.firstName || user.primaryEmailAddress?.emailAddress?.split('@')[0] } : undefined}
          />
        </div>
      );

    default:
      return null;
  }
}

// =============================================================================
// Analyzing View (Loading State)
// =============================================================================

interface AnalyzingViewProps {
  step: string;
}

function AnalyzingView({ step }: AnalyzingViewProps) {
  return (
    <div className="min-h-screen bg-bg flex items-center justify-center px-4">
      <div className="text-center">
        {/* Spinner */}
        <div className="flex justify-center mb-5">
          <Loader2 className="w-8 h-8 text-accent animate-spin" />
        </div>

        {/* Title */}
        <h2 className="text-base font-semibold text-textPrimary mb-1">
          Analyzing Your Finances
        </h2>

        {/* Step indicator */}
        <p className="text-sm text-muted mb-6">
          {step || 'Processing transactions...'}
        </p>

        {/* Steps list */}
        <div className="max-w-xs mx-auto space-y-2 text-left">
          <StepItem
            label="Categorizing transactions"
            isActive={step.includes('Analyzing')}
          />
          <StepItem
            label="Detecting anomalies"
            isActive={step.includes('Analyzing')}
          />
          <StepItem
            label="Finding recurring charges"
            isActive={step.includes('Analyzing')}
          />
          <StepItem
            label="Generating insights"
            isActive={step.includes('Analyzing')}
          />
        </div>
      </div>
    </div>
  );
}

// Step item component
function StepItem({ label, isActive }: { label: string; isActive: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div
        className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-accent' : 'bg-border'
          }`}
      />
      <span className={`text-sm ${isActive ? 'text-textPrimary' : 'text-muted'
        }`}>
        {label}
      </span>
    </div>
  );
}

// =============================================================================
// Sign In View (when Clerk is enabled)
// =============================================================================

function SignInView() {
  return (
    <div className="min-h-screen bg-bg flex items-center justify-center px-4">
      <div className="max-w-md w-full text-center">
        {/* Logo/Brand */}
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-textPrimary mb-2">
            Smart Financial Coach
          </h1>
          <p className="text-muted text-sm">
            AI-powered insights for your spending habits
          </p>
        </div>

        {/* Sign In Card */}
        <div className="bg-surface border border-border rounded-xl p-8">
          <div className="mb-6">
            <div className="w-16 h-16 mx-auto mb-4 bg-bg rounded-full flex items-center justify-center">
              <User className="w-8 h-8 text-muted" />
            </div>
            <h2 className="text-lg font-medium text-textPrimary mb-2">
              Welcome Back
            </h2>
            <p className="text-sm text-muted">
              Sign in to access your financial data and insights
            </p>
          </div>

          <SignInButton mode="modal">
            <button className="w-full px-4 py-3 bg-accent text-bg rounded-lg font-medium flex items-center justify-center gap-2 hover:opacity-90 transition-opacity">
              <LogIn className="w-5 h-5" />
              Sign In
            </button>
          </SignInButton>
        </div>

        {/* Footer */}
        <p className="mt-6 text-xs text-muted">
          Your data is encrypted and secure
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// User Header (when signed in)
// =============================================================================

interface UserHeaderProps {
  user: { firstName?: string | null; primaryEmailAddress?: { emailAddress?: string } } | null;
  clerkEnabled: boolean;
}

function UserHeader({ user, clerkEnabled }: UserHeaderProps) {
  if (!clerkEnabled) return null;

  return (
    <div className="absolute top-4 right-4 flex items-center gap-3">
      <SignedIn>
        <div className="flex items-center gap-2 text-sm text-muted">
          <span>{user?.primaryEmailAddress?.emailAddress || 'User'}</span>
        </div>
        <SignOutButton>
          <button className="p-2 text-muted hover:text-textPrimary transition-colors" title="Sign out">
            <LogOut className="w-4 h-4" />
          </button>
        </SignOutButton>
      </SignedIn>
    </div>
  );
}

// =============================================================================
// Main App with Error Boundary
// =============================================================================

function App() {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        // Log to console (could also send to error tracking service)
        console.error('[App] Caught error:', error.message);
        console.error('[App] Component stack:', errorInfo.componentStack);
      }}
    >
      {CLERK_ENABLED ? <AppWithClerk /> : <AppWithoutClerk />}
    </ErrorBoundary>
  );
}

export default App;
