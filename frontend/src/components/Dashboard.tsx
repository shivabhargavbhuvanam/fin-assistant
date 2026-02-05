/**
 * Dashboard Component
 * 
 * Main view with financial insights + chat interface side-by-side.
 * Layout:
 * - Left: Financial data (summary, anomalies, recurring, insights)
 * - Right: AI Chat interface
 */

import { useState, useEffect } from 'react';
import { ArrowLeft, AlertTriangle, RefreshCw, TrendingDown, TrendingUp } from 'lucide-react';
import type { DashboardResponse, RecurringCharge, Anomaly, Insight } from '../types';
import { SpendingChart, SummaryStats } from './SpendingChart';
import { ChatInterface } from './ChatInterface';
import { FinancialFortuneCookie } from './FinancialFortuneCookie';

const API_BASE = import.meta.env.VITE_API_BASE_URL;


interface DashboardProps {
  data: DashboardResponse;
  onReset: () => void;
  sessions?: { id: string; name: string | null; is_sample: boolean }[];
  activeSessionId?: string | null;
  onSwitchSession?: (sessionId: string) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  user?: any;
}

export function Dashboard({
  data,
  onReset,
  sessions = [],
  activeSessionId,
  onSwitchSession,
  user
}: DashboardProps) {
  const { summary, insights, anomalies, recurring_charges } = data;
  const [suggestedPrompts, setSuggestedPrompts] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'anomalies' | 'recurring' | 'insights'>('overview');

  // Calculate gray charges total
  const grayCharges = recurring_charges.filter(r => r.is_gray_charge);
  const grayChargesTotal = grayCharges.reduce((sum, r) => sum + Math.abs(r.amount), 0);

  // Fetch suggested prompts on load
  useEffect(() => {
    const fetchPrompts = async () => {
      try {
        const response = await fetch(`${API_BASE}/chat/${data.session_id}/prompts`);
        if (response.ok) {
          const result = await response.json();
          setSuggestedPrompts(result.prompts || []);
        }
      } catch (err) {
        console.error('Failed to fetch prompts:', err);
        // Fallback prompts
        setSuggestedPrompts([
          "What are my biggest expenses?",
          "Tell me about unusual transactions",
          "How can I save more money?",
          "What subscriptions am I paying for?"
        ]);
      }
    };
    fetchPrompts();
  }, [data.session_id]);

  // Count high-priority items for badges
  const highAnomalyCount = anomalies.filter(a => a.severity === 'high').length;
  const grayChargeCount = grayCharges.length;

  return (
    <div className="h-screen flex flex-col bg-bg overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 bg-surface border-b border-border/60">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onReset}
              className="p-1.5 rounded-lg text-muted hover:text-textPrimary hover:bg-bg transition-colors"
              aria-label="Upload new file"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div className="flex flex-col">
              {user?.username ? (
                <h1 className="text-xl sm:text-2xl font-medium tracking-wide text-textPrimary">
                  {user.username}'s DINERA
                </h1>
              ) : (
                <h1 className="text-xl sm:text-2xl font-medium tracking-wide text-textPrimary">
                  DINERA
                </h1>
              )}
            </div>

            {/* Fortune Cookie - beside the title */}
            <FinancialFortuneCookie sessionId={data.session_id} />
          </div>
        </div>
      </header>

      {/* Main Content - Responsive Two Column Layout */}
      <main className="flex-1 min-h-0 overflow-hidden">
        <div className="h-full max-w-7xl mx-auto px-4 sm:px-6 py-4">
          <div className="h-full grid grid-cols-1 lg:grid-cols-5 gap-4 lg:gap-6">

            {/* Left Column: Financial Data (3/5 on desktop) */}
            <div className="lg:col-span-3 min-h-0 overflow-y-auto space-y-4 pb-4 lg:pr-2">

              {/* Top Insight Hero - Most Actionable Insight */}
              {insights.length > 0 && (
                <TopInsightHero insight={insights.find(i => i.priority === 1) || insights[0]} />
              )}

              {/* Quick Stats Row */}
              <div className="grid grid-cols-3 gap-3">
                <QuickStatCard
                  label="Income"
                  value={summary.total_income}
                  icon={<TrendingUp className="w-4 h-4" />}
                  positive
                />
                <QuickStatCard
                  label="Spending"
                  value={summary.total_spending}
                  icon={<TrendingDown className="w-4 h-4" />}
                />
                <QuickStatCard
                  label="Net"
                  value={summary.net}
                  icon={summary.net >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                  positive={summary.net >= 0}
                />
              </div>

              {/* Tab Navigation */}
              <div className="flex gap-1 border-b border-border/60 overflow-x-auto">
                <TabButton
                  active={activeTab === 'overview'}
                  onClick={() => setActiveTab('overview')}
                >
                  Overview
                </TabButton>
                <TabButton
                  active={activeTab === 'anomalies'}
                  onClick={() => setActiveTab('anomalies')}
                  badge={highAnomalyCount > 0 ? highAnomalyCount : undefined}
                  alert
                >
                  Anomalies
                </TabButton>
                <TabButton
                  active={activeTab === 'recurring'}
                  onClick={() => setActiveTab('recurring')}
                  badge={grayChargeCount > 0 ? grayChargeCount : undefined}
                >
                  Recurring
                </TabButton>
                <TabButton
                  active={activeTab === 'insights'}
                  onClick={() => setActiveTab('insights')}
                >
                  Insights
                </TabButton>
              </div>

              {/* Tab Content */}
              <div>
                {activeTab === 'overview' && (
                  <div className="space-y-4">
                    <SpendingChart summary={summary} />

                    {/* Priority Alert: Anomalies */}
                    {anomalies.length > 0 && (
                      <PriorityAlert
                        title={`${anomalies.length} Unusual Transaction${anomalies.length > 1 ? 's' : ''} Detected`}
                        description="Some transactions are significantly different from your normal spending patterns."
                        onClick={() => setActiveTab('anomalies')}
                      />
                    )}

                    {/* Priority Alert: Gray Charges */}
                    {grayCharges.length > 0 && (
                      <PriorityAlert
                        title={`${grayCharges.length} Forgotten Subscription${grayCharges.length > 1 ? 's' : ''}`}
                        description={`Small recurring charges totaling $${grayChargesTotal.toFixed(2)}/month`}
                        onClick={() => setActiveTab('recurring')}
                      />
                    )}
                  </div>
                )}

                {activeTab === 'anomalies' && (
                  <AnomaliesList anomalies={anomalies} />
                )}

                {activeTab === 'recurring' && (
                  <RecurringChargesTable
                    charges={recurring_charges}
                    grayChargesTotal={grayChargesTotal}
                  />
                )}

                {activeTab === 'insights' && (
                  <InsightsList insights={insights} />
                )}
              </div>
            </div>

            {/* Right Column: Chat Interface (2/5 on desktop, fixed height) */}
            <div className="lg:col-span-2 min-h-[400px] lg:min-h-0 h-[400px] lg:h-full">
              <ChatInterface
                sessionId={data.session_id}
                suggestedPrompts={suggestedPrompts}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// =============================================================================
// Quick Stat Card
// =============================================================================

// =============================================================================
// Top Insight Hero - Prominent display for most actionable insight
// =============================================================================

interface TopInsightHeroProps {
  insight: Insight;
}

function TopInsightHero({ insight }: TopInsightHeroProps) {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  const getEmoji = (type: string) => {
    switch (type) {
      case 'savings': return '💡';
      case 'spending': return '📊';
      case 'anomaly': return '⚠️';
      case 'subscription': return '🔄';
      default: return '💡';
    }
  };

  // Extract savings amount from action or data
  const savingsMatch = insight.action?.match(/\$[\d,]+/);
  const savingsAmount = savingsMatch ? savingsMatch[0] : null;

  return (
    <div className="bg-surface rounded-2xl p-4 shadow-sm border border-border/40 relative">
      <button
        onClick={() => setDismissed(true)}
        className="absolute top-3 right-3 text-muted hover:text-textSecondary text-xs"
        aria-label="Dismiss"
      >
        ✕
      </button>

      <div className="flex items-start gap-3">
        <div className="text-2xl flex-shrink-0">{getEmoji(insight.type)}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-medium text-muted uppercase tracking-wide">Top Insight</span>
            {insight.priority === 1 && (
              <span className="text-[10px] px-1.5 py-0.5 bg-accent/10 text-accent rounded-full">High Priority</span>
            )}
          </div>
          <h3 className="text-sm font-semibold text-textPrimary mb-1">{insight.title}</h3>
          <p className="text-xs text-textSecondary leading-relaxed mb-2">{insight.description}</p>

          {savingsAmount && (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-bg rounded-lg">
              <span className="text-xs font-medium text-textPrimary">
                Potential savings: {savingsAmount}/year
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Quick Stat Card
// =============================================================================

interface QuickStatCardProps {
  label: string;
  value: number;
  icon: React.ReactNode;
  positive?: boolean;
}

function QuickStatCard({ label, value, icon, positive }: QuickStatCardProps) {
  const formatCurrency = (v: number) => {
    const prefix = v >= 0 ? '' : '-';
    return prefix + '$' + Math.abs(v).toLocaleString('en-US', {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    });
  };

  return (
    <div className="bg-surface rounded-xl p-3 sm:p-4 shadow-sm border border-border/40">
      <div className="flex items-center justify-between">
        <span className="text-[10px] sm:text-xs text-muted uppercase tracking-wide">{label}</span>
        <span className={positive ? 'text-textPrimary' : 'text-muted'}>{icon}</span>
      </div>
      <p className="text-lg sm:text-xl font-semibold text-textPrimary mt-1">
        {formatCurrency(value)}
      </p>
    </div>
  );
}

// =============================================================================
// Tab Button
// =============================================================================

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
  badge?: number;
  alert?: boolean;
}

function TabButton({ active, onClick, children, badge, alert }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${active
        ? 'border-accent text-textPrimary'
        : 'border-transparent text-muted hover:text-textSecondary'
        }`}
    >
      <span className="flex items-center gap-1.5">
        <span>{children}</span>
        {badge !== undefined && (
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${alert ? 'bg-accent/90 text-white' : 'bg-bg text-muted'
            }`}>
            {badge}
          </span>
        )}
      </span>
    </button>
  );
}

// =============================================================================
// Priority Alert
// =============================================================================

interface PriorityAlertProps {
  title: string;
  description: string;
  onClick: () => void;
}

function PriorityAlert({ title, description, onClick }: PriorityAlertProps) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left bg-surface rounded-xl p-4 shadow-sm border border-border/40
                 hover:shadow-md transition-shadow group"
    >
      <div className="flex items-start gap-3">
        <div className="p-2 bg-bg rounded-lg flex-shrink-0">
          <AlertTriangle className="w-4 h-4 text-muted" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-textPrimary">
            {title}
          </h3>
          <p className="text-xs text-muted mt-0.5 truncate">{description}</p>
        </div>
        <span className="text-xs text-muted flex-shrink-0">View →</span>
      </div>
    </button>
  );
}

// =============================================================================
// Insights List
// =============================================================================

interface InsightsListProps {
  insights: Insight[];
}

function InsightsList({ insights }: InsightsListProps) {
  if (insights.length === 0) {
    return (
      <div className="bg-surface rounded-xl p-8 text-center shadow-sm border border-border/40">
        <p className="text-sm text-muted">
          No insights available yet.
        </p>
      </div>
    );
  }

  const sortedInsights = [...insights].sort((a, b) => a.priority - b.priority);

  return (
    <div className="space-y-3">
      {sortedInsights.map((insight) => (
        <InsightCard key={insight.id} insight={insight} />
      ))}
    </div>
  );
}

function InsightCard({ insight }: { insight: Insight }) {
  const [expanded, setExpanded] = useState(false);

  const getIcon = (type: string) => {
    switch (type) {
      case 'spending': return '📊';
      case 'anomaly': return '⚠️';
      case 'subscription': return '🔄';
      case 'savings': return '💰';
      case 'positive': return '✓';
      default: return '•';
    }
  };

  return (
    <div className="bg-surface rounded-xl p-4 space-y-2 shadow-sm border border-border/40">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm flex-shrink-0">{getIcon(insight.type)}</span>
          <h3 className="text-sm font-medium text-textPrimary truncate">{insight.title}</h3>
        </div>
        <span className="text-[10px] px-2 py-0.5 bg-bg rounded-full text-muted flex-shrink-0">
          {Math.round(insight.confidence * 100)}%
        </span>
      </div>
      <p className="text-sm text-textSecondary leading-relaxed">{insight.description}</p>
      {insight.action && (
        <p className="text-sm text-textPrimary">→ {insight.action}</p>
      )}
      {insight.reasoning && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-muted hover:text-textSecondary transition-colors"
        >
          {expanded ? '▲ Hide' : '▼ Why this matters'}
        </button>
      )}
      {expanded && insight.reasoning && (
        <p className="text-xs text-muted pl-3 border-l-2 border-border/60 leading-relaxed">
          {insight.reasoning}
        </p>
      )}
    </div>
  );
}

// =============================================================================
// Recurring Charges Table
// =============================================================================

interface RecurringChargesTableProps {
  charges: RecurringCharge[];
  grayChargesTotal: number;
}

function RecurringChargesTable({ charges, grayChargesTotal }: RecurringChargesTableProps) {
  const formatCurrency = (v: number) => '$' + Math.abs(v).toFixed(2);

  if (charges.length === 0) {
    return (
      <div className="bg-surface rounded-xl p-8 text-center shadow-sm border border-border/40">
        <p className="text-sm text-muted">No recurring charges detected.</p>
      </div>
    );
  }

  const sortedCharges = [...charges].sort((a, b) => {
    if (a.is_gray_charge && !b.is_gray_charge) return -1;
    if (!a.is_gray_charge && b.is_gray_charge) return 1;
    return Math.abs(b.amount) - Math.abs(a.amount);
  });

  return (
    <div className="space-y-3">
      {grayChargesTotal > 0 && (
        <div className="bg-surface rounded-xl p-4 shadow-sm border border-border/40">
          <div className="flex items-center gap-2">
            <RefreshCw className="w-4 h-4 text-muted" />
            <span className="text-sm text-textPrimary">
              <strong>{charges.filter(c => c.is_gray_charge).length} gray charges</strong> totaling {formatCurrency(grayChargesTotal)}/mo
            </span>
          </div>
          <p className="text-xs text-muted mt-1">Small recurring charges you might have forgotten about</p>
        </div>
      )}

      <div className="bg-surface rounded-xl overflow-hidden shadow-sm border border-border/40">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/60">
              <th className="px-4 py-2.5 text-left text-[10px] font-medium text-muted uppercase tracking-wide">Description</th>
              <th className="px-4 py-2.5 text-left text-[10px] font-medium text-muted uppercase tracking-wide">Amount</th>
              <th className="px-4 py-2.5 text-left text-[10px] font-medium text-muted uppercase tracking-wide">Status</th>
            </tr>
          </thead>
          <tbody>
            {sortedCharges.map((charge) => (
              <tr key={charge.id} className="border-b border-border/40 last:border-b-0">
                <td className="px-4 py-2.5">
                  <span className="text-sm text-textPrimary">{charge.description}</span>
                  <span className="text-xs text-muted ml-1.5">{charge.category.icon}</span>
                </td>
                <td className="px-4 py-2.5 text-sm text-textSecondary">
                  {formatCurrency(charge.amount)}/mo
                </td>
                <td className="px-4 py-2.5">
                  {charge.is_gray_charge ? (
                    <span className="text-[10px] px-2 py-0.5 bg-accent/90 text-white rounded-full font-medium">
                      Gray
                    </span>
                  ) : (
                    <span className="text-xs text-muted">Regular</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// =============================================================================
// Anomalies List
// =============================================================================

interface AnomaliesListProps {
  anomalies: Anomaly[];
}

function AnomaliesList({ anomalies }: AnomaliesListProps) {
  const formatCurrency = (v: number) => '$' + Math.abs(v).toFixed(2);
  const formatDate = (d: string) => new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

  if (anomalies.length === 0) {
    return (
      <div className="bg-surface rounded-xl p-8 text-center shadow-sm border border-border/40">
        <p className="text-sm text-muted">No anomalies detected. Your spending looks consistent.</p>
      </div>
    );
  }

  const severityOrder = { high: 0, medium: 1, low: 2 };
  const sortedAnomalies = [...anomalies].sort(
    (a, b) => severityOrder[a.severity] - severityOrder[b.severity]
  );

  return (
    <div className="space-y-3">
      {sortedAnomalies.map((anomaly) => (
        <div key={anomaly.id} className="bg-surface rounded-xl p-4 space-y-2 shadow-sm border border-border/40">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <h3 className="text-sm font-medium text-textPrimary truncate">
                {anomaly.transaction_description}
              </h3>
              <p className="text-xs text-muted">
                {formatDate(anomaly.transaction_date)} · {anomaly.category_name}
              </p>
            </div>
            <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${anomaly.severity === 'high'
              ? 'bg-[#FFB5B5] text-[#8B0000]'
              : anomaly.severity === 'medium'
                ? 'bg-[#FFEAA7] text-[#856404]'
                : 'bg-[#B5EAD7] text-[#155724]'
              }`}>
              {anomaly.severity === 'high' ? 'HIGH' :
                anomaly.severity === 'medium' ? 'MED' : 'LOW'}
            </span>
          </div>
          <div className="text-sm">
            <span className="font-medium text-textPrimary">{formatCurrency(anomaly.actual)}</span>
            <span className="text-muted"> (typical: {formatCurrency(anomaly.expected)})</span>
          </div>
          <p className="text-xs text-muted leading-relaxed">{anomaly.explanation}</p>
        </div>
      ))}
    </div>
  );
}
