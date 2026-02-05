/**
 * InsightCard Component
 * 
 * Displays AI-generated insights with confidence badges and expandable reasoning.
 * Most important component per design system.
 */

import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import type { Insight } from '../types';

interface InsightCardProps {
  insight: Insight;
}

export function InsightCard({ insight }: InsightCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Get icon based on insight type (using text, not colored icons per design system)
  const getTypeIndicator = (type: string): string => {
    switch (type) {
      case 'spending':
        return '📊';
      case 'anomaly':
        return '[!]';
      case 'subscription':
        return '🔄';
      case 'savings':
        return '💰';
      case 'positive':
        return '✓';
      default:
        return '•';
    }
  };

  // Format confidence as percentage
  const confidencePercent = Math.round(insight.confidence * 100);

  return (
    <div className="bg-surface border border-border rounded-xl p-5 space-y-3">
      {/* Header Row */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          {/* Type Indicator + Title */}
          <div className="flex items-center space-x-2">
            <span className="text-sm text-textSecondary">
              {getTypeIndicator(insight.type)}
            </span>
            <h3 className="text-base font-medium text-textPrimary">
              {insight.title}
            </h3>
          </div>
        </div>

        {/* Confidence Badge */}
        <span className="text-xs px-2 py-1 border border-border rounded-full text-textSecondary">
          {confidencePercent}%
        </span>
      </div>

      {/* Description */}
      <p className="text-sm text-textPrimary">
        {insight.description}
      </p>

      {/* Action (if present) */}
      {insight.action && (
        <p className="text-sm text-textSecondary">
          → {insight.action}
        </p>
      )}

      {/* Expandable Reasoning */}
      {insight.reasoning && (
        <div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center text-xs text-muted hover:text-textSecondary transition-colors"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-3 h-3 mr-1" />
                Hide reasoning
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3 mr-1" />
                Why this matters
              </>
            )}
          </button>

          {isExpanded && (
            <p className="mt-2 text-xs text-textSecondary pl-4 border-l-2 border-border">
              {insight.reasoning}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Insights List Component
// =============================================================================

interface InsightsListProps {
  insights: Insight[];
}

export function InsightsList({ insights }: InsightsListProps) {
  if (insights.length === 0) {
    return (
      <div className="bg-surface border border-border rounded-xl p-8 text-center">
        <p className="text-sm text-muted">
          No insights available yet. Upload transactions to get started.
        </p>
      </div>
    );
  }

  // Sort by priority (1 = highest)
  const sortedInsights = [...insights].sort((a, b) => a.priority - b.priority);

  return (
    <div className="space-y-4">
      {sortedInsights.map((insight) => (
        <InsightCard key={insight.id} insight={insight} />
      ))}
    </div>
  );
}
