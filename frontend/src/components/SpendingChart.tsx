/**
 * SpendingChart Component
 * 
 * Pastel-colored pie chart for spending visualization.
 * Clean, minimal design with soft colors for easy readability.
 */

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import type { SpendingSummary } from '../types';

interface SpendingChartProps {
  summary: SpendingSummary;
}

// Pastel color palette for category visualization
const PASTEL_COLORS = [
  '#A8D8EA', // Soft blue
  '#FFB5B5', // Soft coral/pink
  '#B5EAD7', // Soft mint green
  '#FFEAA7', // Soft yellow
  '#DDA0DD', // Soft plum
  '#F7DC6F', // Soft gold
  '#AED6F1', // Soft sky blue
  '#F5B7B1', // Soft salmon
  '#D5DBDB', // Soft gray
  '#ABEBC6', // Soft lime
  '#FAD7A0', // Soft peach
  '#D2B4DE', // Soft lavender
];

export function SpendingChart({ summary }: SpendingChartProps) {
  // Prepare chart data (expenses only, absolute values)
  const chartData = Object.entries(summary.by_category)
    .filter(([_, data]) => data.amount < 0) // Only expenses
    .map(([name, data]) => ({
      name,
      value: Math.abs(data.amount),
      icon: data.icon,
    }))
    .sort((a, b) => b.value - a.value); // Sort by value descending

  // Format currency
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  // Custom tooltip (minimal, per design system)
  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: { name: string; value: number; icon: string } }> }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-surface rounded-lg px-3 py-2 shadow-md border border-border/40">
          <p className="text-xs font-medium text-textPrimary">
            {data.icon} {data.name}
          </p>
          <p className="text-xs text-muted">
            {formatCurrency(data.value)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-surface rounded-xl p-4 shadow-sm border border-border/40">
      {/* Section Header */}
      <h2 className="text-sm font-semibold text-textPrimary mb-3">
        Spending by Category
      </h2>

      {/* Chart Container */}
      <div className="h-48 sm:h-56">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
              animationDuration={500}
              animationBegin={0}
            >
              {chartData.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={PASTEL_COLORS[index % PASTEL_COLORS.length]}
                  stroke="#FFFFFF"
                  strokeWidth={2}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Legend (inline labels per design system) */}
      <div className="mt-3 space-y-1.5">
        {chartData.slice(0, 5).map((item, index) => (
          <div key={item.name} className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div
                className="w-2.5 h-2.5 rounded-full flex-shrink-0 border border-white/50"
                style={{ backgroundColor: PASTEL_COLORS[index % PASTEL_COLORS.length] }}
              />
              <span className="text-xs text-textPrimary truncate">
                {item.icon} {item.name}
              </span>
            </div>
            <span className="text-xs text-muted flex-shrink-0 ml-2">
              {formatCurrency(item.value)}
            </span>
          </div>
        ))}
        {chartData.length > 5 && (
          <p className="text-[10px] text-muted pt-1">
            +{chartData.length - 5} more categories
          </p>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Summary Stats Component
// =============================================================================

interface SummaryStatsProps {
  summary: SpendingSummary;
}

export function SummaryStats({ summary }: SummaryStatsProps) {
  const formatCurrency = (value: number): string => {
    const prefix = value >= 0 ? '+' : '';
    return prefix + new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  return (
    <div className="bg-surface border border-border rounded-xl p-5">
      <h2 className="text-lg font-semibold text-textPrimary mb-4">
        Overview
      </h2>

      <div className="space-y-3">
        {/* Income */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-textSecondary">Income</span>
          <span className="text-sm font-medium text-textPrimary">
            {formatCurrency(summary.total_income)}
          </span>
        </div>

        {/* Spending */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-textSecondary">Spending</span>
          <span className="text-sm font-medium text-textPrimary">
            {formatCurrency(summary.total_spending)}
          </span>
        </div>

        {/* Divider */}
        <div className="border-t border-border" />

        {/* Net */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-textPrimary">Net</span>
          <span className={`text-sm font-medium ${summary.net >= 0 ? 'text-textPrimary' : 'text-textSecondary'}`}>
            {formatCurrency(summary.net)}
          </span>
        </div>
      </div>
    </div>
  );
}
