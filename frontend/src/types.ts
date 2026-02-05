/**
 * TypeScript interfaces matching backend Pydantic schemas.
 * These types ensure type safety across the frontend.
 */

// =============================================================================
// Category
// =============================================================================

export interface Category {
  id: number;
  name: string;
  icon: string | null;
  color: string | null;
  is_essential: boolean;
}

// =============================================================================
// Transaction
// =============================================================================

export interface Transaction {
  id: number;
  date: string;
  description: string;
  amount: number;
  category: Category | null;
  confidence: number | null;
  source: 'rule' | 'ai' | 'user' | 'fallback' | null;
}

// =============================================================================
// Anomaly
// =============================================================================

export interface Anomaly {
  id: number;
  transaction_id: number;
  transaction_description: string;
  transaction_amount: number;
  transaction_date: string;
  category_name: string;
  anomaly_type: 'amount' | 'frequency' | 'merchant';
  severity: 'low' | 'medium' | 'high';
  expected: number;
  actual: number;
  z_score: number;
  explanation: string;
}

// =============================================================================
// Recurring Charge
// =============================================================================

export interface RecurringCharge {
  id: number;
  description: string;
  category: Category;
  amount: number;
  frequency: string;
  frequency_days: number;
  occurrences: number;
  is_gray_charge: boolean;
  confidence: number;
}

// =============================================================================
// Insight
// =============================================================================

export type InsightType = 'spending' | 'anomaly' | 'subscription' | 'savings' | 'positive';

export interface Insight {
  id: number;
  type: InsightType;
  priority: 1 | 2 | 3;
  title: string;
  description: string;
  action: string | null;
  reasoning: string;
  confidence: number;
  data: Record<string, unknown> | null;
}

// =============================================================================
// Delta (Month-over-Month Change)
// =============================================================================

export interface Delta {
  category_id: number;
  category_name: string;
  category_icon: string | null;
  current_month: string;
  previous_month: string;
  current_amount: number;
  previous_amount: number;
  change_amount: number;
  change_percent: number;
}

// =============================================================================
// Spending Summary
// =============================================================================

export interface CategorySpending {
  amount: number;
  count: number;
  icon: string | null;
  color: string | null;
  is_essential: boolean;
}

export interface SpendingSummary {
  total_income: number;
  total_spending: number;
  net: number;
  by_category: Record<string, CategorySpending>;
}

// =============================================================================
// API Responses
// =============================================================================

export interface UploadResponse {
  session_id: string;
  filename: string;
  row_count: number;
  status: string;
}

export interface AnalyzeResponse {
  session_id: string;
  status: string;
  transactions_categorized: number;
  anomalies_detected: number;
  recurring_charges_found: number;
  insights_generated: number;
}

export interface DashboardResponse {
  session_id: string;
  status: string;
  summary: SpendingSummary;
  insights: Insight[];
  anomalies: Anomaly[];
  recurring_charges: RecurringCharge[];
  deltas: Delta[];
}

// =============================================================================
// Goal
// =============================================================================

export interface GoalCut {
  category: string;
  category_icon: string | null;
  current_amount: number;
  suggested_amount: number;
  savings: number;
  difficulty: 'easy' | 'moderate' | 'hard';
  is_essential: boolean;
}

export interface GoalResponse {
  target_amount: number;
  current_discretionary: number;
  achievable: boolean;
  suggested_cuts: GoalCut[];
  total_potential_savings: number;
  gap_amount: number | null;
  ai_advice: string;
}

// =============================================================================
// App State
// =============================================================================

export type AppView = 'upload' | 'analyzing' | 'dashboard';

export interface AppState {
  view: AppView;
  sessionId: string | null;
  isLoading: boolean;
  error: string | null;
}

// =============================================================================
// Chat
// =============================================================================

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
}

export interface ChatResponse {
  conversation_id: string;
  message: string;
  suggested_prompts: string[];
}

export interface SuggestedPromptsResponse {
  prompts: string[];
}

export interface ConversationHistory {
  conversation_id: string;
  messages: ChatMessage[];
  created_at?: string;
}

// =============================================================================
// User & Sessions
// =============================================================================

export interface Session {
  id: string;
  name: string | null;
  filename: string | null;
  row_count: number | null;
  status: string;
  is_sample: boolean;
  created_at: string;
}

export interface SessionListResponse {
  sessions: Session[];
  active_session_id: string | null;
}

export interface UserSession {
  user_id: string;
  sessions: Session[];
  has_sample_session: boolean;
  active_session_id: string | null;
}

// =============================================================================
// Fortune Cookie
// =============================================================================

export type FortuneSentiment = 'positive' | 'neutral' | 'warning';

export interface FortuneResponse {
  fortune: string;
  sentiment: FortuneSentiment;
  lucky_number: string | null;
}
