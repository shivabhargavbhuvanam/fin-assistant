/**
 * Financial Fortune Cookie Component
 * 
 * An animated fortune cookie that "cracks open" to reveal cryptic
 * but actionable financial advice based on user's spending patterns.
 * 
 * Features:
 * - Animated cookie crack with rotation
 * - Fortune paper slide-out effect
 * - Typewriter text reveal
 * - Confetti particles on open
 * - Color-coded by sentiment (gold/orange/red)
 * - "Crack Another Cookie" to regenerate
 * 
 * Aesthetic: Retro/playful with Macintosh 1984 vibes
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Sparkles, RefreshCw } from 'lucide-react';
import { getFortune } from '../api/client';
import './FinancialFortuneCookie.css';

interface FortuneData {
  fortune: string;
  sentiment: 'positive' | 'neutral' | 'warning';
  lucky_number: string | null;
}

interface FinancialFortuneCookieProps {
  sessionId: string;
}

// =============================================================================
// Confetti Particle Component
// =============================================================================

interface Particle {
  id: number;
  x: number;
  y: number;
  rotation: number;
  color: string;
  delay: number;
}

function Confetti({ show }: { show: boolean }) {
  const [particles, setParticles] = useState<Particle[]>([]);

  useEffect(() => {
    if (show) {
      // Generate 12 particles with random properties
      const colors = ['#FFD700', '#FFA500', '#FF6347', '#FFB6C1', '#87CEEB', '#98FB98'];
      const newParticles: Particle[] = Array.from({ length: 12 }, (_, i) => ({
        id: i,
        x: Math.random() * 100 - 50, // -50 to 50
        y: Math.random() * -100 - 50, // -50 to -150
        rotation: Math.random() * 720 - 360,
        color: colors[Math.floor(Math.random() * colors.length)],
        delay: Math.random() * 0.3,
      }));
      setParticles(newParticles);

      // Clear particles after animation
      const timer = setTimeout(() => setParticles([]), 1500);
      return () => clearTimeout(timer);
    }
  }, [show]);

  if (!show || particles.length === 0) return null;

  return (
    <div className="confetti-container">
      {particles.map((p) => (
        <div
          key={p.id}
          className="confetti-particle"
          style={{
            '--x': `${p.x}px`,
            '--y': `${p.y}px`,
            '--rotation': `${p.rotation}deg`,
            '--delay': `${p.delay}s`,
            backgroundColor: p.color,
          } as React.CSSProperties}
        />
      ))}
    </div>
  );
}

// =============================================================================
// Typewriter Text Component
// =============================================================================

function TypewriterText({ text, onComplete }: { text: string; onComplete?: () => void }) {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    // Reset when text changes
    setDisplayText('');
    setCurrentIndex(0);
  }, [text]);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, 45); // 45ms per character for smooth typing

      return () => clearTimeout(timer);
    } else if (currentIndex === text.length && onComplete) {
      onComplete();
    }
  }, [currentIndex, text, onComplete]);

  return (
    <span className="typewriter-text">
      {displayText}
      {currentIndex < text.length && <span className="typewriter-cursor">|</span>}
    </span>
  );
}

// =============================================================================
// Main Fortune Cookie Component
// =============================================================================

export function FinancialFortuneCookie({ sessionId }: FinancialFortuneCookieProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fortune, setFortune] = useState<FortuneData | null>(null);
  const [showConfetti, setShowConfetti] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFortuneText, setShowFortuneText] = useState(false);
  const cookieRef = useRef<HTMLButtonElement>(null);

  // Cache fortune for session
  const hasFetchedRef = useRef(false);

  const fetchFortune = useCallback(async () => {
    if (isLoading) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await getFortune(sessionId);
      setFortune(data);
    } catch (err) {
      console.error('Fortune fetch error:', err);
      setError('The spirits are unclear...');
      // Fallback fortune
      setFortune({
        fortune: 'Patience reveals the path to savings',
        sentiment: 'neutral',
        lucky_number: null,
      });
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, isLoading]);

  // Toggle cookie open/closed on click
  const handleCookieClick = useCallback(async () => {
    if (isOpen) {
      // Close the cookie
      setIsOpen(false);
      setShowConfetti(false);
      setShowFortuneText(false);
      return;
    }
    
    // Opening the cookie
    // Fetch fortune if not already loaded
    if (!fortune && !hasFetchedRef.current) {
      hasFetchedRef.current = true;
      await fetchFortune();
    }
    
    // Trigger open animation
    setIsOpen(true);
    
    // Show confetti after crack animation (only on first open or after regenerate)
    if (!fortune || !hasFetchedRef.current) {
      setTimeout(() => {
        setShowConfetti(true);
      }, 400);
    }
    
    // Show fortune text after paper slides out
    setTimeout(() => {
      setShowFortuneText(true);
    }, 800);
  }, [isOpen, fortune, fetchFortune]);

  // Handle "Crack Another" button
  const handleCrackAnother = useCallback(async () => {
    // Reset states
    setIsOpen(false);
    setShowConfetti(false);
    setShowFortuneText(false);
    setFortune(null);
    
    // Small delay before fetching new fortune
    setTimeout(async () => {
      hasFetchedRef.current = true;
      await fetchFortune();
    }, 300);
  }, [fetchFortune]);

  // Determine cookie color based on sentiment
  const getCookieClass = () => {
    if (!fortune) return 'cookie-neutral';
    switch (fortune.sentiment) {
      case 'positive':
        return 'cookie-positive';
      case 'warning':
        return 'cookie-warning';
      default:
        return 'cookie-neutral';
    }
  };

  return (
    <div className="fortune-cookie-wrapper">
      {/* The Cookie */}
      <button
        ref={cookieRef}
        className={`fortune-cookie ${isOpen ? 'fortune-cookie--open' : ''} ${getCookieClass()}`}
        onClick={handleCookieClick}
        disabled={isLoading}
        title={isOpen ? "Click to close" : "Click for your fortune"}
      >
        {isLoading ? (
          <div className="cookie-loading">
            <RefreshCw className="w-4 h-4 animate-spin" />
          </div>
        ) : (
          <>
            {/* Cookie halves */}
            <div className="cookie-half cookie-half--left">
              <span className="cookie-emoji">🥠</span>
            </div>
            <div className="cookie-half cookie-half--right">
              <span className="cookie-emoji cookie-emoji--flipped">🥠</span>
            </div>
            
            {/* Sparkle indicator when closed */}
            {!isOpen && (
              <div className="cookie-sparkle">
                <Sparkles className="w-3 h-3" />
              </div>
            )}
          </>
        )}
      </button>

      {/* Confetti */}
      <Confetti show={showConfetti} />

      {/* Fortune Paper (slides out when open) */}
      {isOpen && fortune && (
        <div className={`fortune-paper ${showFortuneText ? 'fortune-paper--visible' : ''}`}>
          <div className="fortune-paper-inner">
            {/* Fortune Text */}
            <div className="fortune-text">
              {showFortuneText && (
                <TypewriterText text={`"${fortune.fortune}"`} />
              )}
            </div>

            {/* Lucky Number */}
            {fortune.lucky_number && showFortuneText && (
              <div className="fortune-lucky-number">
                <span className="lucky-label">Lucky Number:</span>
                <span className="lucky-value">{fortune.lucky_number}</span>
              </div>
            )}

            {/* Crack Another Button */}
            {showFortuneText && (
              <button
                className="crack-another-btn"
                onClick={handleCrackAnother}
                disabled={isLoading}
              >
                <RefreshCw className="w-3 h-3" />
                <span>Crack Another</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Error state (hidden in fortune) */}
      {error && !fortune && (
        <div className="fortune-error">
          {error}
        </div>
      )}
    </div>
  );
}

export default FinancialFortuneCookie;
