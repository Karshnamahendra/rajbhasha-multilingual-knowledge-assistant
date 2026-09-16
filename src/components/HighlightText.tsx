import React from 'react';
import { HighlightSpan } from '../types';

interface HighlightTextProps {
  text: string;
  spans?: HighlightSpan[];
  highlight?: string;
  searchQuery?: string;
  className?: string;
  highlightClassName?: string;
}

export const HighlightText: React.FC<HighlightTextProps> = ({
  text,
  spans,
  highlight,
  searchQuery,
  className = '',
  highlightClassName = 'bg-amber-400/30 text-amber-200 border-b-2 border-amber-400 font-semibold px-1 py-0.5 rounded transition-all duration-300 shadow-sm',
}) => {
  if (!text) return null;

  // 1. If explicit spans are provided and valid, render exact character slice marks
  if (spans && spans.length > 0) {
    const validSpans = spans
      .filter((s) => s.start >= 0 && s.end <= text.length && s.start < s.end)
      .sort((a, b) => a.start - b.start);

    if (validSpans.length > 0) {
      const elements: React.ReactNode[] = [];
      let currentIndex = 0;

      validSpans.forEach((span, idx) => {
        if (span.start > currentIndex) {
          elements.push(
            <React.Fragment key={`text-${idx}`}>
              {text.slice(currentIndex, span.start)}
            </React.Fragment>
          );
        }

        const markText = text.slice(span.start, span.end);
        elements.push(
          <mark
            key={`mark-${idx}`}
            className={highlightClassName}
          >
            {markText}
          </mark>
        );

        currentIndex = span.end;
      });

      if (currentIndex < text.length) {
        elements.push(
          <React.Fragment key="text-end">
            {text.slice(currentIndex)}
          </React.Fragment>
        );
      }

      return <span className={className}>{elements}</span>;
    }
  }

  // 2. Multi-phrase substring search fallback
  const targetHighlight = highlight?.trim();
  const targetSearch = searchQuery?.trim();

  if (!targetHighlight && !targetSearch) {
    return <span className={className}>{text}</span>;
  }

  const rawTerms: string[] = [];
  if (targetHighlight) {
    // Split multi-entity answers (commas, newlines, semicolons)
    const splitHighlight = targetHighlight.split(/[\n,;•–\-]|(?:\s+एवं\s+)/);
    splitHighlight.forEach((s) => {
      const clean = s.trim();
      if (clean.length >= 2) rawTerms.push(clean);
    });
    if (targetHighlight.length >= 2) rawTerms.push(targetHighlight);
  }

  if (targetSearch && targetSearch.length >= 3) {
    rawTerms.push(targetSearch);
  }

  // Clean up terms and escape for regex
  const escapedTerms = Array.from(new Set(rawTerms))
    .map((term) => term.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&'))
    .filter((t) => t.length >= 2);

  if (escapedTerms.length === 0) {
    return <span className={className}>{text}</span>;
  }

  try {
    const regex = new RegExp(`(${escapedTerms.join('|')})`, 'gi');
    const parts = text.split(regex);

    return (
      <span className={className}>
        {parts.map((part, i) => {
          const isMatch = escapedTerms.some((term) =>
            new RegExp(`^${term}$`, 'i').test(part)
          );
          if (isMatch) {
            return (
              <mark
                key={i}
                className={highlightClassName}
              >
                {part}
              </mark>
            );
          }
          return <React.Fragment key={i}>{part}</React.Fragment>;
        })}
      </span>
    );
  } catch {
    return <span className={className}>{text}</span>;
  }
};

