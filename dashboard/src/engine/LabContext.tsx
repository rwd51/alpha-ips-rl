import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

interface SharedAlpha {
  value: number;
  source: string;
  n: number;
}

interface LabState {
  /** "Pause all" in the top bar. */
  paused: boolean;
  setPaused(v: boolean): void;
  /** "Link alpha across instruments". */
  linkAlpha: boolean;
  setLinkAlpha(v: boolean): void;
  shared: SharedAlpha | null;
  broadcast(value: number, source: string): void;
}

const LabCtx = createContext<LabState | null>(null);

export function LabProvider({ children }: { children: ReactNode }) {
  const [paused, setPaused] = useState(false);
  const [linkAlpha, setLinkAlpha] = useState(false);
  const [shared, setShared] = useState<SharedAlpha | null>(null);
  const broadcast = useCallback((value: number, source: string) => {
    setShared((prev) => ({ value, source, n: (prev?.n ?? 0) + 1 }));
  }, []);
  const value = useMemo(
    () => ({ paused, setPaused, linkAlpha, setLinkAlpha, shared, broadcast }),
    [paused, linkAlpha, shared, broadcast],
  );
  return <LabCtx.Provider value={value}>{children}</LabCtx.Provider>;
}

export function useLab(): LabState {
  const ctx = useContext(LabCtx);
  if (!ctx) throw new Error("useLab must be used inside <LabProvider>");
  return ctx;
}
