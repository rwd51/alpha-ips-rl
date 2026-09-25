import { useEffect, useState } from "react";
import { useLab } from "./LabContext";

/**
 * An instrument's own alpha, which follows the other instruments when
 * "Link alpha across instruments" is switched on.
 */
export function useLinkedAlpha(id: string, initial: number, min: number, max: number) {
  const [alpha, setAlpha] = useState(initial);
  const { linkAlpha, shared, broadcast } = useLab();
  useEffect(() => {
    if (linkAlpha && shared && shared.source !== id) setAlpha(Math.min(max, Math.max(min, shared.value)));
    // only react to new broadcasts
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shared?.n]);
  const setFromUser = (v: number) => {
    setAlpha(v);
    if (linkAlpha) broadcast(v, id);
  };
  return [alpha, setFromUser] as const;
}
