/** Format as Indian Rupees: Rs.3,82,260 */
export function inr(n: number): string {
  if (n == null || isNaN(n)) return 'Rs.0';
  const abs = Math.abs(n);
  const sign = n < 0 ? '-' : '';
  if (abs >= 10000000) return `${sign}Rs.${(abs / 10000000).toFixed(1)}Cr`;
  if (abs >= 100000) return `${sign}Rs.${(abs / 100000).toFixed(1)}L`;
  return `${sign}Rs.${abs.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}

/** Format as Indian Rupees with full number */
export function inrFull(n: number): string {
  if (n == null || isNaN(n)) return 'Rs.0';
  return `Rs.${Math.round(n).toLocaleString('en-IN')}`;
}

/** Format percentage */
export function pct(n: number, decimals = 1): string {
  if (n == null || isNaN(n)) return '0%';
  return `${(n * 100).toFixed(decimals)}%`;
}

/** Format percentage from already-percent value */
export function pctRaw(n: number, decimals = 1): string {
  if (n == null || isNaN(n)) return '0%';
  return `${n.toFixed(decimals)}%`;
}

/** Action display name */
export function actionLabel(action: string): string {
  const map: Record<string, string> = {
    do_nothing: 'Do Nothing', retry: 'Retry', reminder: 'Reminder',
    alternative_method: 'Alt Method', control: 'Do Nothing',
  };
  return map[action] || action.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/** Failure reason display name */
export function failureLabel(reason: string): string {
  return reason.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
