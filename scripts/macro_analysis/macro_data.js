/**
 * Macro Economic Data for Vietnam F&B Industry Analysis
 * Source: SBV, VGCA, SJC, PetroVietnam, GSO, Internal DB
 * Deposit Rate: Vietcombank - Lãi suất tiết kiệm kỳ hạn 12 tháng (tại quầy)
 *   Ref: vietcombank.com.vn, CafeF, Lao Dong, DNSE, webgia.com
 * Period: Jan 2022 - Mar 2026 (51 months)
 * Guest data: v_fact_db_payment_hub_transactions + v_fact_db_rk_dc_transactions
 */

const MACRO_DATA = [
  // 2022
  { year: 2022, month: 1,  label: "T01/2022", gold_sjc: 62.0, gas_ron95: 23200, deposit_rate: 5.50, lending_rate: 9.20, cpi: 1.94, gdp_growth: 5.03, tourism_intl: 0.09 },
  { year: 2022, month: 2,  label: "T02/2022", gold_sjc: 63.5, gas_ron95: 24500, deposit_rate: 5.50, lending_rate: 9.20, cpi: 1.42, gdp_growth: 5.03, tourism_intl: 0.12 },  // VCB 12T: 5.50%
  { year: 2022, month: 3,  label: "T03/2022", gold_sjc: 68.0, gas_ron95: 28000, deposit_rate: 5.50, lending_rate: 9.10, cpi: 2.41, gdp_growth: 5.03, tourism_intl: 0.15 },
  { year: 2022, month: 4,  label: "T04/2022", gold_sjc: 69.5, gas_ron95: 29500, deposit_rate: 5.40, lending_rate: 9.00, cpi: 2.64, gdp_growth: 7.72, tourism_intl: 0.18 },
  { year: 2022, month: 5,  label: "T05/2022", gold_sjc: 69.0, gas_ron95: 30000, deposit_rate: 5.30, lending_rate: 8.90, cpi: 2.86, gdp_growth: 7.72, tourism_intl: 0.22 },
  { year: 2022, month: 6,  label: "T06/2022", gold_sjc: 67.5, gas_ron95: 31500, deposit_rate: 5.30, lending_rate: 8.80, cpi: 3.37, gdp_growth: 7.72, tourism_intl: 0.28 },
  { year: 2022, month: 7,  label: "T07/2022", gold_sjc: 66.0, gas_ron95: 30000, deposit_rate: 5.50, lending_rate: 8.70, cpi: 3.14, gdp_growth: 13.67, tourism_intl: 0.35 },
  { year: 2022, month: 8,  label: "T08/2022", gold_sjc: 66.5, gas_ron95: 28500, deposit_rate: 5.50, lending_rate: 8.70, cpi: 2.89, gdp_growth: 13.67, tourism_intl: 0.42 },  // VCB stable pre-hike
  { year: 2022, month: 9,  label: "T09/2022", gold_sjc: 66.0, gas_ron95: 27000, deposit_rate: 6.40, lending_rate: 8.80, cpi: 3.94, gdp_growth: 13.67, tourism_intl: 0.55 },  // VCB bắt đầu tăng mạnh
  { year: 2022, month: 10, label: "T10/2022", gold_sjc: 66.5, gas_ron95: 27500, deposit_rate: 6.80, lending_rate: 8.90, cpi: 4.30, gdp_growth: 5.92, tourism_intl: 0.68 },
  { year: 2022, month: 11, label: "T11/2022", gold_sjc: 67.0, gas_ron95: 26500, deposit_rate: 7.10, lending_rate: 9.30, cpi: 4.37, gdp_growth: 5.92, tourism_intl: 0.78 },
  { year: 2022, month: 12, label: "T12/2022", gold_sjc: 67.5, gas_ron95: 25000, deposit_rate: 7.40, lending_rate: 9.50, cpi: 4.55, gdp_growth: 5.92, tourism_intl: 0.85 },  // VCB đỉnh cuối 2022
  // 2023
  { year: 2023, month: 1,  label: "T01/2023", gold_sjc: 67.0, gas_ron95: 24000, deposit_rate: 7.40, lending_rate: 9.60, cpi: 4.89, gdp_growth: 3.32, tourism_intl: 0.72 },  // VCB đỉnh chu kỳ
  { year: 2023, month: 2,  label: "T02/2023", gold_sjc: 67.5, gas_ron95: 23500, deposit_rate: 7.40, lending_rate: 9.50, cpi: 4.31, gdp_growth: 3.32, tourism_intl: 0.90 },
  { year: 2023, month: 3,  label: "T03/2023", gold_sjc: 68.0, gas_ron95: 23000, deposit_rate: 7.40, lending_rate: 9.40, cpi: 3.35, gdp_growth: 3.32, tourism_intl: 0.95 },  // VCB vẫn giữ đỉnh Q1
  { year: 2023, month: 4,  label: "T04/2023", gold_sjc: 68.5, gas_ron95: 22500, deposit_rate: 6.80, lending_rate: 9.20, cpi: 2.81, gdp_growth: 4.14, tourism_intl: 0.98 },  // VCB bắt đầu giảm
  { year: 2023, month: 5,  label: "T05/2023", gold_sjc: 67.5, gas_ron95: 21500, deposit_rate: 6.50, lending_rate: 9.00, cpi: 2.43, gdp_growth: 4.14, tourism_intl: 1.02 },
  { year: 2023, month: 6,  label: "T06/2023", gold_sjc: 67.0, gas_ron95: 21000, deposit_rate: 6.30, lending_rate: 8.80, cpi: 2.00, gdp_growth: 4.14, tourism_intl: 1.08 },
  { year: 2023, month: 7,  label: "T07/2023", gold_sjc: 67.5, gas_ron95: 22000, deposit_rate: 5.80, lending_rate: 8.60, cpi: 2.06, gdp_growth: 5.33, tourism_intl: 1.15 },
  { year: 2023, month: 8,  label: "T08/2023", gold_sjc: 68.0, gas_ron95: 23000, deposit_rate: 5.50, lending_rate: 8.50, cpi: 2.96, gdp_growth: 5.33, tourism_intl: 1.22 },
  { year: 2023, month: 9,  label: "T09/2023", gold_sjc: 69.0, gas_ron95: 24000, deposit_rate: 5.30, lending_rate: 8.50, cpi: 3.66, gdp_growth: 5.33, tourism_intl: 1.25 },
  { year: 2023, month: 10, label: "T10/2023", gold_sjc: 70.5, gas_ron95: 23500, deposit_rate: 5.10, lending_rate: 8.50, cpi: 3.59, gdp_growth: 6.72, tourism_intl: 1.30 },
  { year: 2023, month: 11, label: "T11/2023", gold_sjc: 71.0, gas_ron95: 22500, deposit_rate: 4.90, lending_rate: 8.50, cpi: 3.45, gdp_growth: 6.72, tourism_intl: 1.22 },
  { year: 2023, month: 12, label: "T12/2023", gold_sjc: 73.0, gas_ron95: 22000, deposit_rate: 4.80, lending_rate: 8.50, cpi: 3.58, gdp_growth: 6.72, tourism_intl: 1.35 },  // VCB kết thúc đà giảm mạnh 2023
  // 2024
  { year: 2024, month: 1,  label: "T01/2024", gold_sjc: 74.2, gas_ron95: 21900, deposit_rate: 4.70, lending_rate: 8.50, cpi: 3.37, gdp_growth: 5.72, tourism_intl: 1.28 },
  { year: 2024, month: 2,  label: "T02/2024", gold_sjc: 76.5, gas_ron95: 22300, deposit_rate: 4.70, lending_rate: 8.40, cpi: 3.51, gdp_growth: 5.72, tourism_intl: 1.35 },
  { year: 2024, month: 3,  label: "T03/2024", gold_sjc: 80.0, gas_ron95: 22800, deposit_rate: 4.60, lending_rate: 8.30, cpi: 3.97, gdp_growth: 5.66, tourism_intl: 1.42 },  // VCB chạm đáy vùng 4.6%
  { year: 2024, month: 4,  label: "T04/2024", gold_sjc: 84.5, gas_ron95: 23500, deposit_rate: 4.60, lending_rate: 8.10, cpi: 4.40, gdp_growth: 5.80, tourism_intl: 1.55 },
  { year: 2024, month: 5,  label: "T05/2024", gold_sjc: 90.0, gas_ron95: 23000, deposit_rate: 4.60, lending_rate: 7.90, cpi: 4.44, gdp_growth: 5.90, tourism_intl: 1.60 },
  { year: 2024, month: 6,  label: "T06/2024", gold_sjc: 85.2, gas_ron95: 22500, deposit_rate: 4.60, lending_rate: 7.70, cpi: 4.34, gdp_growth: 6.93, tourism_intl: 1.48 },
  { year: 2024, month: 7,  label: "T07/2024", gold_sjc: 82.0, gas_ron95: 22000, deposit_rate: 4.60, lending_rate: 7.50, cpi: 4.36, gdp_growth: 6.50, tourism_intl: 1.65 },
  { year: 2024, month: 8,  label: "T08/2024", gold_sjc: 81.0, gas_ron95: 21500, deposit_rate: 4.60, lending_rate: 7.30, cpi: 3.45, gdp_growth: 6.80, tourism_intl: 1.70 },
  { year: 2024, month: 9,  label: "T09/2024", gold_sjc: 84.0, gas_ron95: 21800, deposit_rate: 4.60, lending_rate: 7.20, cpi: 2.63, gdp_growth: 7.40, tourism_intl: 1.80 },
  { year: 2024, month: 10, label: "T10/2024", gold_sjc: 87.5, gas_ron95: 22000, deposit_rate: 4.70, lending_rate: 7.10, cpi: 2.89, gdp_growth: 7.40, tourism_intl: 1.85 },
  { year: 2024, month: 11, label: "T11/2024", gold_sjc: 92.0, gas_ron95: 22500, deposit_rate: 4.70, lending_rate: 7.00, cpi: 2.77, gdp_growth: 7.40, tourism_intl: 1.75 },
  { year: 2024, month: 12, label: "T12/2024", gold_sjc: 95.0, gas_ron95: 23000, deposit_rate: 4.70, lending_rate: 6.90, cpi: 2.94, gdp_growth: 7.55, tourism_intl: 1.90 },  // VCB ổn định ~4.7% cả năm 2024
  // 2025
  { year: 2025, month: 1,  label: "T01/2025", gold_sjc: 100.0, gas_ron95: 23500, deposit_rate: 4.70, lending_rate: 6.90, cpi: 3.10, gdp_growth: 6.80, tourism_intl: 1.50 },
  { year: 2025, month: 2,  label: "T02/2025", gold_sjc: 105.0, gas_ron95: 24000, deposit_rate: 4.70, lending_rate: 6.80, cpi: 4.06, gdp_growth: 6.80, tourism_intl: 1.65 },
  { year: 2025, month: 3,  label: "T03/2025", gold_sjc: 108.0, gas_ron95: 24500, deposit_rate: 4.70, lending_rate: 6.80, cpi: 2.80, gdp_growth: 6.93, tourism_intl: 1.72 },
  { year: 2025, month: 4,  label: "T04/2025", gold_sjc: 112.0, gas_ron95: 25000, deposit_rate: 4.60, lending_rate: 6.80, cpi: 3.20, gdp_growth: 6.90, tourism_intl: 1.80 },
  { year: 2025, month: 5,  label: "T05/2025", gold_sjc: 118.0, gas_ron95: 25500, deposit_rate: 4.60, lending_rate: 6.70, cpi: 3.40, gdp_growth: 7.00, tourism_intl: 1.85 },
  { year: 2025, month: 6,  label: "T06/2025", gold_sjc: 125.0, gas_ron95: 26000, deposit_rate: 4.60, lending_rate: 6.70, cpi: 3.15, gdp_growth: 7.20, tourism_intl: 1.70 },
  { year: 2025, month: 7,  label: "T07/2025", gold_sjc: 130.0, gas_ron95: 26500, deposit_rate: 4.60, lending_rate: 6.70, cpi: 3.30, gdp_growth: 7.10, tourism_intl: 1.90 },
  { year: 2025, month: 8,  label: "T08/2025", gold_sjc: 135.0, gas_ron95: 27000, deposit_rate: 4.70, lending_rate: 6.70, cpi: 3.10, gdp_growth: 7.30, tourism_intl: 1.95 },
  { year: 2025, month: 9,  label: "T09/2025", gold_sjc: 140.0, gas_ron95: 27500, deposit_rate: 4.70, lending_rate: 6.60, cpi: 2.90, gdp_growth: 7.50, tourism_intl: 2.00 },
  { year: 2025, month: 10, label: "T10/2025", gold_sjc: 148.0, gas_ron95: 28000, deposit_rate: 4.70, lending_rate: 6.60, cpi: 3.05, gdp_growth: 7.50, tourism_intl: 2.10 },
  { year: 2025, month: 11, label: "T11/2025", gold_sjc: 155.0, gas_ron95: 28500, deposit_rate: 4.70, lending_rate: 6.60, cpi: 2.95, gdp_growth: 7.60, tourism_intl: 2.05 },
  { year: 2025, month: 12, label: "T12/2025", gold_sjc: 160.0, gas_ron95: 29000, deposit_rate: 4.80, lending_rate: 6.60, cpi: 3.20, gdp_growth: 7.70, tourism_intl: 2.20 },  // VCB nhích nhẹ cuối 2025
  // 2026
  { year: 2026, month: 1,  label: "T01/2026", gold_sjc: 165.0, gas_ron95: 29500, deposit_rate: 5.20, lending_rate: 6.60, cpi: 3.35, gdp_growth: 7.10, tourism_intl: 1.80 },  // VCB bắt đầu tăng
  { year: 2026, month: 2,  label: "T02/2026", gold_sjc: 170.0, gas_ron95: 30000, deposit_rate: 5.50, lending_rate: 6.60, cpi: 4.20, gdp_growth: 7.10, tourism_intl: 2.00 },
  { year: 2026, month: 3,  label: "T03/2026", gold_sjc: 175.0, gas_ron95: 30000, deposit_rate: 5.90, lending_rate: 6.60, cpi: 3.10, gdp_growth: 7.20, tourism_intl: 2.10 },  // VCB tăng mạnh Q1/2026
];

// Guest data from DB (v_fact_db_payment_hub_transactions + v_fact_db_rk_dc_transactions)
// Data pulled: 2022-01 → 2026-03 (51 months)
const GUEST_DATA = [
  // 2022 — Post-COVID recovery / payment_hub transition period
  { year: 2022, month: 1,  label: "T01/2022", total_guests: 555363,  total_revenue: 179575950303 },
  { year: 2022, month: 2,  label: "T02/2022", total_guests: 612332,  total_revenue: 187648762010 },
  { year: 2022, month: 3,  label: "T03/2022", total_guests: 136533,  total_revenue: 183022292935 },
  { year: 2022, month: 4,  label: "T04/2022", total_guests: 12597,   total_revenue: 234029524096 },
  { year: 2022, month: 5,  label: "T05/2022", total_guests: 15716,   total_revenue: 244443950079 },
  { year: 2022, month: 6,  label: "T06/2022", total_guests: 16622,   total_revenue: 243113247216 },
  { year: 2022, month: 7,  label: "T07/2022", total_guests: 19121,   total_revenue: 270834243388 },
  { year: 2022, month: 8,  label: "T08/2022", total_guests: 18123,   total_revenue: 254112648342 },
  { year: 2022, month: 9,  label: "T09/2022", total_guests: 16514,   total_revenue: 237812951315 },
  { year: 2022, month: 10, label: "T10/2022", total_guests: 14805,   total_revenue: 251292768704 },
  { year: 2022, month: 11, label: "T11/2022", total_guests: 11849,   total_revenue: 283581522198 },
  { year: 2022, month: 12, label: "T12/2022", total_guests: 14786,   total_revenue: 631521239482 },
  // 2023 — Full stabilization
  { year: 2023, month: 1,  label: "T01/2023", total_guests: 1344938, total_revenue: 3042510670741 },
  { year: 2023, month: 2,  label: "T02/2023", total_guests: 1281902, total_revenue: 527981163947 },
  { year: 2023, month: 3,  label: "T03/2023", total_guests: 1315526, total_revenue: 540827816659 },
  { year: 2023, month: 4,  label: "T04/2023", total_guests: 1230973, total_revenue: 500529345748 },
  { year: 2023, month: 5,  label: "T05/2023", total_guests: 1857575, total_revenue: 551721930568 },
  { year: 2023, month: 6,  label: "T06/2023", total_guests: 1439638, total_revenue: 564331085343 },
  { year: 2023, month: 7,  label: "T07/2023", total_guests: 1556711, total_revenue: 569502852832 },
  { year: 2023, month: 8,  label: "T08/2023", total_guests: 1547959, total_revenue: 559134447820 },
  { year: 2023, month: 9,  label: "T09/2023", total_guests: 1545015, total_revenue: 572812938592 },
  { year: 2023, month: 10, label: "T10/2023", total_guests: 1532572, total_revenue: 566466309753 },
  { year: 2023, month: 11, label: "T11/2023", total_guests: 1366088, total_revenue: 499710667278 },
  { year: 2023, month: 12, label: "T12/2023", total_guests: 1582233, total_revenue: 593892588592 },
  // 2024 — Growth phase (mid-year expansion)
  { year: 2024, month: 1,  label: "T01/2024", total_guests: 1451037, total_revenue: 550789899062 },
  { year: 2024, month: 2,  label: "T02/2024", total_guests: 1598325, total_revenue: 612208861119 },
  { year: 2024, month: 3,  label: "T03/2024", total_guests: 1506576, total_revenue: 555654084573 },
  { year: 2024, month: 4,  label: "T04/2024", total_guests: 1489458, total_revenue: 548139120376 },
  { year: 2024, month: 5,  label: "T05/2024", total_guests: 1606125, total_revenue: 577532118535 },
  { year: 2024, month: 6,  label: "T06/2024", total_guests: 3597964, total_revenue: 1243776397585 },
  { year: 2024, month: 7,  label: "T07/2024", total_guests: 3443981, total_revenue: 1177088692920 },
  { year: 2024, month: 8,  label: "T08/2024", total_guests: 3516903, total_revenue: 1210259844013 },
  { year: 2024, month: 9,  label: "T09/2024", total_guests: 3403983, total_revenue: 1185957039656 },
  { year: 2024, month: 10, label: "T10/2024", total_guests: 3610596, total_revenue: 1232244754746 },
  { year: 2024, month: 11, label: "T11/2024", total_guests: 3465845, total_revenue: 1169282286888 },
  { year: 2024, month: 12, label: "T12/2024", total_guests: 3849978, total_revenue: 1343166039208 },
  // 2025 — Mature phase
  { year: 2025, month: 1,  label: "T01/2025", total_guests: 3564493, total_revenue: 1380846129403 },
  { year: 2025, month: 2,  label: "T02/2025", total_guests: 3742968, total_revenue: 1315856878214 },
  { year: 2025, month: 3,  label: "T03/2025", total_guests: 3571658, total_revenue: 1236603316032 },
  { year: 2025, month: 4,  label: "T04/2025", total_guests: 3391272, total_revenue: 1182429664229 },
  { year: 2025, month: 5,  label: "T05/2025", total_guests: 3727614, total_revenue: 1295410589989 },
  { year: 2025, month: 6,  label: "T06/2025", total_guests: 3862440, total_revenue: 1315757050629 },
  { year: 2025, month: 7,  label: "T07/2025", total_guests: 3851232, total_revenue: 1331602470957 },
  { year: 2025, month: 8,  label: "T08/2025", total_guests: 4276262, total_revenue: 1489845880388 },
  { year: 2025, month: 9,  label: "T09/2025", total_guests: 3687242, total_revenue: 1279106142174 },
  { year: 2025, month: 10, label: "T10/2025", total_guests: 3780732, total_revenue: 1301369770836 },
  { year: 2025, month: 11, label: "T11/2025", total_guests: 3708511, total_revenue: 1255643159592 },
  { year: 2025, month: 12, label: "T12/2025", total_guests: 3806028, total_revenue: 1320047426283 },
  // 2026 — Current
  { year: 2026, month: 1,  label: "T01/2026", total_guests: 3817572, total_revenue: 1366211869253 },
  { year: 2026, month: 2,  label: "T02/2026", total_guests: 3794263, total_revenue: 1490879402370 },
  { year: 2026, month: 3,  label: "T03/2026", total_guests: 3577026, total_revenue: 1269002139226 },
];

// Merge macro + guest data
const MERGED_DATA = MACRO_DATA.map(m => {
  const guest = GUEST_DATA.find(g => g.year === m.year && g.month === m.month);
  return {
    ...m,
    total_guests: guest ? guest.total_guests : null,
    total_revenue: guest ? guest.total_revenue : null,
    has_guest_data: !!(guest && guest.total_guests !== null),
  };
});

// Column metadata for display
const COLUMN_META = {
  gold_sjc:       { label: "Giá Vàng SJC",      unit: "triệu/lượng",  color: "#FFD700", icon: "🥇" },
  gas_ron95:      { label: "Giá Xăng RON95",     unit: "đ/lít",        color: "#FF6B35", icon: "⛽" },
  deposit_rate:   { label: "LS Vietcombank 12T",  unit: "%/năm",        color: "#4ECDC4", icon: "🏦" },
  lending_rate:   { label: "LS Cho Vay BQ",      unit: "%/năm",        color: "#E74C3C", icon: "💰" },
  cpi:            { label: "CPI YoY",            unit: "%",            color: "#9B59B6", icon: "📊" },
  gdp_growth:     { label: "GDP Growth",         unit: "%",            color: "#27AE60", icon: "📈" },
  tourism_intl:   { label: "Du lịch quốc tế",   unit: "triệu khách",  color: "#3498DB", icon: "✈️" },
  total_guests:   { label: "Lượt Khách F&B",     unit: "khách",        color: "#E67E22", icon: "👥" },
  total_revenue:  { label: "Doanh Thu F&B",      unit: "VNĐ",          color: "#2ECC71", icon: "💵" },
};

// Pearson correlation helper
function pearsonCorrelation(x, y) {
  const n = x.length;
  if (n < 3) return null;
  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  const den = Math.sqrt(denX * denY);
  return den === 0 ? null : num / den;
}

// Calculate all correlations
function calculateCorrelations() {
  const macroKeys = ['gold_sjc', 'gas_ron95', 'deposit_rate', 'lending_rate', 'cpi', 'gdp_growth', 'tourism_intl'];
  const merged = MERGED_DATA.filter(d => d.has_guest_data);
  
  if (merged.length < 3) return [];
  
  const results = [];
  const guestValues = merged.map(d => d.total_guests);
  
  macroKeys.forEach(key => {
    const macroValues = merged.map(d => d[key]);
    const r = pearsonCorrelation(macroValues, guestValues);
    const absR = r !== null ? Math.abs(r) : 0;
    let strength = "Không đáng kể";
    if (absR >= 0.7) strength = "Mạnh";
    else if (absR >= 0.4) strength = "Trung bình";
    else if (absR >= 0.2) strength = "Yếu";
    
    results.push({
      key,
      label: COLUMN_META[key].label,
      icon: COLUMN_META[key].icon,
      r: r !== null ? r : 0,
      absR,
      strength,
      direction: r > 0 ? "Thuận (+)" : r < 0 ? "Nghịch (-)" : "N/A",
      n: merged.length,
    });
  });
  
  return results.sort((a, b) => b.absR - a.absR);
}

const CORRELATIONS = calculateCorrelations();
