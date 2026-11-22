const { chromium } = require('playwright');

async function capture() {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 950 } });
  const page = await context.newPage();

  console.log("Navigating to http://localhost:8501 ...");
  await page.goto("http://localhost:8501", { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(5000);

  // 1. KPI & Traffic Tab
  console.log("Capturing dashboard_kpi.png ...");
  await page.screenshot({ path: "docs/screenshots/dashboard_kpi.png", fullPage: true });

  const tabButtons = await page.$$('button[data-baseweb="tab"], [role="tab"]');
  console.log("Found tabs count:", tabButtons.length);
  for (let i = 0; i < tabButtons.length; i++) {
    const text = await tabButtons[i].innerText();
    console.log(`Tab ${i}: "${text}"`);
  }

  // 2. Click Tab 1 (SHAP)
  if (tabButtons.length > 1) {
    console.log("Clicking SHAP tab (index 1)...");
    await tabButtons[1].click();
    await page.waitForTimeout(4000);
    console.log("Capturing dashboard_shap.png ...");
    await page.screenshot({ path: "docs/screenshots/dashboard_shap.png", fullPage: true });
  }

  // 3. Click Tab 2 (Pattern Memory)
  if (tabButtons.length > 2) {
    console.log("Clicking Pattern Memory tab (index 2)...");
    await tabButtons[2].click();
    await page.waitForTimeout(4000);
    console.log("Capturing dashboard_patterns.png ...");
    await page.screenshot({ path: "docs/screenshots/dashboard_patterns.png", fullPage: true });
  }

  // 4. Click Tab 3 (Regulatory / SLA)
  if (tabButtons.length > 3) {
    console.log("Clicking Regulatory Compliance tab (index 3)...");
    await tabButtons[3].click();
    await page.waitForTimeout(4000);
    console.log("Capturing dashboard_sla.png ...");
    await page.screenshot({ path: "docs/screenshots/dashboard_sla.png", fullPage: true });
  }

  await browser.close();
  console.log("All screenshots captured successfully!");
}

capture().catch(err => {
  console.error("Error capturing screenshots:", err);
  process.exit(1);
});
