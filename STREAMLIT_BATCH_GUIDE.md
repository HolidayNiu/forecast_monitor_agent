# 📊 Forecast Monitor Agent - Batch-First Interface

Your Streamlit app now **defaults to batch summary** as the main page, providing immediate portfolio-wide insights with optional drill-down to individual parts.

## 🚀 How to Run the App

```bash
/usr/local/opt/python@3.9/bin/python3.9 -m streamlit run app.py
```

## 📋 Navigation Modes

The app opens directly to **📊 Batch Summary** (main page). Use the sidebar to switch modes:

### 📊 Batch Summary (Main Page - Default)
- **What it shows**: Portfolio-wide forecast health at a glance
- **Perfect for**: Executive dashboards, team meetings, portfolio overviews
- **Key Features**:
  - Portfolio-wide metrics and KPIs
  - Interactive issue breakdown charts
  - Top problematic parts ranking
  - Downloadable CSV/JSON reports
  - Severity distribution visualization

### 🔍 Individual Part Analysis (Drill-Down Mode)
- **What it shows**: Deep-dive analysis of specific parts
- **Perfect for**: Investigating specific forecast issues, detailed analysis
- **Key Features**:
  - Time series plots with diagnostics
  - AI-powered explanations
  - Detailed diagnostic breakdown
  - Historical vs forecast comparisons

## 🎯 Batch Summary Features

### Tab 1: 📊 Summary
**Key Metrics Dashboard:**
- Total parts analyzed
- Parts with issues (count + percentage)
- Average risk score across portfolio
- Average issues per part

**Interactive Charts:**
- **Issue Breakdown Bar Chart**: Shows how many parts have each type of issue
- **Severity Distribution Pie Chart**: Visual breakdown of severity levels

**Issue Details Table:**
- Lists each issue type with affected part counts
- Shows sample part IDs for each issue

### Tab 2: 🔥 Top Issues
**Most Problematic Parts Table:**
- Top 10 parts ranked by risk score
- Shows severity level and specific problems
- Helps prioritize which parts need immediate attention

### Tab 3: 📁 Downloads
**Export Options:**
- **📊 CSV Report**: Complete diagnostic results for all parts
  - All diagnostic flags and confidence scores
  - Part metadata (historical means, volatility, etc.)
  - Ready for Excel analysis
  
- **🤖 JSON Summary**: LLM-ready structured data
  - Portfolio summary statistics
  - Top problematic parts
  - Actionable recommendations
  - Perfect for feeding into AI analysis

## 📈 Sample Batch Summary Output

When you select "📊 Batch Summary", you'll see something like:

```
📊 OVERVIEW METRICS
Total Parts: 120        Parts with Issues: 38 (31.7%)
Average Risk Score: 0.445    Avg Issues/Part: 1.2

📊 ISSUE BREAKDOWN
Trend Mismatch: 22 parts
Missing Seasonality: 15 parts  
Volatility Mismatch: 18 parts
Magnitude Mismatch: 8 parts

🔥 TOP PROBLEMATIC PARTS
Part ID    Risk Score    Issues    Severity    Problems
part_001   2.145        4         High        Trend Mismatch; Missing Seasonality; Volatility Mismatch; Magnitude Mismatch
part_017   1.923        3         High        Trend Mismatch; Missing Seasonality; Volatility Mismatch
...
```

## 🔄 Performance Features

- **⚡ Caching**: Batch analysis results are cached to avoid recomputation
- **🔄 Progress Indicators**: Shows "Running batch diagnostics..." while processing  
- **💾 Memory Efficient**: Uses single-worker processing optimized for Streamlit
- **📱 Responsive**: Charts and tables adapt to your screen size

## 💡 Usage Tips

1. **Start with Batch Summary** to get the big picture of your forecast health
2. **Identify top problematic parts** using the "Top Issues" tab
3. **Switch to Individual Analysis** to deep-dive into specific problematic parts
4. **Download reports** for offline analysis or sharing with stakeholders
5. **Use JSON exports** to feed insights into other AI/LLM systems

## 🎨 Visual Features

- **Color-coded severity**: Red (high), Orange (medium), Yellow (low), Green (none)
- **Interactive charts**: Hover for details, zoom, pan
- **Responsive design**: Works on desktop and tablet
- **Professional styling**: Clean, business-ready presentation

## 🛠️ Technical Notes

- **Compatibility**: Uses the same diagnostic functions as individual analysis
- **Data consistency**: Same thresholds and logic across batch and individual modes
- **Export formats**: CSV for spreadsheet analysis, JSON for programmatic use
- **Error handling**: Graceful degradation if batch processing fails

## 🚨 Troubleshooting

**If you see "Batch processing modules not available":**
- Make sure you're running from the project root directory
- Verify the `src/` folder structure exists
- Check that Python can import the batch modules

**If charts don't display:**
- Ensure plotly is installed: `/usr/local/opt/python@3.9/bin/python3.9 -m pip install plotly`

**For best performance:**
- Use the caching feature (results persist across page refreshes)
- For very large datasets (100+ parts), consider running batch analysis offline first

---

Your forecast monitoring agent now provides both detailed individual analysis and scalable portfolio-wide insights in one unified interface! 🎉