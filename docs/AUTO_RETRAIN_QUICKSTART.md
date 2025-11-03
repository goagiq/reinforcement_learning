# Auto-Retrain Quickstart

## 🚀 Get Started in 3 Steps

### 1️⃣ Install Dependency
```bash
.venv\Scripts\pip install watchdog
```

### 2️⃣ Configure Settings (UI)
- Open **Settings** panel
- Enter NT8 path: `C:\Users\sovan\Documents\NinjaTrader 8\export`
- ✅ Check **"Automatic Retraining"**
- Click **Save**

### 3️⃣ Restart Backend
```bash
# Stop current backend (Ctrl+C)
# Start fresh
.venv\Scripts\python src\api_server.py
```

**Done!** You should see:
```
✅ Auto-retrain monitoring started on: C:\Users\sovan\Documents\NinjaTrader 8\export
```

---

## 🧪 Test It

Export a CSV from NT8 to your export folder:
```
C:\Users\sovan\Documents\NinjaTrader 8\export\ES_1min.csv
```

Watch backend console:
```
📁 New file detected: ES_1min.csv
🚀 Triggering retrain for 1 new file(s)
```

---

## ⚙️ Settings

**Enable/Disable:**
- Settings → "Automatic Retraining" checkbox

**Change Path:**
- Settings → "NT8 Data Folder Path"

**No Restart Needed:** Settings save immediately.

---

## ❓ Troubleshooting

**Monitor not starting:**
```bash
# Check backend logs for:
⚠️  Could not initialize auto-retrain monitor
```

**Common fixes:**
1. ✅ Install watchdog: `pip install watchdog`
2. ✅ Path exists and is readable
3. ✅ Path entered correctly in settings

**No triggers:**
1. ✅ File is CSV or TXT format
2. ✅ Wait 30 seconds after file appears
3. ✅ File is not being written (complete)

---

## 📋 Status Check

**Backend Console:**
```
✅ Auto-retrain monitoring started on: [PATH]
```

**Settings API:**
```bash
curl http://localhost:8200/api/settings/get
```

---

**That's it!** Drop CSV files and watch them auto-train! 🎉

