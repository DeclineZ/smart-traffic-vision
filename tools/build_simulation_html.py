import json
from pathlib import Path

art_dir = Path("/Users/febbuary/.gemini/antigravity/brain/f381bf8f-9b87-4c63-95c6-0afed9036475")

with open(art_dir / "images_b64.json", "r") as f:
    b64_data = json.load(f)

clear_b64 = b64_data["clear"]
rain_b64 = b64_data["rain"]

template = """<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <title>Smart Traffic Weather Simulation</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    @keyframes scanline {
      0% { transform: translateY(0%); }
      100% { transform: translateY(1000%); }
    }
    .scanner-line {
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.8), transparent);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
      animation: scanline 4s linear infinite;
    }
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-4 font-sans">
  <div class="max-w-4xl mx-auto bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-6 shadow-xl space-y-6">
    
    <!-- Header -->
    <div class="flex flex-wrap items-center justify-between gap-4 border-b border-[var(--border)] pb-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="inline-block w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></span>
          <h1 class="text-xl font-bold tracking-tight text-[var(--foreground)]">Smart Traffic Vision: Simulation Monitor</h1>
        </div>
        <p class="text-sm text-[var(--muted-foreground)] mt-0.5">ระบบตรวจจับสภาพอากาศและผิวถนนเปียกแบบ Real-time ด้วย YOLOv8-cls</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs px-2.5 py-1 rounded-md bg-[var(--sidebar)] border border-[var(--border)] font-mono text-[var(--muted-foreground)]">Model: yolov8n-cls (2.8MB)</span>
        <span class="text-xs px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-500 border border-emerald-500/30 font-semibold">Ready</span>
      </div>
    </div>

    <!-- Scenario Switcher Buttons -->
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <button id="btn-clear" onclick="setScenario('clear')" class="flex items-center justify-between p-3.5 rounded-xl border-2 border-emerald-500 bg-emerald-500/10 text-left transition-all hover:scale-[1.01]">
        <div>
          <div class="font-semibold text-sm flex items-center gap-2 text-emerald-400">
            <span>☀️ กล้อง 03: ถนนแห้ง (Dry Road)</span>
          </div>
          <div class="text-xs text-[var(--muted-foreground)] mt-0.5">สภาพแสงปกติ พื้นแห้ง เส้นจราจรชัดเจน</div>
        </div>
        <span class="text-xs font-bold uppercase px-2 py-0.5 rounded bg-emerald-500 text-slate-950">Active</span>
      </button>

      <button id="btn-rain" onclick="setScenario('rain')" class="flex items-center justify-between p-3.5 rounded-xl border border-[var(--border)] bg-[var(--sidebar)] text-left transition-all hover:scale-[1.01]">
        <div>
          <div class="font-semibold text-sm flex items-center gap-2 text-sky-400">
            <span>🌧️ แยกกรุงเทพฯ: ฝนตก/ถนนเปียก (Wet Road)</span>
          </div>
          <div class="text-xs text-[var(--muted-foreground)] mt-0.5">น้ำขัง แสงสะท้อนบนพื้น มีคนกางร่ม</div>
        </div>
        <span class="text-xs font-bold uppercase px-2 py-0.5 rounded bg-slate-700 text-slate-300">Click</span>
      </button>
    </div>

    <!-- Live Simulation Display -->
    <div class="relative rounded-xl overflow-hidden border border-[var(--border)] bg-slate-950 aspect-video flex items-center justify-center group shadow-inner">
      <img id="view-image" src="data:image/jpeg;base64,__CLEAR_B64__" alt="CCTV Stream" class="w-full h-full object-cover select-none">
      <div class="scanner-line"></div>

      <!-- Live Dynamic HUD Panel -->
      <div class="absolute top-4 left-4 right-4 sm:right-auto sm:w-96 bg-slate-950/85 backdrop-blur-md rounded-xl p-4 border border-slate-700/80 shadow-2xl space-y-2.5 pointer-events-none text-white">
        
        <div class="flex items-center justify-between">
          <span class="text-xs font-semibold tracking-wider text-slate-400">WEATHER STATUS</span>
          <span id="badge-status" class="px-2.5 py-0.5 rounded-md text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">CLEAR (ถนนแห้ง)</span>
        </div>

        <div class="border-t border-slate-800 pt-2">
          <div class="text-xs text-slate-400">LANE CONTROLLER MODE:</div>
          <div id="lane-mode" class="text-sm font-bold text-emerald-300 mt-0.5">STANDARD DRY ROAD MODE</div>
        </div>

        <div class="space-y-1 text-xs">
          <div class="flex justify-between text-slate-300">
            <span>Instant Prediction:</span>
            <span id="pred-conf" class="font-mono font-semibold text-emerald-400">CLEAR (100.0%)</span>
          </div>
          <div class="flex justify-between text-slate-400">
            <span>Inference Latency:</span>
            <span id="pred-latency" class="font-mono">1.8 ms (Apple M4)</span>
          </div>
        </div>

        <!-- Smoothing Buffer Bar -->
        <div class="space-y-1 pt-1">
          <div class="flex justify-between text-[11px] text-slate-400">
            <span>Smoothing Window (Rain votes):</span>
            <span id="buffer-text" class="font-mono font-bold text-slate-200">0/10 (0%)</span>
          </div>
          <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
            <div id="buffer-bar" class="h-full bg-emerald-500 rounded-full transition-all duration-300" style="width: 0%"></div>
          </div>
        </div>

      </div>

      <!-- Live Timestamp Footer -->
      <div class="absolute bottom-3 left-3 text-[11px] font-mono bg-slate-950/70 backdrop-blur px-2.5 py-1 rounded text-slate-300 border border-slate-800">
        FEED: <span id="feed-name">CAM03_EAST_TH</span> | FPS: 30.0 | REC: <span class="text-red-500">● LIVE</span>
      </div>
    </div>

    <!-- Interactive Simulation Controls -->
    <div class="bg-[var(--sidebar)] border border-[var(--border)] rounded-xl p-4 space-y-3">
      <div class="flex items-center justify-between">
        <span class="text-xs font-bold uppercase tracking-wider text-[var(--muted-foreground)]">🎮 ทดลองสลับสถานะจำลอง (Interactive Transition)</span>
        <span id="sim-status" class="text-xs text-slate-400">พร้อมทดสอบ</span>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
        <button onclick="triggerSuddenRain()" class="py-2.5 px-3 rounded-lg bg-sky-600 hover:bg-sky-500 text-white font-medium transition shadow-sm">
          🌧️ จำลอง: ฝนตกกะทันหัน
        </button>
        <button onclick="triggerRainStop()" class="py-2.5 px-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition shadow-sm">
          ☀️ จำลอง: ฝนหยุด & ถนนแห้ง
        </button>
        <button onclick="toggleRandomSpike()" class="py-2.5 px-3 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 font-medium transition">
          ⚡ จำลอง: แสงสะท้อนชั่วคราว (Flicker)
        </button>
      </div>
      <p class="text-[11px] text-[var(--muted-foreground)] leading-relaxed">
        *ปุ่มจำลองด้านบนจะแสดงให้เห็นว่า <b>Sliding Window Buffer</b> ค่อยๆ นับสะสมสถานะตามเวลาอย่างไร เพื่อป้องกันไม่ให้ระบบ Lane Detection สวิตช์ไปมาแบบกระตุก
      </p>
    </div>

  </div>

  <script>
    const b64Clear = "data:image/jpeg;base64,__CLEAR_B64__";
    const b64Rain = "data:image/jpeg;base64,__RAIN_B64__";

    let currentMode = "clear";
    let windowBuffer = 0;
    const windowMax = 10;
    let animInterval = null;

    function setScenario(mode) {
      currentMode = mode;
      clearInterval(animInterval);

      const btnClear = document.getElementById("btn-clear");
      const btnRain = document.getElementById("btn-rain");
      const viewImg = document.getElementById("view-image");
      const badgeStatus = document.getElementById("badge-status");
      const laneMode = document.getElementById("lane-mode");
      const predConf = document.getElementById("pred-conf");
      const predLatency = document.getElementById("pred-latency");
      const bufferText = document.getElementById("buffer-text");
      const bufferBar = document.getElementById("buffer-bar");
      const feedName = document.getElementById("feed-name");

      if (mode === "clear") {
        viewImg.src = b64Clear;
        feedName.textContent = "CAM03_EAST_TH";
        btnClear.className = "flex items-center justify-between p-3.5 rounded-xl border-2 border-emerald-500 bg-emerald-500/10 text-left transition-all";
        btnRain.className = "flex items-center justify-between p-3.5 rounded-xl border border-[var(--border)] bg-[var(--sidebar)] text-left transition-all";
        
        badgeStatus.textContent = "CLEAR (ถนนแห้ง)";
        badgeStatus.className = "px-2.5 py-0.5 rounded-md text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";

        laneMode.textContent = "STANDARD DRY ROAD MODE";
        laneMode.className = "text-sm font-bold text-emerald-300 mt-0.5";

        predConf.textContent = "CLEAR (100.0%)";
        predConf.className = "font-mono font-semibold text-emerald-400";
        predLatency.textContent = "1.8 ms (Apple M4)";

        windowBuffer = 0;
        bufferText.textContent = "0/10 (0%)";
        bufferBar.style.width = "0%";
        bufferBar.className = "h-full bg-emerald-500 rounded-full transition-all duration-300";
      } else {
        viewImg.src = b64Rain;
        feedName.textContent = "BKK_ZEBRA_JUNCTION_TH";
        btnRain.className = "flex items-center justify-between p-3.5 rounded-xl border-2 border-rose-500 bg-rose-500/10 text-left transition-all";
        btnClear.className = "flex items-center justify-between p-3.5 rounded-xl border border-[var(--border)] bg-[var(--sidebar)] text-left transition-all";

        badgeStatus.textContent = "RAINY / WET (ถนนเปียกน้ำ)";
        badgeStatus.className = "px-2.5 py-0.5 rounded-md text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40";

        laneMode.textContent = "ADAPTIVE WET SENSITIVITY MODE";
        laneMode.className = "text-sm font-bold text-amber-300 mt-0.5";

        predConf.textContent = "RAINY (100.0%)";
        predConf.className = "font-mono font-semibold text-rose-400";
        predLatency.textContent = "2.1 ms (Apple M4)";

        windowBuffer = 10;
        bufferText.textContent = "10/10 (100%)";
        bufferBar.style.width = "100%";
        bufferBar.className = "h-full bg-rose-500 rounded-full transition-all duration-300";
      }
    }

    function triggerSuddenRain() {
      clearInterval(animInterval);
      setScenario("clear");
      document.getElementById("view-image").src = b64Rain;
      document.getElementById("feed-name").textContent = "BKK_ZEBRA_JUNCTION_TH";
      document.getElementById("sim-status").textContent = "กำลังสะสมตัวอย่างฝนตก...";

      let step = 0;
      animInterval = setInterval(() => {
        step++;
        windowBuffer = step;
        const pct = (step / windowMax) * 100;
        document.getElementById("buffer-text").textContent = step + "/" + windowMax + " (" + pct.toFixed(0) + "%)";
        document.getElementById("buffer-bar").style.width = pct + "%";
        document.getElementById("pred-conf").textContent = "RAINY (99.8%)";
        document.getElementById("pred-conf").className = "font-mono font-semibold text-rose-400";

        if (step >= 6) {
          document.getElementById("badge-status").textContent = "RAINY / WET (ถนนเปียกน้ำ)";
          document.getElementById("badge-status").className = "px-2.5 py-0.5 rounded-md text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40";
          document.getElementById("lane-mode").textContent = "ADAPTIVE WET SENSITIVITY MODE";
          document.getElementById("lane-mode").className = "text-sm font-bold text-amber-300 mt-0.5";
          document.getElementById("buffer-bar").className = "h-full bg-rose-500 rounded-full transition-all duration-300";
        }

        if (step >= windowMax) {
          clearInterval(animInterval);
          document.getElementById("sim-status").textContent = "สลับสู่โหมดถนนเปียกเรียบร้อย (Active)";
        }
      }, 300);
    }

    function triggerRainStop() {
      clearInterval(animInterval);
      setScenario("rain");
      document.getElementById("view-image").src = b64Clear;
      document.getElementById("feed-name").textContent = "CAM03_EAST_TH";
      document.getElementById("sim-status").textContent = "ฝนหยุด กำลังรอถนนแห้งสนิท...";

      let step = 10;
      animInterval = setInterval(() => {
        step--;
        windowBuffer = step;
        const pct = (step / windowMax) * 100;
        document.getElementById("buffer-text").textContent = step + "/" + windowMax + " (" + pct.toFixed(0) + "%)";
        document.getElementById("buffer-bar").style.width = pct + "%";
        document.getElementById("pred-conf").textContent = "CLEAR (100.0%)";
        document.getElementById("pred-conf").className = "font-mono font-semibold text-emerald-400";

        if (step <= 3) {
          document.getElementById("badge-status").textContent = "CLEAR (ถนนแห้ง)";
          document.getElementById("badge-status").className = "px-2.5 py-0.5 rounded-md text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
          document.getElementById("lane-mode").textContent = "STANDARD DRY ROAD MODE";
          document.getElementById("lane-mode").className = "text-sm font-bold text-emerald-300 mt-0.5";
          document.getElementById("buffer-bar").className = "h-full bg-emerald-500 rounded-full transition-all duration-300";
        }

        if (step <= 0) {
          clearInterval(animInterval);
          document.getElementById("sim-status").textContent = "สลับกลับสู่โหมดถนนแห้งปกติ (Active)";
        }
      }, 300);
    }

    function toggleRandomSpike() {
      document.getElementById("pred-conf").textContent = "RAINY (88.4%)";
      document.getElementById("pred-conf").className = "font-mono font-semibold text-rose-400";
      document.getElementById("buffer-text").textContent = "1/10 (10%)";
      document.getElementById("buffer-bar").style.width = "10%";
      document.getElementById("sim-status").textContent = "ตรวจพบแสงวาบ 1 เฟรม -> Smoothing กรองทิ้ง ไม่สลับโหมด!";

      setTimeout(() => {
        if (currentMode === "clear") {
          document.getElementById("pred-conf").textContent = "CLEAR (100.0%)";
          document.getElementById("pred-conf").className = "font-mono font-semibold text-emerald-400";
          document.getElementById("buffer-text").textContent = "0/10 (0%)";
          document.getElementById("buffer-bar").style.width = "0%";
          document.getElementById("sim-status").textContent = "ระบบยังคงอยู่ในโหมดเดิมอย่างเสถียร";
        }
      }, 1000);
    }
  </script>
</body>
</html>
"""

html_final = template.replace("__CLEAR_B64__", clear_b64).replace("__RAIN_B64__", rain_b64)

with open(art_dir / "simulation.html", "w", encoding="utf-8") as f:
    f.write(html_final)

print("[✔] Successfully built simulation.html in artifact directory!")
