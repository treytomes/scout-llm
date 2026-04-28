const { app, BrowserWindow, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const VENV_PYTHON = path.join(PROJECT_ROOT, ".venv", "bin", "python");
const PORT = 8000;
const READY_POLL_MS = 250;
const READY_TIMEOUT_MS = 30_000;

let win = null;
let server = null;

// ── Server lifecycle ──────────────────────────────────────────────────────────

function startServer() {
    server = spawn(
        VENV_PYTHON,
        ["-m", "uvicorn", "app:app", "--app-dir", path.join(PROJECT_ROOT, "src", "server"), "--port", String(PORT)],
        {
            cwd: PROJECT_ROOT,
            // Inherit stdout/stderr so logs appear in the Electron console
            stdio: ["ignore", "pipe", "pipe"],
            env: { ...process.env },
        }
    );

    server.stdout.on("data", (d) => process.stdout.write(`[scout] ${d}`));
    server.stderr.on("data", (d) => process.stderr.write(`[scout] ${d}`));

    server.on("exit", (code) => {
        if (code !== 0 && code !== null) {
            console.error(`Scout server exited with code ${code}`);
        }
    });
}

function stopServer() {
    if (server) {
        server.kill();
        server = null;
    }
}

function waitForServer(timeout) {
    return new Promise((resolve, reject) => {
        const deadline = Date.now() + timeout;

        function probe() {
            http.get(`http://localhost:${PORT}/`, (res) => {
                res.resume();
                resolve();
            }).on("error", () => {
                if (Date.now() >= deadline) {
                    reject(new Error(`Scout server did not start within ${timeout}ms`));
                } else {
                    setTimeout(probe, READY_POLL_MS);
                }
            });
        }

        probe();
    });
}

// ── Window ────────────────────────────────────────────────────────────────────

function createWindow() {
    win = new BrowserWindow({
        width: 1280,
        height: 900,
        minWidth: 800,
        minHeight: 600,
        title: "Scout",
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
        },
    });

    win.loadURL(`http://localhost:${PORT}/`);

    // Open external links in the system browser, not Electron
    win.webContents.setWindowOpenHandler(({ url }) => {
        if (!url.startsWith(`http://localhost:${PORT}`)) {
            shell.openExternal(url);
            return { action: "deny" };
        }
        return { action: "allow" };
    });

    win.on("closed", () => { win = null; });
}

// ── App lifecycle ─────────────────────────────────────────────────────────────

app.whenReady().then(async () => {
    startServer();

    try {
        await waitForServer(READY_TIMEOUT_MS);
    } catch (err) {
        console.error(err.message);
        app.quit();
        return;
    }

    createWindow();

    app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on("window-all-closed", () => {
    stopServer();
    if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => stopServer());
