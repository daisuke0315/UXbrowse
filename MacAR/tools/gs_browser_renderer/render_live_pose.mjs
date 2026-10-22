// Persistent, appearance-only 3DGS renderer for optional live pose QC.
// One Chrome/PLY load; newline-delimited JSON requests on stdin and replies
// on stdout. Rendering never participates in localization or PnP.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import puppeteer from '../../.tools/gs-browser/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(scriptDir, '../..');
const playcanvasPath = path.join(
    projectDir, '.tools/gs-browser/node_modules/playcanvas/build/playcanvas.mjs'
);
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const [plyArgument, widthArgument, heightArgument] = process.argv.slice(2);
const plyPath = plyArgument && path.resolve(plyArgument);
const width = Number(widthArgument);
const height = Number(heightArgument);
if (!plyPath || !fs.existsSync(plyPath) || !Number.isInteger(width) ||
        !Number.isInteger(height) || width < 64 || height < 64) {
    console.error('Usage: node render_live_pose.mjs SPLAT.ply WIDTH HEIGHT');
    process.exit(2);
}

const staticFiles = new Map([
    ['/', [path.join(scriptDir, 'index.html'), 'text/html; charset=utf-8']],
    ['/main.js', [path.join(scriptDir, 'main.js'), 'text/javascript; charset=utf-8']],
    ['/playcanvas.mjs', [playcanvasPath, 'text/javascript; charset=utf-8']],
    ['/asset.ply', [plyPath, 'application/octet-stream']]
]);
for (const stage of ['vert', 'frag']) {
    staticFiles.set(`/gsplat-${stage}.js`, [path.join(projectDir,
        `.tools/gs-browser/node_modules/playcanvas/build/playcanvas/src/scene/shader-lib/glsl/chunks/gsplat/${stage}/gsplat.js`),
        'text/javascript; charset=utf-8']);
}
const server = http.createServer((request, response) => {
    const pathname = new URL(request.url, 'http://127.0.0.1').pathname;
    if (pathname === '/favicon.ico') {
        response.writeHead(204);
        response.end();
        return;
    }
    const entry = staticFiles.get(pathname);
    if (!entry) {
        response.writeHead(404);
        response.end('Not found');
        return;
    }
    const [filename, contentType] = entry;
    try {
        const stat = fs.statSync(filename);
        response.writeHead(200, {
            'Content-Type': contentType,
            'Content-Length': stat.size,
            'Cache-Control': 'no-store'
        });
        fs.createReadStream(filename).pipe(response);
    } catch (error) {
        response.writeHead(500);
        response.end(String(error));
    }
});

const send = value => process.stdout.write(`${JSON.stringify(value)}\n`);
let browser;
try {
    await new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(0, '127.0.0.1', resolve);
    });
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('Renderer server failed');
    const query = new URLSearchParams({
        width: String(width),
        height: String(height),
        fov: '65',
        geometry: '0'
    });
    browser = await puppeteer.launch({
        executablePath: chromePath,
        headless: true,
        args: [
            '--disable-gpu-sandbox',
            '--use-angle=metal',
            '--enable-unsafe-webgpu',
            '--hide-scrollbars'
        ]
    });
    const page = await browser.newPage();
    await page.setViewport({ width, height, deviceScaleFactor: 1 });
    page.on('pageerror', error => console.error(error.stack || error));
    await page.goto(`http://127.0.0.1:${address.port}/?${query}`, {
        waitUntil: 'domcontentloaded', timeout: 120000
    });
    await page.waitForFunction(() => window.renderReady === true || window.renderError, {
        timeout: 120000, polling: 100
    });
    const ready = await page.evaluate(() => ({
        ready: window.renderReady,
        error: window.renderError
    }));
    if (!ready.ready) throw new Error(`3DGS load failed: ${ready.error}`);
    const canvas = await page.$('#application-canvas');
    if (!canvas) throw new Error('Gaussian canvas missing');
    send({ ready: true, width, height });

    const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
    for await (const line of input) {
        let request;
        try {
            request = JSON.parse(line);
            if (request.command === 'close') break;
            const vector = value => Array.isArray(value) && value.length === 3 &&
                value.every(Number.isFinite);
            if (!vector(request.eye) || !vector(request.target) ||
                    !vector(request.up) || !Number.isFinite(request.fov) ||
                    request.fov <= 1 || request.fov >= 179) {
                throw new Error('Invalid camera pose or FOV');
            }
            const started = performance.now();
            const settleFrames = Number.isInteger(request.settle_frames)
                ? Math.max(1, Math.min(8, request.settle_frames)) : 1;
            await page.evaluate((eye, target, up, fov, frames) =>
                window.setCamera(eye, target, up, fov, frames),
                request.eye, request.target, request.up, request.fov, settleFrames);
            const cameraMs = performance.now() - started;
            const png = await canvas.screenshot({ type: 'png' });
            send({
                id: request.id,
                png: png.toString('base64'),
                render_ms: Math.round(performance.now() - started),
                camera_ms: Math.round(cameraMs),
                screenshot_ms: Math.round(performance.now() - started - cameraMs)
            });
        } catch (error) {
            send({ id: request?.id, error: String(error.stack || error) });
        }
    }
} catch (error) {
    send({ ready: false, error: String(error.stack || error) });
    process.exitCode = 1;
} finally {
    if (browser) await browser.close();
    server.closeAllConnections?.();
    await new Promise(resolve => server.close(resolve));
}
